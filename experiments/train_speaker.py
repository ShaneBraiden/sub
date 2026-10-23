"""Train our speaker-embedding network for diarization (replaces pyannote).

python experiments/train_speaker.py [--epochs 12] [--steps 1500]
-> models/speaker/model.pt (best dev-clean speaker-verification EER) + models/speaker/metrics.json
   models/speaker/last.pt is the resumable checkpoint (every --save-every steps and every epoch end).
   The clustering threshold is tuned afterwards on synthetic dev conversations and written into
   model.pt by `python experiments/eval_diarize.py --stage tune`.

Data: LibriSpeech train-clean-100 (251 speakers, disjoint from dev-clean / test-clean speakers).
Each example is a random 2 s crop of a random utterance, read straight from the FLAC file, with
random gain and, half of the time, white noise at 5-30 dB SNR. The network (src/subfix/speaker.py)
is trained as a 251-way classifier with an additive angular margin softmax.
Model selection: equal error rate of cosine scores on 4000 same/different-speaker trials of 3 s
crops from dev-clean (40 unseen speakers).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from subfix.paths import MODELS_DIR  # noqa: E402
from subfix.speaker import build_aam_head, build_speaker_net  # noqa: E402
from train_aligner import manifest  # noqa: E402

SR = 16000


def speaker_of(row: dict) -> str:
    return Path(row["path"]).name.split("-")[0]


def read_crop(path: str, dur: float, seconds: float, rng) -> np.ndarray:
    """A random `seconds` crop of a 16 kHz FLAC (shorter files are repeated)."""
    import soundfile as sf
    n = int(seconds * SR)
    total = int(dur * SR)
    start = int(rng.integers(0, max(1, total - n)))
    wav, _ = sf.read(str(ROOT / path), start=start, frames=n, dtype="float32")
    return np.resize(wav, n) if len(wav) < n else wav


class Crops:
    """Map-style dataset of random crops: index -> (waveform, speaker id). Each index draws a fresh crop."""

    def __init__(self, rows: list[dict], spk_ids: dict[str, int], seconds: float, seed: int):
        self.rows, self.spk_ids, self.seconds, self.seed = rows, spk_ids, seconds, seed

    def __len__(self) -> int:
        return 10 ** 9

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, i))
        r = self.rows[int(rng.integers(len(self.rows)))]
        wav = read_crop(r["path"], r["dur"], self.seconds, rng)
        wav = wav * float(10 ** rng.uniform(-0.5, 0.3))                      # gain -10 .. +6 dB
        if rng.random() < 0.5:
            p = float(np.mean(wav ** 2)) + 1e-8
            snr = rng.uniform(5, 30)
            wav = wav + rng.normal(0, math.sqrt(p / 10 ** (snr / 10)), len(wav)).astype(np.float32)
        return wav.astype(np.float32), self.spk_ids[speaker_of(r)]


def collate(items):
    import torch
    return torch.from_numpy(np.stack([w for w, _ in items])), torch.tensor([s for _, s in items])


def eer(scores: np.ndarray, same: np.ndarray) -> float:
    """Equal error rate: where false acceptance = false rejection."""
    order = np.argsort(-scores)
    s = same[order].astype(float)
    tp = np.cumsum(s) / s.sum()                   # true positive rate as the threshold falls
    fp = np.cumsum(1 - s) / (1 - s).sum()
    k = int(np.argmin(np.abs((1 - tp) - fp)))
    return float(((1 - tp[k]) + fp[k]) / 2)


def make_trials(rows: list[dict], n: int, seed: int = 0) -> tuple[list, np.ndarray]:
    """n trials (half same speaker) as pairs of (path, dur, crop seed)."""
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(speaker_of(r), []).append(r)
    spk = sorted(by)
    pairs, same = [], []
    for k in range(n):
        a = rng.choice(spk)
        if k % 2 == 0:
            r1, r2 = rng.sample(by[a], 2)
        else:
            b = rng.choice([s for s in spk if s != a])
            r1, r2 = rng.choice(by[a]), rng.choice(by[b])
        pairs.append((r1, r2, k))
        same.append(k % 2 == 0)
    return pairs, np.array(same)


def eval_eer(model, trials, same, device: str, seconds: float = 3.0) -> float:
    import torch
    cache: dict = {}

    def emb(row, seed):
        key = (row["path"], seed)
        if key not in cache:
            wav = read_crop(row["path"], row["dur"], seconds, np.random.default_rng(seed))
            with torch.inference_mode():
                e = model(torch.from_numpy(wav)[None].to(device)).float().cpu().numpy()[0]
            cache[key] = e / max(np.linalg.norm(e), 1e-8)
        return cache[key]
    model.eval()
    scores = np.array([float(emb(r1, 2 * k) @ emb(r2, 2 * k + 1)) for r1, r2, k in trials])
    model.train()
    return round(eer(scores, same), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train-clean-100")
    ap.add_argument("--dev-split", default="dev-clean")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1500, help="steps per epoch")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(MODELS_DIR / "speaker"))
    args = ap.parse_args()
    import torch
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = [r for r in manifest(args.split, max_dur=40.0) if r["dur"] >= args.seconds]
    spk_ids = {s: k for k, s in enumerate(sorted({speaker_of(r) for r in rows}))}
    dev = [r for r in manifest(args.dev_split, max_dur=40.0) if r["dur"] >= 3.0]
    trials, same = make_trials(dev, args.trials)
    print(f"[spk] {len(rows)} train utterances, {len(spk_ids)} speakers; {len(trials)} dev trials "
          f"({len({speaker_of(r) for r in dev})} speakers)", flush=True)

    arch: dict = {}
    model = build_speaker_net(**arch).to(device)
    head = build_aam_head(192, len(spk_ids), margin=args.margin).to(device)
    print(f"[spk] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters (+ "
          f"{sum(p.numel() for p in head.parameters()) / 1e6:.2f}M classifier head, dropped after training)", flush=True)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    total = args.epochs * args.steps

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = min(1.0, (step - args.warmup) / max(1, total - args.warmup))
        return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    last = out_dir / "last.pt"
    step, best, history = 0, None, []
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best, history = ck["step"], ck["best"], ck["history"]
        print(f"[spk] resumed at step {step}", flush=True)

    def save_last():
        torch.save({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "best": best, "history": history}, last)

    # sample index = global step * batch + position, so a resumed run continues the same crop stream
    ds = Crops(rows, spk_ids, args.seconds, args.seed)
    xent = torch.nn.CrossEntropyLoss()
    model.train()
    while step < total:
        epoch = step // args.steps
        end = (epoch + 1) * args.steps
        sampler = [list(range(s * args.batch, (s + 1) * args.batch)) for s in range(step, end)]
        loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                                             num_workers=args.workers, persistent_workers=False)
        t1, tot, acc, nb = time.time(), 0.0, 0.0, 0
        for wav, lab in loader:
            wav, lab = wav.to(device, non_blocking=True), lab.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.no_grad():                 # fp32 features: STFT power overflows float16
                feats = model.frontend(wav)
                for b in range(len(feats)):       # one frequency mask + one time mask per example
                    f0, w = random.randrange(0, feats.shape[1] - 8), random.randrange(0, 9)
                    t0, tw = random.randrange(0, feats.shape[2] - 20), random.randrange(0, 21)
                    feats[b, f0:f0 + w] = 0.0
                    feats[b, :, t0:t0 + tw] = 0.0
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                h = model.layers(feats)
                w_ = torch.softmax(model.attn(h), dim=-1)
                mu = (h * w_).sum(-1)
                sd = torch.sqrt(((h ** 2) * w_).sum(-1) - mu ** 2 + 1e-5)
            pooled = torch.cat([mu, sd], 1).float()
            emb = model.emb_norm(model.embed(model.pool_norm(pooled)))
            logits = head(emb, lab)
            loss = xent(logits, lab)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            nb += 1
            tot += loss.item()
            acc += (logits.argmax(1) == lab).float().mean().item()
            if nb % 100 == 0:
                print(f"[spk] epoch {epoch + 1} step {step}/{total} loss {tot / nb:.3f} acc {acc / nb:.3f} "
                      f"lr {lr_at(step):.2e} ({time.time() - t1:.0f}s)", flush=True)
            if step % args.save_every == 0:
                save_last()
        m = {"epoch": epoch + 1, "step": step, "train_loss": round(tot / max(1, nb), 4),
             "train_acc": round(acc / max(1, nb), 4), "dev_eer": eval_eer(model, trials, same, device)}
        history.append(m)
        print(f"[spk] epoch {epoch + 1}: {m} ({time.time() - t1:.0f}s)", flush=True)
        if best is None or m["dev_eer"] <= best["dev_eer"]:
            best = m
            torch.save({"model": model.state_dict(), "arch": arch, "metrics": m, "speakers": len(spk_ids)},
                       out_dir / "model.pt")
        save_last()
        (out_dir / "metrics.json").write_text(json.dumps({"best": best, "history": history, "args": vars(args)},
                                                         indent=1), encoding="utf-8")
    print("[spk] done, best:", best, flush=True)


if __name__ == "__main__":
    main()

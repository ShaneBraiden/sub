"""Train our CTC acoustic model for forced alignment (replaces MMS-300M / wav2vec2-base-960h).

python experiments/train_aligner.py [--split train-clean-100] [--epochs 25]
-> models/aligner/model.pt (best dev word-onset error vs MFA) + models/aligner/metrics.json
   models/aligner/last.pt is the resumable checkpoint (saved every --save-every steps and at every
   epoch end; re-running the command continues from the exact batch).

Data: LibriSpeech train-clean-100 (100 h, 251 speakers; disjoint from the dev/test speakers).
Model selection: the mean |onset error| of word starts when force-aligning the reference text of
the libri-dev-clean benchmark clips, measured against the MFA silver references
(data/refs/*.words.json), compared at 1 ms resolution; ties go to the lower greedy CER on 300
dev-clean utterances. CTC emits a character slightly after its acoustic onset; the median lag on
those dev clips is stored with the model (`onset_shift`) and subtracted from every aligned word
time, and the selection metric is the error after that shift.
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

from subfix.acoustic import TOKENS, build_acoustic, greedy_decode, spec_augment, text_to_ids  # noqa: E402
from subfix.paths import DATA_DIR, MODELS_DIR, cache_dir  # noqa: E402

LIBRI = DATA_DIR / "raw" / "librispeech" / "LibriSpeech"


def manifest(split: str, max_dur: float = 20.0) -> list[dict]:
    """[{path, text, dur}] for a LibriSpeech split (cached)."""
    import soundfile as sf
    cache = cache_dir("train") / f"aligner_{split}.json"
    n_flac = sum(1 for _ in (LIBRI / split).glob("*/*/*.flac"))
    d = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    if d.get("n_flac") == n_flac:             # rebuilt whenever the extracted audio changes
        rows = d["rows"]
    else:
        rows = []
        for trans in sorted((LIBRI / split).glob("*/*/*.trans.txt")):
            for line in trans.read_text(encoding="utf-8").splitlines():
                uid, text = line.split(" ", 1)
                path = trans.parent / f"{uid}.flac"
                if path.exists():
                    rows.append({"path": str(path.relative_to(ROOT)), "text": text.lower(),
                                 "dur": round(sf.info(str(path)).duration, 3)})
        cache.write_text(json.dumps({"n_flac": n_flac, "rows": rows}), encoding="utf-8")
    return [r for r in rows if r["dur"] <= max_dur]


class Utterances:
    """Map-style dataset: index -> (waveform, target ids). Module-level so DataLoader workers can pickle it."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import soundfile as sf
        r = self.rows[i]
        wav, sr = sf.read(str(ROOT / r["path"]), dtype="float32")
        assert sr == 16000
        return wav, text_to_ids(r["text"].split())


def collate(items):
    import torch
    n = max(len(w) for w, _ in items)
    wav = torch.zeros(len(items), n)
    lens = torch.tensor([len(w) for w, _ in items])
    for k, (w, _) in enumerate(items):
        wav[k, :len(w)] = torch.from_numpy(w)
    tgt = torch.tensor([t for _, ids in items for t in ids], dtype=torch.long)
    tlen = torch.tensor([len(ids) for _, ids in items])
    return wav, lens, tgt, tlen


def make_batches(rows: list[dict], max_batch_s: float, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    order = sorted(range(len(rows)), key=lambda i: rows[i]["dur"] * (1 + 0.1 * rng.random()))
    out, cur, longest = [], [], 0.0
    for i in order:
        d = rows[i]["dur"]
        if cur and max(longest, d) * (len(cur) + 1) > max_batch_s:
            out.append(cur)
            cur, longest = [], 0.0
        cur.append(i)
        longest = max(longest, d)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def onset_eval(model, device: str, n_clips: int = 4, seed: int = 0) -> dict:
    """Force-align the reference words of libri-dev-clean clips; |start - MFA start| statistics."""
    from subfix.config import Config
    from subfix.preprocess import load_audio
    from subfix.subio import Word
    from subfix.timing import CTCAligner, word_times
    cfg = Config.load(use_tuned=False)
    aligner = CTCAligner(device=device, star_penalty=cfg.timing.star_penalty, model=model)
    rng = np.random.default_rng(seed)
    errs = []
    for ref in sorted((DATA_DIR / "refs").glob("libri-dev-clean-*.words.json"))[:n_clips]:
        clip = ref.name.replace("libri-dev-clean-", "").replace(".words.json", "")
        wav = DATA_DIR / "clips" / "libri-dev-clean" / f"{clip}.wav"
        gold = json.loads(ref.read_text(encoding="utf-8"))
        # provisional times = MFA times with 0.3 s noise (the pipeline has ASR times here)
        words = [Word(g["text"], g["start"] + float(rng.normal(0, 0.3)), g["end"] + float(rng.normal(0, 0.3)))
                 for g in gold]
        for w in words:
            w.end = max(w.end, w.start + 0.05)
        out, _ = word_times(words, str(wav), cfg, aligner=aligner, backend="ctc", audio=load_audio(str(wav)))
        errs.extend(o.start - g["start"] for o, g in zip(out, gold))
    e = np.array(errs)
    bias = float(np.median(e))
    c = e - bias          # after removing the constant CTC lag (stored with the model as onset_shift)
    model.train()
    return {"onset_raw_mean_abs_ms": round(float(np.mean(np.abs(e))) * 1000, 1),
            "onset_bias_ms": round(bias * 1000, 1),
            "onset_mean_abs_ms": round(float(np.mean(np.abs(c))) * 1000, 1),
            "onset_median_abs_ms": round(float(np.median(np.abs(c))) * 1000, 1),
            "within_50ms": round(float(np.mean(np.abs(c) <= 0.05)), 4),
            "within_100ms": round(float(np.mean(np.abs(c) <= 0.1)), 4)}


def cer_eval(model, dev_rows: list[dict], device: str) -> float:
    import torch
    from metrics.text_metrics import wer_cer
    ds = Utterances(dev_rows)
    refs, hyps = [], []
    model.eval()
    with torch.inference_mode():
        for k in range(0, len(ds), 16):
            items = [ds[i] for i in range(k, min(len(ds), k + 16))]
            wav, lens, _, _ = collate(items)
            lp, n = model(wav.to(device), lens.to(device))
            for b in range(len(items)):
                hyps.append(greedy_decode(lp[b, :int(n[b])].cpu().numpy()))
                refs.append(dev_rows[k + b]["text"])
    model.train()
    return round(wer_cer(refs, hyps)["cer"], 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["train-clean-100"])
    ap.add_argument("--dev-split", default="dev-clean")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--max-batch-s", type=float, default=320.0)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--n-dev-utts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only N training utterances (debugging)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(MODELS_DIR / "aligner"))
    args = ap.parse_args()
    import torch
    sys.path.insert(0, str(ROOT))
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = [r for s in args.split for r in manifest(s)][:args.limit]
    dev_rows = manifest(args.dev_split)
    random.Random(1).shuffle(dev_rows)
    dev_rows = sorted(dev_rows[:args.n_dev_utts], key=lambda r: r["dur"])
    hours = sum(r["dur"] for r in rows) / 3600
    print(f"[ctc] {len(rows)} train utterances ({hours:.1f} h), {len(dev_rows)} dev utterances", flush=True)

    net: dict = {}
    model = build_acoustic(**net).to(device)
    print(f"[ctc] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, vocab {len(TOKENS)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    steps_per_epoch = len(make_batches(rows, args.max_batch_s, 0))
    total = args.epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = min(1.0, (step - args.warmup) / max(1, total - args.warmup))
        return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    last = out_dir / "last.pt"
    epoch, batch_idx, step, best, history = 0, 0, 0, None, []
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        epoch, batch_idx, step, best, history = ck["epoch"], ck["batch_idx"], ck["step"], ck["best"], ck["history"]
        print(f"[ctc] resumed at epoch {epoch + 1}, batch {batch_idx}, step {step}", flush=True)

    def save_last():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "batch_idx": batch_idx, "step": step, "best": best, "history": history,
                    "net": net}, last)

    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    gen = torch.Generator().manual_seed(args.seed)
    ds = Utterances(rows)
    model.train()
    while epoch < args.epochs:
        bl = make_batches(rows, args.max_batch_s, args.seed * 1000 + epoch)
        loader = torch.utils.data.DataLoader(ds, batch_sampler=bl[batch_idx:], collate_fn=collate,
                                             num_workers=args.workers, persistent_workers=False)
        t1, tot, nb = time.time(), 0.0, 0
        for wav, lens, tgt, tlen in loader:
            wav, lens = wav.to(device, non_blocking=True), lens.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.no_grad():                # fp32: STFT power overflows float16
                feats, n_feat = model.frontend(wav, lens)
                feats = spec_augment(feats, n_feat, generator=gen)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                lp, n = model.forward_features(feats, n_feat)
            loss = ctc(lp.float().transpose(0, 1), tgt, n.cpu(), tlen)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            batch_idx += 1
            nb += 1
            tot += loss.item()
            if nb % 100 == 0:
                print(f"[ctc] epoch {epoch + 1} batch {batch_idx}/{len(bl)} loss {tot / nb:.3f} "
                      f"lr {lr_at(step):.2e} ({time.time() - t1:.0f}s)", flush=True)
            if step % args.save_every == 0:
                save_last()
        m = {"epoch": epoch + 1, "step": step, "train_loss": round(tot / max(1, nb), 4),
             "dev_cer": cer_eval(model, dev_rows, device)}
        m.update(onset_eval(model, device))
        history.append(m)
        print(f"[ctc] epoch {epoch + 1}: {m} ({time.time() - t1:.0f}s)", flush=True)
        # onset error at 1 ms resolution (it is flat after a few epochs), then character error rate
        if best is None or (round(m["onset_mean_abs_ms"]), m["dev_cer"]) < (round(best["onset_mean_abs_ms"]), best["dev_cer"]):
            best = m
            torch.save({"model": model.state_dict(), "net": net, "tokens": TOKENS, "metrics": m,
                        "onset_shift": -m["onset_bias_ms"] / 1000.0}, out_dir / "model.pt")
        epoch += 1
        batch_idx = 0
        save_last()
        (out_dir / "metrics.json").write_text(json.dumps({"best": best, "history": history, "args": vars(args)},
                                                         indent=1), encoding="utf-8")
    print("[ctc] done, best:", best, flush=True)


if __name__ == "__main__":
    main()

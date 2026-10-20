"""Train our from-scratch Transformer subtitle corrector (replaces vennify T5 / CoEdIT / T5-small).

python experiments/train_corrector.py [--epochs 15]
-> models/corrector/{model.pt, bpe.json, metrics.json}
   models/corrector/last.pt is the resumable checkpoint (re-running continues from it).

Training pairs (imp.md 3.3, RoBART/DARAG ideas): LibriSpeech-PC *train* sentences (punctuated,
cased; disjoint from dev/test books) corrupted with the text operations of tools/corrupt.py
(typos, homophones, dropped function words, duplicates, punctuation stripping, lowercasing) plus
real whisper-tiny confusions harvested from the *dev* clips. About a third of the inputs are left
clean so the model learns to copy correct text. Fresh noise is drawn every epoch. Dev: noisy
LibriSpeech-PC dev-clean sentences (fixed seed).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "experiments"))

from corpus import sentences  # noqa: E402
from subfix import text as T  # noqa: E402
from subfix.bpe import BOS, BPE, EOS, PAD  # noqa: E402
from subfix.paths import MODELS_DIR  # noqa: E402
from subfix.textfix.seq2seq import DEFAULT_NET, build_seq2seq  # noqa: E402


def load_sentences(splits=None, min_words: int = 4, max_words: int = 40) -> list[str]:
    sents = sentences(splits) if splits else sentences()
    return [s for s in sents if min_words <= len(s.split()) <= max_words]


def harvest_confusions(model: str = "tiny.en") -> dict[str, list[str]]:
    """canon(ref word) -> whisper-tiny substitutions observed on the dev clips (cached ASR only)."""
    from corrupt import asr_confusions
    from fetch_data import all_clips
    from subfix.asr import _cache_exists, transcribe
    from subfix.subio import tokenize
    conf: dict[str, list[str]] = {}
    for c in all_clips():
        if c["dataset"] not in ("libri-dev-clean", "ted-validation"):
            continue
        wav = str(ROOT / c["audio"])
        if not _cache_exists(wav, model):
            continue
        tiny = transcribe(wav, model=model)
        toks = tokenize(c["text"])
        for i, w in asr_confusions(toks, [x.text for x in tiny.words]).items():
            conf.setdefault(T.canon(toks[i]), []).append(T.norm(w))
    return conf


def noisify(sent: str, rng: random.Random, conf: dict[str, list[str]]) -> str:
    from corrupt import typo
    hom = T.homophone_groups()
    func = T.function_words()
    out = []
    for tok in sent.split():
        pre, core, suf = T.split_affixes(tok)
        r = rng.random()
        if r < 0.04 and sum(ch.isalpha() for ch in core) >= 3:
            out.append(pre + typo(core, rng) + suf)
        elif r < 0.07 and core.lower() in hom:
            new = rng.choice(hom[core.lower()])
            out.append(pre + (new.capitalize() if core[:1].isupper() else new) + suf)
        elif r < 0.11 and T.canon(tok) in conf:
            new = rng.choice(conf[T.canon(tok)])
            out.append(pre + (new.capitalize() if core[:1].isupper() else new) + suf)
        elif r < 0.13 and T.norm(tok) in func and not suf:
            continue
        elif r < 0.14:
            out.extend([pre + core, tok])
        else:
            out.append(tok)
    mode = rng.random()
    if mode < 0.3:
        out = [T.strip_trailing_punct(t) for t in out]
    elif mode < 0.6:
        out = [T.strip_trailing_punct(t) if T.trailing_punct(t) and rng.random() < 0.5 else t for t in out]
    if rng.random() < 0.3:
        out = [t.lower() for t in out]
    return " ".join(out) if out else sent


def make_pairs(sents: list[str], rng: random.Random, conf, clean_frac: float = 0.33) -> list[tuple[str, str]]:
    return [(s if rng.random() < clean_frac else noisify(s, rng, conf), s) for s in sents]


def batches(pairs_ids: list[tuple[list[int], list[int]]], max_tokens: int, rng: random.Random):
    order = sorted(range(len(pairs_ids)), key=lambda i: (len(pairs_ids[i][0]), rng.random()))
    out, cur, longest = [], [], 0
    for i in order:
        n = max(len(pairs_ids[i][0]), len(pairs_ids[i][1])) + 2
        if cur and max(longest, n) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def wer(refs: list[str], hyps: list[str]) -> float:
    from metrics.text_metrics import wer_cer
    return wer_cer(refs, hyps)["wer"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--n-dev", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only N training sentences (debugging)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(MODELS_DIR / "corrector"))
    args = ap.parse_args()
    import torch
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train = load_sentences()[:args.limit]
    conf = harvest_confusions()
    print(f"[corr] {len(train)} train sentences, {len(conf)} confusion entries ({time.time() - t0:.0f}s)", flush=True)
    bpe_path = out_dir / "bpe.json"
    if bpe_path.exists():
        bpe = BPE.load(bpe_path)
    else:
        bpe = BPE.train((w for s in train for w in s.split()), args.vocab_size, log=lambda m: print(m, flush=True))
        bpe.save(bpe_path)
    print(f"[corr] BPE vocabulary {len(bpe)} ({time.time() - t0:.0f}s)", flush=True)

    drng = random.Random(1234)
    dev_sents = load_sentences(("dev-clean",))
    drng.shuffle(dev_sents)
    dev = make_pairs(dev_sents[:args.n_dev], drng, conf, clean_frac=0.0)
    dev_clean = dev_sents[args.n_dev:args.n_dev + 500]

    net = dict(DEFAULT_NET)
    model = build_seq2seq(len(bpe), **net).to(device)
    print(f"[corr] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    est_steps = args.epochs * math.ceil(sum(len(bpe.encode(s)) + 2 for s in train[:20000]) / 20000
                                        * len(train) / (args.max_tokens * 0.8))

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = min(1.0, (step - args.warmup) / max(1, est_steps - args.warmup))
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    last = out_dir / "last.pt"
    start_epoch, step, best, history = 0, 0, None, []
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, step, best, history = ck["epoch"], ck["step"], ck["best"], ck["history"]
        print(f"[corr] resumed after epoch {start_epoch} (step {step})", flush=True)

    def correct(srcs: list[str], beams: int = 1) -> list[str]:
        model.eval()
        hyps = [""] * len(srcs)
        order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))
        for b in range(0, len(order), 64):
            idx = order[b:b + 64]
            ids = [bpe.encode(srcs[i])[:model.max_len - 1] + [EOS] for i in idx]
            L = max(map(len, ids))
            src = torch.full((len(ids), L), PAD, dtype=torch.long)
            for r, x in enumerate(ids):
                src[r, :len(x)] = torch.tensor(x)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                gen = model.generate(src.to(device), beams, min(model.max_len, L + 16))
            for i, g in zip(idx, gen):
                hyps[i] = bpe.decode(g)
        model.train()
        return hyps

    def evaluate() -> dict:
        srcs, refs = [s for s, _ in dev], [t for _, t in dev]
        hyps = correct(srcs)
        kept = correct(dev_clean)
        return {"wer_model": round(wer(refs, hyps), 4), "wer_copy": round(wer(refs, srcs), 4),
                "exact": round(sum(h == r for h, r in zip(hyps, refs)) / len(refs), 4),
                "clean_unchanged": round(sum(h == r for h, r in zip(kept, dev_clean)) / len(dev_clean), 4)}

    model.train()
    for epoch in range(start_epoch, args.epochs):
        erng = random.Random(args.seed * 1000 + epoch)
        t1 = time.time()
        pairs = make_pairs(train, erng, conf)
        ids = [(bpe.encode(s)[:model.max_len - 1], bpe.encode(t)[:model.max_len - 2]) for s, t in pairs]
        bl = batches(ids, args.max_tokens, erng)
        print(f"[corr] epoch {epoch + 1}: {len(bl)} batches (data {time.time() - t1:.0f}s)", flush=True)
        tot, n_tok = 0.0, 0
        for k, b in enumerate(bl):
            S = max(len(ids[i][0]) for i in b) + 1
            Tt = max(len(ids[i][1]) for i in b) + 2
            src = torch.full((len(b), S), PAD, dtype=torch.long)
            tgt = torch.full((len(b), Tt), PAD, dtype=torch.long)
            for r, i in enumerate(b):
                s_ids, t_ids = ids[i]
                src[r, :len(s_ids) + 1] = torch.tensor(s_ids + [EOS])
                tgt[r, :len(t_ids) + 2] = torch.tensor([BOS] + t_ids + [EOS])
            src, tgt = src.to(device), tgt.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(src, tgt[:, :-1])
            y = tgt[:, 1:]
            loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                                                     ignore_index=PAD, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            nt = int((y != PAD).sum())
            tot += loss.item() * nt
            n_tok += nt
            if (k + 1) % 500 == 0:
                print(f"[corr] epoch {epoch + 1} batch {k + 1}/{len(bl)} loss {tot / n_tok:.4f} "
                      f"lr {lr_at(step):.2e} ({time.time() - t1:.0f}s)", flush=True)
        m = evaluate()
        m.update(epoch=epoch + 1, step=step, train_loss=round(tot / max(1, n_tok), 4))
        history.append(m)
        print(f"[corr] epoch {epoch + 1}: {m} ({time.time() - t1:.0f}s)", flush=True)
        if best is None or m["wer_model"] < best["wer_model"]:
            best = m
            torch.save({"model": model.state_dict(), "net": net}, out_dir / "model.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1, "step": step,
                    "best": best, "history": history}, last)
        (out_dir / "metrics.json").write_text(json.dumps({"best": best, "history": history, "args": vars(args)},
                                                         indent=1), encoding="utf-8")
    print("[corr] done, best:", best, flush=True)


if __name__ == "__main__":
    main()

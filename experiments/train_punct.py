"""Train our BiLSTM punctuation tagger (replaces the FullStop XLM-R model).

python experiments/train_punct.py [--epochs 8]
-> models/punct/model.pt (best dev macro-F1) + models/punct/metrics.json
   models/punct/last.pt is the resumable checkpoint (re-running the command continues from it).

Data: LibriSpeech-PC train chapters (utterances joined in order, ~8.5M words), labels 0 . , ?
from each word's trailing mark. Every training window is lowercased with probability 0.5 so the
tagger works with and without casing. Dev: LibriSpeech-PC dev-clean + dev-other chapters.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from corpus import chapters  # noqa: E402
from subfix import text as T  # noqa: E402
from subfix.paths import MODELS_DIR, cache_dir  # noqa: E402
from subfix.textfix.punct import CASES, LABELS, MAX_CHARS, build_net, char_ids, label_of  # noqa: E402


def encode(splits, types: dict[str, int] | None = None):
    """Stream of (type id, case id, label id) over all chapters; `types` maps lowercase forms
    (new forms are added, so dev-only words keep their spelling for the character CNN)."""
    types = {} if types is None else types
    tid, case, lab = [], [], []
    for toks in chapters(splits):
        for tok in toks:
            form = T.norm(tok)
            if not form:
                m = label_of(tok)
                if m != "0" and lab and lab[-1] == 0:
                    lab[-1] = LABELS.index(m)
                continue
            tid.append(types.setdefault(form, len(types)))
            case.append(CASES.index(T.case_pattern(T.core(tok))))
            lab.append(LABELS.index(label_of(tok)))
    return types, np.array(tid, np.int32), np.array(case, np.int8), np.array(lab, np.int8)


def prepare(max_vocab: int):
    path = cache_dir("train") / "punct_data.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    types, tid, case, lab = encode(("train-clean-100", "train-clean-360", "train-other-500"))
    _, dtid, dcase, dlab = encode(("dev-clean", "dev-other"), types)
    forms = [None] * len(types)
    for f, i in types.items():
        forms[i] = f
    freq = Counter(tid.tolist())
    vocab = ["<pad>", "<unk>"] + [forms[i] for i, _ in freq.most_common(max_vocab)]
    vid = {w: i for i, w in enumerate(vocab)}
    type2word = np.array([vid.get(f, 1) for f in forms], np.int32)
    chars = np.zeros((len(forms), MAX_CHARS), np.int8)
    for i, f in enumerate(forms):
        ci = char_ids(f) or [1]
        chars[i, :len(ci)] = ci
    d = dict(vocab=np.array(vocab, dtype=object), type2word=type2word, chars=chars, tid=tid, case=case, lab=lab,
             dtid=dtid, dcase=dcase, dlab=dlab)
    np.savez(path, **d)
    return d


def f1_report(gold: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for k, name in enumerate(LABELS[1:], start=1):
        tp = int(np.sum((pred == k) & (gold == k)))
        p = tp / max(1, int(np.sum(pred == k)))
        r = tp / max(1, int(np.sum(gold == k)))
        out[name] = round(2 * p * r / max(p + r, 1e-9), 4)
    out["macro"] = round(float(np.mean([out[n] for n in LABELS[1:]])), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-vocab", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(MODELS_DIR / "punct"))
    args = ap.parse_args()
    import torch
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    d = prepare(args.max_vocab)
    vocab = list(d["vocab"])
    t2w = torch.from_numpy(d["type2word"].astype(np.int64))
    chars = torch.from_numpy(d["chars"].astype(np.int64))
    print(f"[punct] {len(d['tid'])} train / {len(d['dtid'])} dev words, vocab {len(vocab)}, "
          f"labels {np.bincount(d['lab'], minlength=4).tolist()} ({time.time() - t0:.0f}s)", flush=True)

    net_cfg = {}
    model = build_net(len(vocab), **net_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    L, Bsz = args.seq_len, args.batch_size
    n_seq = (len(d["tid"]) - L) // L
    steps_total = args.epochs * (n_seq // Bsz)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_total, pct_start=0.05)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    last = out_dir / "last.pt"
    start_epoch, best, history = 0, None, []
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best, history = ck["epoch"], ck["best"], ck["history"]
        print(f"[punct] resumed after epoch {start_epoch}", flush=True)

    tid = torch.from_numpy(d["tid"].astype(np.int64))
    case = torch.from_numpy(d["case"].astype(np.int64))
    lab = torch.from_numpy(d["lab"].astype(np.int64))

    def batch_tensors(idx: torch.Tensor, lower: torch.Tensor):
        t = tid[idx]
        c = case[idx].clone()
        c[lower.unsqueeze(1).expand_as(c) & (c > 0)] = CASES.index("lower")
        return t2w[t].to(device), c.to(device), chars[t].to(device), lab[idx].to(device)

    def evaluate() -> dict:
        model.eval()
        dt = torch.from_numpy(d["dtid"].astype(np.int64))
        dc = torch.from_numpy(d["dcase"].astype(np.int64))
        gold = d["dlab"].astype(np.int64)
        res = {}
        for name, lower in (("cased", False), ("lower", True)):
            preds = []
            with torch.inference_mode():
                for s in range(0, len(dt), 4096):
                    t = dt[s:s + 4096].unsqueeze(0)
                    c = dc[s:s + 4096].unsqueeze(0).clone()
                    if lower:
                        c[c > 0] = CASES.index("lower")
                    lg = model(t2w[t].to(device), c.to(device), chars[t].to(device))
                    preds.append(lg.argmax(-1).squeeze(0).cpu().numpy())
            res[name] = f1_report(gold, np.concatenate(preds))
        model.train()
        res["score"] = round((res["cased"]["macro"] + res["lower"]["macro"]) / 2, 4)
        return res

    gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        off = int(torch.randint(0, L, (1,), generator=gen))
        starts = off + torch.arange(n_seq) * L
        starts = starts[torch.randperm(n_seq, generator=gen)]
        t1, tot = time.time(), 0.0
        for k in range(n_seq // Bsz):
            st = starts[k * Bsz:(k + 1) * Bsz]
            idx = st.unsqueeze(1) + torch.arange(L).unsqueeze(0)
            lower = torch.rand(len(st), generator=gen) < 0.5
            w, c, ch, y = batch_tensors(idx, lower)
            logits = model(w, c, ch)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(LABELS)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item()
            if (k + 1) % 400 == 0:
                print(f"[punct] epoch {epoch + 1} step {k + 1}/{n_seq // Bsz} loss {tot / (k + 1):.4f} "
                      f"({time.time() - t1:.0f}s)", flush=True)
        m = evaluate()
        m["epoch"] = epoch + 1
        m["train_loss"] = round(tot / max(1, n_seq // Bsz), 4)
        history.append(m)
        print(f"[punct] epoch {epoch + 1}: {m}", flush=True)
        if best is None or m["score"] > best["score"]:
            best = m
            torch.save({"model": model.state_dict(), "vocab": vocab, "net": net_cfg, "labels": LABELS},
                       out_dir / "model.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "best": best, "history": history}, last)
    (out_dir / "metrics.json").write_text(json.dumps({"best": best, "history": history, "args": vars(args)},
                                                     indent=1), encoding="utf-8")
    print("[punct] done, best:", best, flush=True)


if __name__ == "__main__":
    main()

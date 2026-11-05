"""Train the neural fusion network (subfix.fusenet) on the data of experiments/fusion_data.py.

python experiments/train_fusion.py [--epochs 14] [--arch signals] [--no-ctc] [--out models/fusion]
-> models/fusion/model.pt (best dev score, with the dev-tuned thresholds) + models/fusion/metrics.json
   models/fusion/last.pt is the resumable checkpoint (re-running the command continues from it).

The loss is binary cross-entropy per candidate: did applying this edit lower the edit distance to the
reference? Thresholds and the best epoch are chosen on the benchmark dev clips, on the same
candidate-level F0.5 as experiments/tune_fusion.py with a small WER term: F0.5 is nearly flat around
its peak while WER is not, so the term picks the side of the plateau that leaves fewer errors.
"WER" below is the estimate after the fusion stage alone (errors of the corrupted subtitle minus
the gain of the accepted edits).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from fusion_data import FEATURES as FEATURE_DIR, WER_WEIGHT  # noqa: E402
from subfix.fusenet import CTC_FEATURES, FEATURE_INDEX, FEATURES, KINDS, MAX_CHARS, SCORED, build_net, char_ids  # noqa: E402
from subfix.paths import MODELS_DIR  # noqa: E402

OUT_OF_KIND = np.array([SCORED.index(k) if k in SCORED else 0 for k in KINDS], np.int64)
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


class Corpus:
    """Concatenated feature files of one group (train / dev), with global word-form ids."""

    def __init__(self, group: str, datasets: list[str] | None, forms: dict[str, int]):
        cols: dict[str, list] = {k: [] for k in ("kinds", "num", "gain", "rule", "scored", "s", "a")}
        self.seqs: list[dict] = []
        pos = 0
        for path in sorted((FEATURE_DIR / group).glob("*.npz")):
            with np.load(path) as z:
                dataset = str(z["dataset"])
                if datasets and dataset not in datasets:
                    continue
                remap = np.array([forms.setdefault(f, len(forms)) for f in z["forms"].tolist()], np.int32)
                for name in ("kinds", "num", "gain", "rule", "scored"):
                    cols[name].append(z[name])
                cols["s"].append(remap[z["s"]])
                cols["a"].append(remap[z["a"]])
                for n, seed, err, n_ref in zip(z["lengths"].tolist(), z["seeds"].tolist(), z["err"].tolist(),
                                               z["n_ref"].tolist()):
                    self.seqs.append({"start": pos, "end": pos + n, "seed": seed, "err": err, "n_ref": n_ref,
                                      "dataset": dataset, "clip": path.stem})
                    pos += n
        for name, parts in cols.items():
            setattr(self, name, np.concatenate(parts))
        self.kinds = self.kinds.astype(np.int64)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if not n1 or not n0:
        return float("nan")
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def f05(tp: float, fp: float, gold: float) -> float:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / gold if gold else 1.0
    return 1.25 * p * r / (0.25 * p + r) if p + r else 0.0


def summarize(accept: np.ndarray, gain: np.ndarray, err: float, n_ref: float) -> dict:
    tp, fp = int((accept & (gain > 0)).sum()), int((accept & (gain <= 0)).sum())
    return {"f05": round(f05(tp, fp, err), 4), "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(err, 1), 4), "wer": round(float(100 * (err - gain[accept].sum()) / n_ref), 3),
            "tp": tp, "fp": fp}


def group_masks(dev: Corpus, seed_zero: bool = False) -> dict:
    """libri / ted / all -> (units of those sequences, their errors, their reference words)."""
    out = {}
    for tag in ("libri", "ted", "all"):
        m = np.zeros(len(dev.kinds), bool)
        err = nref = 0
        for s in dev.seqs:
            if seed_zero and s["seed"] != 0:
                continue
            if tag == "all" or s["dataset"].startswith(tag):
                m[s["start"]:s["end"]] = True
                err += s["err"]
                nref += s["n_ref"]
        out[tag] = (m, err, nref)
    return out


def rule_numbers(dev: Corpus) -> dict:
    """The tuned rule on the dev sequences: its decisions were saved with the features."""
    accept, cand = dev.rule == 1, dev.rule >= 0
    return {f"{pre}{tag}": summarize(accept[m & cand], dev.gain[m & cand], err, nref)
            for pre, flag in (("", False), ("official_", True)) for tag, (m, err, nref) in group_masks(dev, flag).items()}


def threshold_counts(prob: np.ndarray, gain: np.ndarray, out: np.ndarray, base: np.ndarray) -> tuple:
    """Right edits, wrong edits and net gain for every combination of the per-kind thresholds (GRID),
    as arrays with one axis per kind. `base` = decisions that stay with the rule."""
    tp, fp, net = float((base & (gain > 0)).sum()), float((base & (gain <= 0)).sum()), float(gain[base].sum())
    for k in range(len(SCORED)):
        p, g = prob[out == k], gain[out == k]
        acc = p[None] > GRID[:, None]
        shape = [1] * len(SCORED)
        shape[k] = len(GRID)
        tp = tp + (acc & (g > 0)).sum(1).reshape(shape)
        fp = fp + (acc & (g <= 0)).sum(1).reshape(shape)
        net = net + (acc * g).sum(1).reshape(shape)
    return tp, fp, net


def tune_thresholds(prob: np.ndarray, gain: np.ndarray, out: np.ndarray, base: np.ndarray, err: float,
                    n_ref: float) -> dict:
    """Per-kind thresholds, by exhaustive search for the best selection score."""
    tp, fp, net = threshold_counts(prob, gain, out, base)
    precision, recall = tp / np.maximum(tp + fp, 1), tp / max(err, 1)
    score = 1.25 * precision * recall / np.maximum(0.25 * precision + recall, 1e-9) - WER_WEIGHT * 100 * (err - net) / n_ref
    best = np.unravel_index(np.argmax(score), score.shape)
    return {name: float(GRID[best[k]]) for k, name in enumerate(SCORED)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--arch", default="signals", choices=["signals", "words", "mlp", "linear"])
    ap.add_argument("--no-ctc", action="store_true", help="ablation: hide the CTC audio evidence")
    ap.add_argument("--datasets", nargs="*", default=None, help="train only on these sets (dev-clean ted train-clean-100)")
    ap.add_argument("--train-group", default="train", help="feature folder to train on (train_no_<type> = fusion_data.py --holdout)")
    ap.add_argument("--seq-len", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-vocab", type=int, default=8000)
    ap.add_argument("--word-dropout", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--ted-share", type=float, default=0.25, help="share of training windows taken from TED talks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(MODELS_DIR / "fusion"))
    args = ap.parse_args()
    import torch
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    forms: dict[str, int] = {"": 0}
    train = Corpus(args.train_group, args.datasets, forms)
    dev = Corpus("dev", None, forms)
    counts = np.bincount(np.concatenate([train.s, train.a]), minlength=len(forms))
    counts[0] = 0
    order = np.argsort(-counts)[:args.max_vocab]
    form_list = list(forms)
    vocab = ["<pad>", "<unk>"] + [form_list[i] for i in order if counts[i] > 0]
    vid = {w: i for i, w in enumerate(vocab)}
    type2word = np.array([vid.get(f, 1) if f else 0 for f in form_list], np.int64)
    chars = np.zeros((len(form_list), MAX_CHARS), np.int64)
    for i, f in enumerate(form_list):
        if f:
            ci = char_ids(f) or [1]
            chars[i, :len(ci)] = ci
    drop = list(CTC_FEATURES) if args.no_ctc else []
    drop_idx = [FEATURE_INDEX[n] for n in drop]
    train.num[:, drop_idx] = 0.0
    dev.num[:, drop_idx] = 0.0
    print(f"[fusion] train {len(train.seqs)} sequences, {len(train.kinds)} units, {int(train.scored.sum())} candidates "
          f"({(train.gain[train.scored] > 0).mean():.1%} good); dev {len(dev.seqs)} sequences, "
          f"{int(dev.scored.sum())} candidates; vocab {len(vocab)}; {len(FEATURES)} signals, drop {drop}", flush=True)

    net_cfg = {"arch": args.arch, "dropout": args.dropout, "hidden": args.hidden}
    model = build_net(len(vocab), **net_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fusion] {args.arch}: {n_params:,} parameters on {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    L, B = args.seq_len, args.batch_size
    is_ted = np.array([s["dataset"] == "ted" for s in train.seqs])
    n_win = np.array([max(1, (s["end"] - s["start"]) // L) for s in train.seqs])
    ted_win, other_win = int(n_win[is_ted].sum()), int(n_win[~is_ted].sum())
    # TED talks are few but hold what read speech lacks (real repeats, false starts): oversample them
    repeat = max(1, round(args.ted_share * other_win / max((1 - args.ted_share) * ted_win, 1))) if ted_win else 1
    steps = (other_win + repeat * ted_win) // B
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps + 1, pct_start=0.1)
    print(f"[fusion] {other_win} + {repeat} x {ted_win} TED windows, {steps} steps per epoch", flush=True)

    last = out_dir / "last.pt"
    start_epoch, best, history = 0, None, []
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best, history = ck["epoch"], ck["best"], ck["history"]
        print(f"[fusion] resumed after epoch {start_epoch}", flush=True)

    def batch_arrays(corpus: Corpus, windows: list[tuple[int, int]], train_mode: bool):
        lens = np.array([e - s for s, e in windows])
        idx = np.zeros((len(windows), lens.max()), np.int64)
        mask = np.zeros(idx.shape, bool)
        for b, (s, e) in enumerate(windows):
            idx[b, :e - s] = np.arange(s, e)
            mask[b, :e - s] = True
        s_t, a_t = corpus.s[idx] * mask, corpus.a[idx] * mask
        s_w, a_w = type2word[s_t], type2word[a_t]
        if train_mode and args.word_dropout > 0:
            s_w = np.where((s_w > 1) & (rng.random(s_w.shape) < args.word_dropout), 1, s_w)
            a_w = np.where((a_w > 1) & (rng.random(a_w.shape) < args.word_dropout), 1, a_w)
        kinds = corpus.kinds[idx] * mask
        num = corpus.num[idx] * mask[..., None]
        to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
        x = [to(kinds), to(num.astype(np.float32))]
        if args.arch == "words":
            x += [to(s_w), to(chars[s_t]), to(a_w), to(chars[a_t])]
        y = (corpus.gain[idx] > 0) & mask
        w = corpus.scored[idx] & mask
        return x, to(lens), to(OUT_OF_KIND[kinds]), to(y.astype(np.float32)), to(w)

    def predict(corpus: Corpus) -> np.ndarray:
        """P(edit helps) of every unit (for the output of its own kind)."""
        model.eval()
        prob = np.zeros(len(corpus.kinds), np.float32)
        with torch.inference_mode():
            for s in corpus.seqs:
                x, lens, out, _, _ = batch_arrays(corpus, [(s["start"], s["end"])], False)
                p = torch.sigmoid(model(*x, lengths=lens)).gather(-1, out.unsqueeze(-1)).squeeze(-1)
                prob[s["start"]:s["end"]] = p[0].cpu().numpy()
        model.train()
        return prob

    sel = dev.scored
    kinds_out = OUT_OF_KIND[dev.kinds]
    rule_accept = dev.rule == 1
    base = rule_accept & ~dev.scored                 # split/merge stay with their rule
    err_all = float(sum(s["err"] for s in dev.seqs))
    nref_all = float(sum(s["n_ref"] for s in dev.seqs))
    all_cand = dev.rule >= 0

    def evaluate() -> dict:
        prob = predict(dev)
        tau = tune_thresholds(prob[all_cand], dev.gain[all_cand], np.where(sel, kinds_out, -1)[all_cand],
                              base[all_cand], err_all, nref_all)
        accept = base.copy()
        for k, name in enumerate(SCORED):
            m = sel & (kinds_out == k)
            accept[m] = prob[m] > tau[name]
        res: dict = {"thresholds": tau, "auc": {}}
        for k, name in enumerate(SCORED):
            m = sel & (kinds_out == k)
            res["auc"][name] = round(auc((dev.gain[m] > 0).astype(np.int64), prob[m]), 4)
        for tag, (m, err, nref) in group_masks(dev).items():
            res[tag] = summarize(accept[m & all_cand], dev.gain[m & all_cand], err, nref)
        for tag, (m, err, nref) in group_masks(dev, seed_zero=True).items():
            res[f"official_{tag}"] = summarize(accept[m & all_cand], dev.gain[m & all_cand], err, nref)
        res["score"] = round(res["all"]["f05"] - WER_WEIGHT * res["all"]["wer"], 5)
        return res

    rule = rule_numbers(dev)
    oracle = {f"official_{tag}": round(float(100 * (err - np.maximum(dev.gain[m & all_cand], 0).sum()) / nref), 3)
              for tag, (m, err, nref) in group_masks(dev, True).items()}
    print(f"[fusion] dev, tuned rule: {rule['all']}\n[fusion] dev official seed, tuned rule: {rule['official_all']}\n"
          f"[fusion] dev official seed, perfect decisions: WER {oracle}", flush=True)

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        windows = []
        for s, reps in zip(train.seqs, (repeat if t else 1 for t in is_ted)):
            n = s["end"] - s["start"]
            for _ in range(reps):
                off = int(rng.integers(0, L)) if n > L else 0
                ws = [(s["start"], s["start"] + off)] if off >= 8 else []
                ws += [(p, min(p + L, s["end"])) for p in range(s["start"] + off, s["end"], L)]
                windows += [(a, b) for a, b in ws if train.scored[a:b].any()]
        windows = [windows[k] for k in rng.permutation(len(windows))]
        tot = cnt = 0.0
        t1 = time.time()
        for k in range(min(steps, len(windows) // B)):
            x, lens, out, y, w = batch_arrays(train, windows[k * B:(k + 1) * B], True)
            logits = model(*x, lengths=lens).gather(-1, out.unsqueeze(-1)).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[w], y[w])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item()
            cnt += 1
        m = evaluate()
        m.update(epoch=epoch + 1, train_loss=round(tot / max(cnt, 1), 4), seconds=round(time.time() - t1, 1))
        history.append(m)
        print(f"[fusion] epoch {epoch + 1}: score {m['score']} loss {m['train_loss']} auc {m['auc']} tau {m['thresholds']}\n"
              f"[fusion]   dev all   {m['all']}\n[fusion]   dev libri {m['libri']}\n[fusion]   dev ted   {m['ted']}\n"
              f"[fusion]   official  {m['official_all']} ({m['seconds']:.0f}s)", flush=True)
        if best is None or m["score"] > best["score"]:
            best = m
            torch.save({"model": model.state_dict(), "vocab": vocab, "net": net_cfg, "drop": drop,
                        "thresholds": m["thresholds"], "features": list(FEATURES)}, out_dir / "model.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "best": best, "history": history}, last)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"best": best, "history": history, "rule": rule, "oracle_wer": oracle, "params": n_params,
         "train_candidates": int(train.scored.sum()), "train_units": int(len(train.kinds)),
         "train_sequences": len(train.seqs), "vocab": len(vocab), "minutes": round((time.time() - t_start) / 60, 1),
         "args": vars(args)}, indent=1), encoding="utf-8")
    print("[fusion] done, best:", best, flush=True)


if __name__ == "__main__":
    main()

"""Error analysis (imp.md 11, Week 9): worst files and most frequent residual errors of a system.

python experiments/error_analysis.py [--system ours] [--manifest test]  -> results/error_analysis.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from metrics.text_metrics import normalize_text  # noqa: E402
from subfix.subio import read_srt  # noqa: E402


def words(cues) -> list[str]:
    return normalize_text(" ".join(c.text.replace("\n", " ") for c in cues)).split()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ours")
    ap.add_argument("--manifest", default="test")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    from rapidfuzz.distance import Levenshtein
    items = [json.loads(x) for x in (ROOT / "data" / "benchmark" / f"{args.manifest}.jsonl")
             .read_text(encoding="utf-8").splitlines() if x.strip()]
    subs, dels, ins = Counter(), Counter(), Counter()
    kept_src_errors = Counter()
    for it in items:
        hyp_p = ROOT / "experiments" / "outputs" / args.system / it["variant"] / f"{it['id']}.srt"
        if not hyp_p.exists():
            continue
        ref, hyp = words(read_srt(ROOT / it["ref_srt"])), words(read_srt(hyp_p))
        src = words(read_srt(ROOT / it["corrupted_srt"]))
        src_set = Counter(src)
        for op in Levenshtein.opcodes(ref, hyp):
            tag, i1, i2, j1, j2 = op
            if tag == "replace":
                for r, h in zip(ref[i1:i2], hyp[j1:j2]):
                    subs[(r, h)] += 1
                    if src_set.get(h):
                        kept_src_errors["substitution also present in the corrupted subtitle"] += 1
                    else:
                        kept_src_errors["substitution introduced (ASR/correction)"] += 1
                for r in ref[i1 + (j2 - j1):i2]:
                    dels[r] += 1
                for h in hyp[j1 + (i2 - i1):j2]:
                    ins[h] += 1
            elif tag == "delete":
                for r in ref[i1:i2]:
                    dels[r] += 1
            elif tag == "insert":
                for h in hyp[j1:j2]:
                    ins[h] += 1
    per_item = ROOT / "results" / args.manifest / "per_item.csv"
    out = [f"# Error analysis: `{args.system}` on {args.manifest}", ""]
    if per_item.exists():
        df = pd.read_csv(per_item)
        df = df[df.system == args.system].sort_values("wer", ascending=False).head(10)
        out += ["## 10 worst files (WER)", "", "| File | Dataset | WER % | Edit F0.5 | Onset |Δt| ms |", "|---|---|---|---|---|"]
        out += [f"| {r.id} | {r.dataset} | {r.wer * 100:.2f} | {getattr(r, 'edit_f05', float('nan')) * 100:.1f} | "
                f"{getattr(r, 't_mean_ms', float('nan')):.0f} |" for r in df.itertuples()]
        out.append("")
    out += ["## Most frequent residual substitutions (reference → output)", "", "| Reference | Output | Count |",
            "|---|---|---|"] + [f"| {r} | {h} | {n} |" for (r, h), n in subs.most_common(args.top)] + [""]
    out += ["## Most frequent residual deletions / insertions", "",
            "Deleted (missing in output): " + ", ".join(f"{w} ({n})" for w, n in dels.most_common(args.top)), "",
            "Inserted (extra in output): " + ", ".join(f"{w} ({n})" for w, n in ins.most_common(args.top)), ""]
    out += ["## Origin of residual substitutions", ""] + [f"* {k}: {v}" for k, v in kept_src_errors.most_common()] + [""]
    dest = ROOT / "results" / "error_analysis.md"
    dest.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

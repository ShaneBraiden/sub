"""Coarse-sync recovery on the benchmark: estimated (a, b) vs the injected drift/offset.

The corruptor maps true time t -> subtitle time s = drift*t + offset (+ jump after the split cue),
so the ideal subtitle->audio mapping is a = 1/drift, b = -offset/drift (first piece).
python experiments/sync_eval.py [--manifest test]  -> results/sync_recovery.{md,json}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subfix.asr import transcribe  # noqa: E402
from subfix.config import Config  # noqa: E402
from subfix.subio import cues_to_words, read_srt  # noqa: E402
from subfix.sync import estimate  # noqa: E402
from subfix.utils.cache import save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="test")
    args = ap.parse_args()
    cfg = Config.load()
    rows = []
    for line in (ROOT / "data" / "benchmark" / f"{args.manifest}.jsonl").read_text(encoding="utf-8").splitlines():
        it = json.loads(line)
        t = json.loads((ROOT / it["corruption_log"]).read_text(encoding="utf-8"))["timing"]
        asr = transcribe(str(ROOT / it["audio"]), model="large-v3-turbo")
        res = estimate(cues_to_words(read_srt(ROOT / it["corrupted_srt"])), asr.words, cfg)
        a, b = res.params_at(0.0)
        a_true, b_true = 1.0 / t["drift"], -t["offset"] / t["drift"]
        rows.append({"id": it["id"], "dataset": it["dataset"], "method": res.method, "anchors": res.n_anchors,
                     "split": t["split_at_cue"] is not None, "drift": t["drift"], "offset": t["offset"],
                     "abs_da": abs(a - a_true), "abs_db": abs(b - b_true)})
        r = rows[-1]
        print(f"{r['id'][:36]:36s} {r['method']:9s} split={r['split']!s:5s} da={r['abs_da']:.5f} "
              f"db={r['abs_db']:.3f}s anchors={r['anchors']}", flush=True)
    da = np.array([r["abs_da"] for r in rows])
    db = np.array([r["abs_db"] for r in rows])
    summary = {
        "files": len(rows),
        "median_abs_da": float(np.median(da)), "max_abs_da": float(da.max()),
        "median_abs_db_s": float(np.median(db)), "share_db_below_250ms": float((db < 0.25).mean()),
        "share_db_below_500ms": float((db < 0.5).mean()),
        "splits_injected": int(sum(r["split"] for r in rows)),
        "splits_detected": int(sum(r["split"] and r["method"] == "piecewise" for r in rows)),
        "false_piecewise": int(sum((not r["split"]) and r["method"] == "piecewise" for r in rows)),
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    save_json(out / "sync_recovery.json", {"summary": summary, "files": rows})
    md = ["# Coarse-sync recovery (" + args.manifest + ")", "",
          "| File | Method | Split injected | Drift | Offset (s) | abs Δa | abs Δb (s) |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['id']} | {r['method']} | {'yes' if r['split'] else 'no'} | {r['drift']:.4f} | {r['offset']:+.2f} | "
           f"{r['abs_da']:.5f} | {r['abs_db']:.3f} |" for r in rows]
    md += ["", "Summary: " + json.dumps(summary)]
    (out / "sync_recovery.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()

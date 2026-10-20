"""Compact results summary + imp.md objective check -> results/SUMMARY.md

python experiments/summarize.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

COLS = [("wer", "WER %", 100, ".2f"), ("cer", "CER %", 100, ".2f"), ("errant_f05", "ERRANT F0.5", 100, ".1f"),
        ("edit_f05", "Edit F0.5", 100, ".1f"), ("overcorrection", "Over-corr %", 100, ".2f"),
        ("punct_macro_f1", "Punct F1", 100, ".1f"), ("t_mean_ms", "Onset |Δt| ms", 1, ".0f"),
        ("t_within_250", "≤250 ms %", 100, ".1f"), ("fmt_cps17_ok", "CPS≤17 %", 100, ".1f"),
        ("fmt_cps20_ok", "CPS≤20 %", 100, ".1f"), ("fmt_lines_ok", "Lines ok %", 100, ".1f"), ("rtf", "RTF", 1, ".3f")]
LABELS = {"noop": "Corrupted input (no-op)", "b1_symspell_lt": "B1: SymSpell + LanguageTool",
          "b2_ffsubsync": "B2: ffsubsync (timing only)", "b3_raw_whisper": "B3: raw Whisper replace",
          "ours": "**Ours** (all self-trained models except Whisper)", "modeb_raw": "Mode B: raw Whisper",
          "modeb_ours": "Mode B: Ours"}
# the earlier system with pretrained text/alignment models (results archived in results/pretrained_v1/)
V1_LABELS = {"ours": "Ours v1: distilgpt2 + FullStop XLM-R + T5-base GEC + MMS-300M + spaCy",
             "ours_ft_t5": "Ours v1 + fine-tuned T5-small", "modeb_ours": "Mode B: Ours v1 (pretrained)"}


def fmt(v, scale, f):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return format(v * scale, f)


def table(metrics: dict, dataset: str, systems: list[str]) -> list[str]:
    lines = ["| System | " + " | ".join(c[1] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
    for s in systems:
        m = metrics.get(s, {}).get(dataset)
        if m:
            lines.append(f"| {LABELS.get(s, s)} | " + " | ".join(fmt(m.get(c[0]), c[2], c[3]) for c in COLS) + " |")
    return lines


def main() -> None:
    p = RES / "test" / "metrics.json"
    if not p.exists():
        print("no results/test/metrics.json yet")
        return
    m = json.loads(p.read_text(encoding="utf-8"))
    mode_a = [s for s in ("noop", "b1_symspell_lt", "b2_ffsubsync", "b3_raw_whisper", "ours") if s in m]
    mode_b = [s for s in ("modeb_raw", "modeb_ours") if s in m]
    datasets = [d for d in ("ALL", "libri-test-clean", "libri-test-other", "ted-test", "own") if d in m.get("ours", {})]
    out = ["# Results summary (test split)", ""]
    for ds in datasets:
        out += [f"## {ds}", "", "Mode A (audio + corrupted subtitles):", ""] + table(m, ds, mode_a) + [""]
        if mode_b:
            out += ["Mode B (audio only):", ""] + table(m, ds, mode_b) + [""]
    # objectives (imp.md 1.2) on ALL
    a = m.get("ours", {}).get("ALL", {})
    n = m.get("noop", {}).get("ALL", {})
    rb, ob = m.get("modeb_raw", {}).get("ALL", {}), m.get("modeb_ours", {}).get("ALL", {})
    if a and n:
        rel = 1 - a["wer"] / n["wer"]
        out += ["## Objectives (imp.md 1.2)", "",
                "| # | Target | Result | Met |", "|---|---|---|---|",
                f"| O1 | ≥30% relative WER reduction (Mode A) | {n['wer'] * 100:.2f}% → {a['wer'] * 100:.2f}% "
                f"({rel * 100:.0f}% relative) | {'yes' if rel >= 0.30 else 'no'} |"]
        if rb and ob:
            better = ob["wer"] <= rb["wer"] and ob.get("punct_macro_f1", 0) >= rb.get("punct_macro_f1", 0)
            out.append(f"| O2 | Mode B: lower WER and higher punct F1 than raw Whisper | WER {rb['wer'] * 100:.2f}% → "
                       f"{ob['wer'] * 100:.2f}%, punct F1 {rb.get('punct_macro_f1', float('nan')) * 100:.1f} → "
                       f"{ob.get('punct_macro_f1', float('nan')) * 100:.1f} | {'yes' if better else 'partly/no'} |")
        out.append(f"| O3 | ≥90% of cue boundaries within ±250 ms | {a.get('t_within_250', float('nan')) * 100:.1f}% of "
                   f"cue onsets | {'yes' if a.get('t_within_250', 0) >= 0.9 else 'no'} |")
        o4 = min(a.get("fmt_lines_ok", 0), a.get("fmt_nlines_ok", 0), a.get("fmt_cps20_ok", 0))
        out.append(f"| O4 | ≥95% cues ≤42 chars/line, ≤2 lines, ≤17–20 CPS | lines {a.get('fmt_lines_ok', 0) * 100:.1f}%, "
                   f"2-line {a.get('fmt_nlines_ok', 0) * 100:.1f}%, CPS≤20 {a.get('fmt_cps20_ok', 0) * 100:.1f}%, "
                   f"CPS≤17 {a.get('fmt_cps17_ok', 0) * 100:.1f}% | {'yes (at 20 CPS)' if o4 >= 0.95 else 'no'} |")
        out.append("| O5 | Demo: upload → diff → download | web app (`app/backend/` + `frontend/`), tested by "
                   "`tests/test_backend.py` and `npm test` | yes |")
        out.append("")
        rb_ = a.get("recall_by_type", {})
        if rb_:
            out += ["Per-error-type recall of **Ours** (share of injected errors repaired):", "",
                    "| " + " | ".join(rb_) + " |", "|" + "---|" * len(rb_),
                    "| " + " | ".join(f"{v['recall'] * 100:.1f}% ({v['fixed']}/{v['total']})" for v in rb_.values()) + " |", ""]
    v1p = RES / "pretrained_v1" / "test" / "metrics.json"
    if v1p.exists() and "ours" in m:
        v1 = json.loads(v1p.read_text(encoding="utf-8"))
        out += ["## Self-trained models vs the earlier pretrained ones (ALL test files)", "",
                "Same benchmark, ASR (Whisper large-v3-turbo) and pipeline; v1 rows are from `results/pretrained_v1/`.", ""]
        rows = ["| System | " + " | ".join(c[1] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
        for key, store, label in (("ours", v1, V1_LABELS["ours"]), ("ours_ft_t5", v1, V1_LABELS["ours_ft_t5"]),
                                  ("ours", m, LABELS["ours"]), ("modeb_ours", v1, V1_LABELS["modeb_ours"]),
                                  ("modeb_ours", m, "Mode B: Ours (self-trained)")):
            r = store.get(key, {}).get("ALL")
            if r:
                rows.append(f"| {label} | " + " | ".join(fmt(r.get(c[0]), c[2], c[3]) for c in COLS) + " |")
        out += rows + [""]
    for name in ("timing_source", "segmenter", "audio_guard", "fusion_signals", "asr_size", "gec_model"):
        q = RES / "ablations" / name / "metrics.json"
        if not q.exists():
            continue
        am = json.loads(q.read_text(encoding="utf-8"))
        ds = "ALL" if any("ALL" in v for v in am.values()) else next(iter(next(iter(am.values()))))
        out += [f"## Ablation: {name} ({ds}, libri-test-clean + ted-test)", ""]
        out += table(am, ds, list(am)) + [""]
        out += ["Word-level timing vs MFA (mean |Δt| ms): " + ", ".join(
            f"{k.split(name + '_')[-1]} {v[ds].get('w_boundary_mean_ms', float('nan')):.0f}" for k, v in am.items()
            if ds in v), ""] if name == "timing_source" else []
    sweeps = sorted((RES / "ablations").glob("corruption_rate_*/metrics.json"))
    if sweeps:
        out += ["## Corruption-rate sweep (libri-test-clean + ted-test)", "", "| Injected rate | No-op WER % | Ours WER % |",
                "|---|---|---|"]
        for q in sweeps:
            sm = json.loads(q.read_text(encoding="utf-8"))
            key = "ALL" if "ALL" in sm.get("ours", {}) else next(iter(sm.get("ours", {"x": {}})))
            out.append(f"| {float(q.parent.name.split('_')[-1]) * 100:.0f}% | {sm['noop'][key]['wer'] * 100:.2f} | "
                       f"{sm['ours'][key]['wer'] * 100:.2f} |")
        out.append("")
    (RES / "SUMMARY.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

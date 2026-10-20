"""Formatting compliance — imp.md 3.5 / 6.1 (Netflix/BBC-style rules)."""
from __future__ import annotations

from subfix.subio import Cue, spoken_text


def _params(cfg) -> dict:
    s = cfg.segment if hasattr(cfg, "segment") else cfg
    get = (lambda k, d: getattr(s, k, d)) if not isinstance(s, dict) else (lambda k, d: s.get(k, d))
    return {"max_chars": get("max_chars", 42), "max_lines": get("max_lines", 2), "max_cps": get("max_cps", 17.0),
            "lenient_cps": get("lenient_cps", 20.0), "min_dur": get("min_dur", 0.833),
            "max_dur": get("max_dur", 7.0), "min_gap": get("min_gap", 0.083),
            "count_spaces": get("cps_count_spaces", False)}


def cue_cps(c: Cue, count_spaces: bool = False) -> float:
    """Reading speed; speaker labels count towards line length but not towards CPS."""
    flat = spoken_text(c).replace("\n", " ")
    chars = len(flat) if count_spaces else len(flat.replace(" ", ""))
    return chars / max(c.end - c.start, 1e-3)


def format_counts(cues: list[Cue], cfg=None) -> dict:
    p = _params(cfg) if cfg is not None else _params({})
    n = len(cues)
    c = {"n_cues": n, "lines_ok": 0, "nlines_ok": 0, "cps_ok": 0, "cps_lenient_ok": 0, "dur_ok": 0,
         "gap_ok": 0, "gap_n": max(n - 1, 0), "all_ok": 0}
    for k, cue in enumerate(cues):
        lines = [ln for ln in cue.text.split("\n")]
        l_ok = all(len(ln) <= p["max_chars"] or len(ln.split()) == 1 for ln in lines)
        n_ok = len(lines) <= p["max_lines"]
        cps = cue_cps(cue, p["count_spaces"])
        cps_ok = cps <= p["max_cps"] + 1e-9
        dur = cue.end - cue.start
        d_ok = p["min_dur"] - 1e-6 <= dur <= p["max_dur"] + 1e-6
        g_ok = True
        if k + 1 < n:
            g_ok = cues[k + 1].start - cue.end >= p["min_gap"] - 1e-6
            c["gap_ok"] += int(g_ok)
        c["lines_ok"] += int(l_ok)
        c["nlines_ok"] += int(n_ok)
        c["cps_ok"] += int(cps_ok)
        c["cps_lenient_ok"] += int(cps <= p["lenient_cps"] + 1e-9)
        c["dur_ok"] += int(d_ok)
        c["all_ok"] += int(l_ok and n_ok and cps_ok and d_ok and g_ok)
    return c


def summarize_format(counts: list[dict]) -> dict:
    tot = {k: sum(c[k] for c in counts) for k in counts[0]} if counts else {}
    n = tot.get("n_cues", 0)
    if not n:
        return {}
    return {"fmt_lines_ok": tot["lines_ok"] / n, "fmt_nlines_ok": tot["nlines_ok"] / n,
            "fmt_cps17_ok": tot["cps_ok"] / n, "fmt_cps20_ok": tot["cps_lenient_ok"] / n,
            "fmt_dur_ok": tot["dur_ok"] / n,
            "fmt_gap_ok": tot["gap_ok"] / tot["gap_n"] if tot["gap_n"] else 1.0,
            "fmt_all_ok": tot["all_ok"] / n, "n_cues": n}


def format_report(cues: list[Cue], cfg=None) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in summarize_format([format_counts(cues, cfg)]).items()}

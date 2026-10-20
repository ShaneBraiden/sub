"""Word-level diff (original vs corrected) rendered as HTML for the UI — imp.md 9.

Colours: green = inserted, red strikethrough = deleted, amber = substituted (tooltip: ASR word,
confidence, fusion score, rule, stages), dotted underline = punctuation/casing only,
blue = cue timing changed (old -> new). CPS is flagged red above the limit.
"""
from __future__ import annotations

import html

from . import text as T
from .align import align
from .subio import Word, spoken_text

CSS = """
<style>
.sfx-wrap{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;font-size:14px;color:var(--sfx-fg,#1d1d1f)}
.sfx-legend span{margin-right:14px}
.sfx-table{border-collapse:collapse;width:100%;margin-top:8px}
.sfx-table th,.sfx-table td{border-bottom:1px solid rgba(128,128,128,.25);padding:6px 8px;vertical-align:top;text-align:left}
.sfx-table th{font-weight:600;position:sticky;top:0;background:rgba(128,128,128,.12)}
.sfx-time{white-space:nowrap;font-variant-numeric:tabular-nums;font-size:12px}
.sfx-moved{color:#1565c0;font-weight:600}
.sfx-ins{background:rgba(46,160,67,.22);border-radius:3px;padding:0 2px}
.sfx-del{background:rgba(218,54,51,.18);text-decoration:line-through;border-radius:3px;padding:0 2px}
.sfx-sub{background:rgba(212,160,23,.30);border-radius:3px;padding:0 2px;cursor:help}
.sfx-fmt{text-decoration:underline dotted;text-underline-offset:3px}
.sfx-cps-bad{color:#d32f2f;font-weight:700}
.sfx-num{font-variant-numeric:tabular-nums;text-align:right}
.sfx-spk{font-size:12px;opacity:.75;margin-right:4px}
</style>
"""


def _fmt_t(t: float | None) -> str:
    if t is None:
        return "-"
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:06.3f}"


def _tooltip(w: Word) -> str:
    m = w.meta
    parts = []
    if m.get("orig_text"):
        parts.append(f"was: {m['orig_text']}")
    if m.get("asr_text"):
        p = m.get("asr_prob")
        parts.append(f"ASR: {m['asr_text']}" + (f" (p={p:.2f})" if isinstance(p, (int, float)) else ""))
    if m.get("fusion_score") is not None:
        parts.append(f"score={m['fusion_score']:.2f}")
    if m.get("rule"):
        parts.append(f"rule={m['rule']}")
    stages = [h["stage"] for h in m.get("history", [])]
    if m.get("stage") and m.get("stage") not in stages:
        stages.append(m["stage"])
    if stages:
        parts.append("stages: " + " > ".join(stages))
    return " | ".join(parts)


def diff_rows(result, max_cps: float | None = None) -> list[dict]:
    """One row per output cue with original span, corrected tokens and change markers."""
    cfg_seg = (result.config or {}).get("segment", {})
    max_cps = max_cps or cfg_seg.get("max_cps", 17.0)
    count_spaces = cfg_seg.get("cps_count_spaces", False)
    if result.mode == "A" and result.orig_words is not None:
        orig = result.orig_words
        orig_times = []
        for w in orig:
            c = result.orig_cues[w.meta["cue"]] if result.orig_cues else None
            orig_times.append((c.start, c.end) if c is not None else (w.start, w.end))
    else:
        orig = result.asr.words if result.asr else []
        orig_times = [(w.start, w.end) for w in orig]
    final = result.words
    ops = align([w.text for w in orig], [w.text for w in final])
    # final word -> output cue
    cue_of: dict[int, int] = {}
    pos = 0
    ids = {id(w): k for k, w in enumerate(final)}
    for ci, c in enumerate(result.cues):
        for w in c.words:
            k = ids.get(id(w))
            if k is not None:
                cue_of[k] = ci
            pos += 1
    rows = [{"orig": [], "new": [], "otimes": []} for _ in result.cues]
    pending_del: list[tuple[str, str]] = []
    last_cue = 0
    for op, i, j in ops:
        if j is not None:
            ci = cue_of.get(j, last_cue)
            last_cue = ci
            row = rows[ci] if rows else None
            if row is None:
                continue
            row["orig"].extend(pending_del)
            pending_del = []
            fw = final[j]
            if op == "ins":
                row["new"].append(("ins", fw.text, _tooltip(fw) or "inserted from audio"))
            elif op == "sub":
                row["orig"].append(("sub", orig[i].text, ""))
                row["new"].append(("sub", fw.text, _tooltip(fw)))
                row["otimes"].append(orig_times[i])
            else:  # match (maybe punctuation/casing change)
                kind = "same" if orig[i].text == fw.text else "fmt"
                row["orig"].append((kind, orig[i].text, ""))
                row["new"].append((kind, fw.text, _tooltip(fw) if kind == "fmt" else ""))
                row["otimes"].append(orig_times[i])
        else:  # deletion: attach to the next cue that receives a word
            pending_del.append(("del", orig[i].text))
    if pending_del and rows:
        rows[-1]["orig"].extend(pending_del)
    out = []
    for c, row in zip(result.cues, rows):
        ots = [t for t in row["otimes"] if t[0] is not None]
        o_start = min(t[0] for t in ots) if ots else None
        o_end = max(t[1] for t in ots) if ots else None
        moved = (o_start is None or abs(o_start - c.start) > 0.1 or abs((o_end or 0) - c.end) > 0.1)
        said = spoken_text(c)
        chars = len(said.replace("\n", " ")) if count_spaces else len(said.replace("\n", "").replace(" ", ""))
        cps = chars / max(c.end - c.start, 1e-3)
        out.append({"index": c.index, "start": c.start, "end": c.end, "old_start": o_start, "old_end": o_end,
                    "moved": moved, "orig": row["orig"], "new": row["new"], "cps": cps,
                    "cps_bad": cps > max_cps, "lines": c.text.split("\n"), "speakers": list(c.speakers),
                    "overlap": any(w.meta.get("overlap") for w in c.words)})
    return out


def _render_tokens(tokens, deleted_ok: bool = True) -> str:
    parts = []
    for tok in tokens:
        kind, txt = tok[0], html.escape(tok[1])
        tip = html.escape(tok[2]) if len(tok) > 2 and tok[2] else ""
        if kind == "same":
            parts.append(txt)
        elif kind == "fmt":
            parts.append(f'<span class="sfx-fmt" title="{tip}">{txt}</span>')
        elif kind == "ins":
            parts.append(f'<span class="sfx-ins" title="{tip}">{txt}</span>')
        elif kind == "del":
            parts.append(f'<span class="sfx-del">{txt}</span>')
        elif kind == "sub":
            parts.append(f'<span class="sfx-sub" title="{tip}">{txt}</span>')
    return " ".join(parts)


def html_side_by_side(result, max_cps: float | None = None, limit: int | None = None) -> str:
    rows = diff_rows(result, max_cps)
    if limit:
        rows = rows[:limit]
    left_title = "Original subtitle" if result.mode == "A" else "Raw ASR"
    head = (f'<div class="sfx-wrap">{CSS}<div class="sfx-legend">'
            '<span class="sfx-ins">inserted</span><span class="sfx-del">deleted</span>'
            '<span class="sfx-sub">substituted</span><span class="sfx-fmt">punct/case</span>'
            '<span class="sfx-moved">timing changed</span><span class="sfx-cps-bad">CPS over limit</span></div>'
            f'<table class="sfx-table"><thead><tr><th>#</th><th>Time (old &rarr; new)</th>'
            f'<th>{left_title}</th><th>Corrected</th><th class="sfx-num">CPS</th></tr></thead><tbody>')
    body = []
    for r in rows:
        old = f"{_fmt_t(r['old_start'])}&ndash;{_fmt_t(r['old_end'])}"
        new = f"{_fmt_t(r['start'])}&ndash;{_fmt_t(r['end'])}"
        tcls = "sfx-time sfx-moved" if r["moved"] else "sfx-time"
        new_html = _render_tokens(r["new"])
        names = list(dict.fromkeys(s for s in r.get("speakers", []) if s))
        if names:
            new_html = f'<b class="sfx-spk">{html.escape(" / ".join(names))}</b> ' + new_html
        cps_cls = "sfx-num sfx-cps-bad" if r["cps_bad"] else "sfx-num"
        body.append(f'<tr><td class="sfx-num">{r["index"]}</td><td class="{tcls}">{old}<br>&rarr; {new}</td>'
                    f'<td>{_render_tokens(r["orig"])}</td><td>{new_html}</td>'
                    f'<td class="{cps_cls}">{r["cps"]:.1f}</td></tr>')
    return head + "".join(body) + "</tbody></table></div>"


def summary_counts(rows: list[dict]) -> dict:
    counts = {"ins": 0, "del": 0, "sub": 0, "fmt": 0, "moved_cues": 0, "cps_over": 0}
    for r in rows:
        for tok in r["new"]:
            if tok[0] in counts:
                counts[tok[0]] += 1
        counts["del"] += sum(1 for tok in r["orig"] if tok[0] == "del")
        counts["moved_cues"] += int(r["moved"])
        counts["cps_over"] += int(r["cps_bad"])
    return counts


__all__ = ["html_side_by_side", "diff_rows", "summary_counts", "T"]

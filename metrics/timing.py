"""Timing metrics — imp.md 6.1.

Primary (segmentation-independent) — cue *onset* error: for every hypothesis cue, the cue start
minus the reference (MFA) start time of the cue's first word, found by aligning hypothesis and
reference words. A re-segmented output and a baseline that keeps the original cues are scored
on the same footing; reported as mean / median ms and % within ±100 / ±250 ms.

Secondary — cue-pair boundaries (imp.md wording): hypothesis cues matched one-to-one to
reference cues by word overlap (Jaccard >= 0.8, i.e. essentially the same words), then
|start_h - start_r| and |end_h - end_r|; the match rate shows how often segmentations agree.
The slope of the signed onset error over time is the residual drift (ms per minute).

Word level: |Δstart| / |Δend| between aligned hypothesis words and MFA reference words.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from subfix import text as T
from subfix.subio import Cue, tokenize


def _cue_tokens(cues: list[Cue]) -> tuple[list[str], list[int]]:
    toks, ids = [], []
    for k, c in enumerate(cues):
        for t in tokenize(c.text.replace("\n", " ")):
            n = T.norm(t)
            if n:
                toks.append(n)
                ids.append(k)
    return toks, ids


def _pairs(a: list[str], b: list[str], allow_replace: bool = False) -> list[tuple[int, int]]:
    from rapidfuzz.distance import Levenshtein
    out = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(a, b):
        if tag == "equal" or (allow_replace and tag == "replace" and i2 - i1 == j2 - j1):
            out.extend(zip(range(i1, i2), range(j1, j2)))
    return out


def match_cues(ref: list[Cue], hyp: list[Cue], min_jaccard: float = 0.8) -> list[tuple[int, int]]:
    rt, rc = _cue_tokens(ref)
    ht, hc = _cue_tokens(hyp)
    overlap: Counter = Counter()
    for i, j in _pairs(rt, ht):
        overlap[(rc[i], hc[j])] += 1
    rl, hl = Counter(rc), Counter(hc)
    cand = sorted(((cnt / (rl[r] + hl[h] - cnt), r, h) for (r, h), cnt in overlap.items()), reverse=True)
    used_r, used_h, out = set(), set(), []
    for jac, r, h in cand:
        if jac < min_jaccard:
            break
        if r in used_r or h in used_h:
            continue
        used_r.add(r)
        used_h.add(h)
        out.append((r, h))
    return sorted(out)


def onset_errors(hyp: list[Cue], ref_words: list[dict]) -> tuple[list[float], list[float]]:
    """(signed onset error, reference time) for every hypothesis cue whose first word aligns."""
    ht, hc = _cue_tokens(hyp)
    rn = [T.norm(w["text"]) for w in ref_words]
    keep = [k for k, n in enumerate(rn) if n]
    rmap = {i: keep[i] for i in range(len(keep))}
    h2r = dict(_pairs(ht, [rn[k] for k in keep], allow_replace=True))
    first_tok: dict[int, int] = {}
    for idx, cid in enumerate(hc):
        first_tok.setdefault(cid, idx)
    errs, times = [], []
    for cid, idx in sorted(first_tok.items()):
        if idx in h2r:
            rw = ref_words[rmap[h2r[idx]]]
            if rw.get("start") is not None:
                errs.append(hyp[cid].start - rw["start"])
                times.append(rw["start"])
    return errs, times


def cue_timing(ref: list[Cue], hyp: list[Cue], ref_words: list[dict] | None = None) -> dict:
    pairs = match_cues(ref, hyp)
    ds = [hyp[h].start - ref[r].start for r, h in pairs]
    de = [hyp[h].end - ref[r].end for r, h in pairs]
    out = {"n_ref_cues": len(ref), "n_hyp_cues": len(hyp), "matched": len(pairs),
           "_start_err": ds, "_end_err": de}
    if ref_words is not None:
        oe, ot = onset_errors(hyp, ref_words)
        out.update({"_onset_err": oe, "_onset_time": ot})
    return out


def _stats(errs: np.ndarray, prefix: str) -> dict:
    a = np.abs(errs)
    return {f"{prefix}_mean_ms": float(a.mean() * 1000), f"{prefix}_median_ms": float(np.median(a) * 1000),
            f"{prefix}_within_100": float((a <= 0.1).mean()), f"{prefix}_within_250": float((a <= 0.25).mean())}


def summarize_cue_timing(parts: list[dict]) -> dict:
    out: dict = {}
    n_ref = sum(p["n_ref_cues"] for p in parts)
    out["cue_match_rate"] = sum(p["matched"] for p in parts) / n_ref if n_ref else float("nan")
    ds = np.concatenate([np.array(p["_start_err"], float) for p in parts]) if parts else np.array([])
    de = np.concatenate([np.array(p["_end_err"], float) for p in parts]) if parts else np.array([])
    if len(ds):
        out.update(_stats(np.concatenate([ds, de]), "pair"))
    oe = [np.array(p["_onset_err"], float) for p in parts if "_onset_err" in p]
    if oe and sum(len(x) for x in oe):
        allo = np.concatenate(oe)
        out.update(_stats(allo, "t"))
        out["onset_bias_ms"] = float(np.median(allo) * 1000)
        slopes = []
        for p in parts:
            x, y = np.array(p.get("_onset_time", []), float), np.array(p.get("_onset_err", []), float)
            if len(x) >= 5 and np.ptp(x) > 30:
                slopes.append(np.polyfit(x, y, 1)[0])
        out["drift_ms_per_min"] = float(np.median(np.abs(slopes)) * 60_000) if slopes else 0.0
    else:
        out.update({"t_mean_ms": float("nan"), "t_median_ms": float("nan"), "t_within_100": float("nan"),
                    "t_within_250": float("nan")})
    return out


def word_timing(ref_words: list[dict], hyp_words: list[dict]) -> dict:
    """ref/hyp: [{"text", "start", "end"}]; aligned on normalised text."""
    rn = [T.norm(w["text"]) for w in ref_words]
    hn = [T.norm(w["text"]) for w in hyp_words]
    errs_s, errs_e = [], []
    for i, j in _pairs(rn, hn):
        r, h = ref_words[i], hyp_words[j]
        if r.get("start") is None or h.get("start") is None:
            continue
        errs_s.append(h["start"] - r["start"])
        errs_e.append(h["end"] - r["end"])
    return {"_w_start_err": errs_s, "_w_end_err": errs_e, "n_ref_words": len(ref_words)}


def summarize_word_timing(parts: list[dict]) -> dict:
    s = np.concatenate([np.array(p["_w_start_err"], float) for p in parts]) if parts else np.array([])
    e = np.concatenate([np.array(p["_w_end_err"], float) for p in parts]) if parts else np.array([])
    if not len(s):
        return {}
    a = np.abs(s)
    both = np.abs(np.concatenate([s, e]))
    return {"w_start_mean_ms": float(a.mean() * 1000), "w_start_median_ms": float(np.median(a) * 1000),
            "w_start_within_50": float((a <= 0.05).mean()), "w_start_within_100": float((a <= 0.1).mean()),
            "w_boundary_mean_ms": float(both.mean() * 1000), "w_aligned": int(len(s))}

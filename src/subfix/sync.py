"""Coarse subtitle sync — imp.md 8.6.

Anchors are unique exact n-gram matches between subtitle words (approximate times inside each
cue) and ASR words (DTW timestamps). A RANSAC line t' = a*t + b gives offset + linear drift
(|a-1| reveals an fps mismatch). If many anchors disagree with a single line, a change-point
search over the residual offsets finds offset jumps (ad cuts / different intro), giving a
piecewise mapping with a shared slope.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import text as T
from .align import find_anchors
from .subio import Cue, Word, cues_to_words

log = logging.getLogger(__name__)

# subtitle->audio slopes produced by common frame-rate mismatches (25 vs 23.976, 25 vs 24, 30 vs 29.97)
FPS_SLOPES = (1.0, 25 / 23.976, 23.976 / 25, 25 / 24, 24 / 25, 30 / 29.97, 29.97 / 30)


@dataclass
class SyncResult:
    a: float = 1.0
    b: float = 0.0
    pieces: list[dict] = field(default_factory=list)   # [{"until": t_sub, "a": .., "b": ..}]
    n_anchors: int = 0
    inlier_ratio: float = 0.0
    residual_mad: float = 0.0
    method: str = "none"

    def map(self, t: float) -> float:
        if self.pieces:
            for p in self.pieces:
                if t < p["until"]:
                    return p["a"] * t + p["b"]
            p = self.pieces[-1]
            return p["a"] * t + p["b"]
        return self.a * t + self.b

    def params_at(self, t: float) -> tuple[float, float]:
        if self.pieces:
            for p in self.pieces:
                if t < p["until"]:
                    return p["a"], p["b"]
            return self.pieces[-1]["a"], self.pieces[-1]["b"]
        return self.a, self.b

    def to_json(self) -> dict:
        d = asdict(self)
        for p in d["pieces"]:
            if p["until"] == float("inf"):
                p["until"] = None
        return d


def anchor_pairs(sub_words: list[Word], asr_words: list[Word], n: int = 3,
                 starts_only: bool = False) -> list[tuple[float, float]]:
    """(subtitle time, audio time) pairs from unique exact n-gram matches.

    Cue *starts* are the only subtitle times that are really known: word times inside a cue are
    interpolated by character count (badly off for long or merged cues) and cue ends are usually
    extended for reading time. With `starts_only` only the first word of each cue contributes
    (cue start, ASR word start); otherwise every matched word contributes its interpolated start.
    """
    sc = ["" if w.meta.get("event") else T.canon(w.text) for w in sub_words]
    ac = [T.canon(w.text) for w in asr_words]
    out = []
    for i, j in find_anchors(sc, ac, n):
        w, a = sub_words[i], asr_words[j]
        first = i == 0 or sub_words[i - 1].meta.get("cue") != w.meta.get("cue")
        if first or not starts_only:
            out.append((w.start, a.start))
    return out


def fit_offset_drift(anchor_pairs_: list[tuple[float, float]], residual_threshold: float = 1.0,
                     max_slope_dev: float = 0.1) -> tuple[float, float, np.ndarray]:
    """anchor_pairs: [(sub_time, asr_time)] -> (a, b, inlier_mask); t' = a*t + b."""
    X = np.array([[s] for s, _ in anchor_pairs_], dtype=np.float64)
    y = np.array([a for _, a in anchor_pairs_], dtype=np.float64)
    if len(y) < 2:
        return 1.0, float(np.median(y - X[:, 0])) if len(y) else 0.0, np.ones(len(y), bool)
    from sklearn.linear_model import LinearRegression, RANSACRegressor
    try:
        r = RANSACRegressor(LinearRegression(), residual_threshold=residual_threshold,
                            random_state=0, max_trials=500).fit(X, y)
        a, b = float(r.estimator_.coef_[0]), float(r.estimator_.intercept_)
        mask = np.asarray(r.inlier_mask_, dtype=bool)
    except ValueError:  # RANSAC could not find a consensus
        a, b, mask = 1.0, float(np.median(y - X[:, 0])), np.ones(len(y), bool)
    if abs(a - 1.0) > max_slope_dev or (X[:, 0].max() - X[:, 0].min()) < 30.0 or len(y) < 8:
        # implausible drift or too short a span to estimate a slope -> offset only
        off = y - X[:, 0]
        b = float(np.median(off))
        a = 1.0
        mask = np.abs(off - b) <= residual_threshold
    return a, b, mask


def _sse_prefix(x: np.ndarray):
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def sse(l: int, r: int) -> float:
        n = r - l
        if n <= 0:
            return 0.0
        s = c1[r] - c1[l]
        return float((c2[r] - c2[l]) - s * s / n)
    return sse


def _binary_segmentation(x: np.ndarray, min_size: int, penalty: float, depth: int = 3) -> list[tuple[int, int]]:
    sse = _sse_prefix(x)

    def rec(l: int, r: int, d: int) -> list[tuple[int, int]]:
        if d == 0 or r - l < 2 * min_size:
            return [(l, r)]
        base = sse(l, r)
        best_k, best_cost = -1, base
        for k in range(l + min_size, r - min_size + 1):
            c = sse(l, k) + sse(k, r)
            if c < best_cost:
                best_k, best_cost = k, c
        if best_k < 0 or (base - best_cost) / (r - l) < penalty:
            return [(l, r)]
        return rec(l, best_k, d - 1) + rec(best_k, r, d - 1)

    return rec(0, len(x), depth)


def fit_piecewise(pairs: list[tuple[float, float]], a: float, min_size: int = 6, penalty: float = 0.3,
                  residual_threshold: float = 1.0) -> list[dict] | None:
    s = np.array([p[0] for p in pairs])
    t = np.array([p[1] for p in pairs])
    order = np.argsort(s, kind="stable")
    s, t = s[order], t[order]
    off = t - a * s
    med = np.median(off)
    x = np.clip(off, med - 60.0, med + 60.0)
    segs = _binary_segmentation(x, min_size, penalty)
    if len(segs) < 2:
        return None
    for l, r in segs:        # a real offset jump leaves each piece flat; a wrong slope leaves a trend
        if r - l >= 5 and abs(np.polyfit(s[l:r], off[l:r], 1)[0]) * np.ptp(s[l:r]) > 1.0:
            return None      # leftover drift moves the offset by > 1 s across the piece
    pieces = []
    for k, (l, r) in enumerate(segs):
        b_k = float(np.median(off[l:r]))
        until = float("inf") if k == len(segs) - 1 else 0.5 * (s[r - 1] + s[r])
        pieces.append({"until": until, "a": float(a), "b": b_k, "n": int(r - l)})
    return pieces


def _inlier_ratio(pairs, mapper, thr) -> tuple[float, float]:
    res = np.array([abs(mapper(s) - t) for s, t in pairs])
    if len(res) == 0:
        return 0.0, 0.0
    return float(np.mean(res <= thr)), float(np.median(res))


def estimate(sub_words: list[Word], asr_words: list[Word], cfg) -> SyncResult:
    sc = cfg.sync
    pairs = []
    for n in sorted({sc.ngram, 2}, reverse=True):          # cue-start anchors first
        pairs = anchor_pairs(sub_words, asr_words, n, starts_only=True)
        if len(pairs) >= sc.min_anchors:
            break
    interior = False
    if len(pairs) < 3:                                      # very short files: interior words too
        pairs = anchor_pairs(sub_words, asr_words, 2, starts_only=False)
        interior = True
    if len(pairs) < 3:
        return SyncResult(n_anchors=len(pairs), method="none")
    a, b, mask = fit_offset_drift(pairs, sc.residual_threshold, sc.max_slope_dev)
    res = SyncResult(a=a, b=b, n_anchors=len(pairs), method="ransac")
    res.inlier_ratio, res.residual_mad = _inlier_ratio(pairs, res.map, sc.residual_threshold)
    if sc.piecewise and not interior and res.inlier_ratio < sc.piecewise_trigger             and len(pairs) >= 2 * sc.piecewise_min_anchors:
        # With an offset jump RANSAC may tilt its line across both clusters, so the piecewise
        # search also tries the slopes real drift comes from (frame-rate ratios).
        best = res
        for slope in dict.fromkeys([a, *FPS_SLOPES]):
            pieces = fit_piecewise(pairs, slope, sc.piecewise_min_anchors, sc.piecewise_penalty,
                                   sc.residual_threshold)
            if not pieces:
                continue
            cand = SyncResult(a=slope, b=pieces[0]["b"], pieces=pieces, n_anchors=len(pairs), method="piecewise")
            cand.inlier_ratio, cand.residual_mad = _inlier_ratio(pairs, cand.map, sc.residual_threshold)
            if cand.inlier_ratio >= max(best.inlier_ratio, res.inlier_ratio + 0.05) + 1e-9 and \
                    (best is res or cand.inlier_ratio > best.inlier_ratio):
                best = cand
        res = best
    return res


def apply(cues: list[Cue], res: SyncResult) -> list[Cue]:
    out = []
    for c in cues:
        a, b = res.params_at(c.start)
        s, e = a * c.start + b, a * c.end + b
        out.append(Cue(c.index, max(0.0, s), max(0.0, e, s + 0.01), c.text))
    return out


def apply_linear(cues: list[Cue], a: float, b: float) -> list[Cue]:
    return apply(cues, SyncResult(a=a, b=b, method="manual"))


def sync_cues(cues: list[Cue], asr_words: list[Word], cfg) -> tuple[list[Cue], SyncResult]:
    sub_words = cues_to_words(cues)
    res = estimate(sub_words, asr_words, cfg)
    if res.method == "none":
        return cues, res
    return apply(cues, res), res


# ------------------------------------------------------------------ ffsubsync baseline (B2)
def ffsubsync_available() -> list[str] | None:
    """Command prefix for ffsubsync. The package is run through the current interpreter: the
    venv's ffsubsync.exe launcher embeds the venv path and stops working when the folder moves."""
    import importlib.util
    if importlib.util.find_spec("ffsubsync") is not None:
        return [sys.executable, "-c", "import sys; from ffsubsync import main; sys.exit(main())"]
    exe = shutil.which("ffsubsync")
    return [exe] if exe else None


def run_ffsubsync(reference_media: str, srt_in: str, srt_out: str, timeout: int = 900) -> bool:
    prefix = ffsubsync_available()
    if not prefix:
        log.warning("ffsubsync not installed")
        return False
    from .preprocess import ffmpeg_dir
    env = dict(os.environ)
    fd = ffmpeg_dir()
    if fd:
        env["PATH"] = fd + os.pathsep + env.get("PATH", "")
    cmd = prefix + [str(reference_media), "-i", str(srt_in), "-o", str(srt_out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        log.warning("ffsubsync timed out")
        return False
    if r.returncode != 0 or not Path(srt_out).exists():
        log.warning("ffsubsync failed: %s", (r.stderr or r.stdout)[-500:])
        return False
    return True

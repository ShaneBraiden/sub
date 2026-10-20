"""Topic-segmentation metrics for the chapter markers (README "Chapter markers").

A segmentation of n units (words) is given by its boundaries: the indices of the units that start
a new segment (0 < b < n; the first segment's start is implicit).

* Pk (Beeferman, Berger & Lafferty 1999): the share of unit pairs k apart for which reference and
  hypothesis disagree on "same segment or not"; k = half the mean reference segment length.
* WindowDiff (Pevzner & Hearst 2002): the share of windows of k units in which the number of
  reference and hypothesis boundaries differ (also penalises near misses and extra boundaries).
* Boundary precision / recall / F1 in time: a hypothesis boundary counts if it lies within `tol`
  seconds of a reference boundary not matched yet (greedy one-to-one matching, closest first).

Pk and WindowDiff are errors (0 = perfect, lower is better).
"""
from __future__ import annotations

import numpy as np


def default_k(ref: list[int], n: int) -> int:
    return max(2, int(round(n / (len(ref) + 1) / 2)))


def _segment_ids(bounds: list[int], n: int) -> np.ndarray:
    ids = np.zeros(n, dtype=np.int64)
    for b in bounds:
        if 0 < b < n:
            ids[b:] += 1
    return ids


def pk(ref: list[int], hyp: list[int], n: int, k: int | None = None) -> float:
    k = k or default_k(ref, n)
    if n <= k:
        return 0.0
    r, h = _segment_ids(ref, n), _segment_ids(hyp, n)
    same_r = r[:-k] == r[k:]
    same_h = h[:-k] == h[k:]
    return float(np.mean(same_r != same_h))


def windowdiff(ref: list[int], hyp: list[int], n: int, k: int | None = None) -> float:
    k = k or default_k(ref, n)
    if n <= k:
        return 0.0
    r, h = _segment_ids(ref, n), _segment_ids(hyp, n)
    # boundaries inside the window (i, i+k] = segment-id difference across it
    return float(np.mean((r[k:] - r[:-k]) != (h[k:] - h[:-k])))


def boundary_prf(ref_times: list[float], hyp_times: list[float], tol: float) -> tuple[int, int, int]:
    """(matched, n_ref, n_hyp) with one-to-one matching within `tol` seconds, closest pairs first."""
    pairs = sorted((abs(r - h), i, j) for i, r in enumerate(ref_times) for j, h in enumerate(hyp_times)
                   if abs(r - h) <= tol)
    used_r, used_h, hit = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_r and j not in used_h:
            used_r.add(i)
            used_h.add(j)
            hit += 1
    return hit, len(ref_times), len(hyp_times)


def f1(hit: int, n_ref: int, n_hyp: int) -> tuple[float, float, float]:
    p = hit / n_hyp if n_hyp else (1.0 if not n_ref else 0.0)
    r = hit / n_ref if n_ref else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

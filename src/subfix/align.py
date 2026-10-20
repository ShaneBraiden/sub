"""Graded Needleman-Wunsch alignment of subtitle tokens vs ASR tokens — imp.md 3.2 / 8.3.

Substitution costs are graded (imp.md 3.2):
  0    identical (canonical form: lowercase, no punctuation, numbers spelled out)
  0.3  phonetically equal (metaphone) -> homophone-ish ("their"/"there")
  0.4 + 0.6*d  near miss, d = normalized char edit distance < 0.5 ("recieve"/"receive")
  1    otherwise;  gap cost 0.8.
Once times are known (after coarse sync) pairs further apart than `time_window` seconds get
an extra `time_penalty`, which keeps the alignment local.

Long files use *anchor-and-fill*: unique exact n-gram matches (made monotonic with an LIS)
pin the alignment, and plain NW fills the gaps between anchors. This is linear in file length
and, unlike overlapping per-cue windows, never matches or inserts one ASR word twice.
"""
from __future__ import annotations

import bisect
from functools import lru_cache
from typing import Sequence

import numpy as np

from . import text as T

Op = tuple[str, "int | None", "int | None"]   # ("match"|"sub"|"del"|"ins", i_sub, j_asr)

EVENT_COST = 10.0          # sound events never align to speech
MAX_BLOCK_CELLS = 16_000_000


def norm(t: str) -> str:
    return T.norm(t)


@lru_cache(maxsize=300_000)
def metaphone(token: str) -> str:
    c = T.canon(token)
    if not c:
        return ""
    try:
        import jellyfish
        return jellyfish.metaphone(c)
    except Exception:  # pragma: no cover
        return c


def sub_cost(a: str, b: str, homophone_cost: float = 0.3, near_miss_max: float = 0.5) -> float:
    """Scalar version of the graded substitution cost (used by tests / fusion features)."""
    ca, cb = T.canon(a), T.canon(b)
    if ca == cb:
        return 0.0
    ma, mb = metaphone(a), metaphone(b)
    if ma and ma == mb:
        return homophone_cost
    from rapidfuzz.distance import Levenshtein
    d = Levenshtein.normalized_distance(ca, cb)
    return 0.4 + 0.6 * d if d < near_miss_max else 1.0


def phon_sim(a: str, b: str) -> float:
    """1 - normalized edit distance between metaphone codes (1.0 = same sound)."""
    ma, mb = metaphone(a), metaphone(b)
    if not ma or not mb:
        return 0.0
    from rapidfuzz.distance import Levenshtein
    return 1.0 - Levenshtein.normalized_distance(ma, mb)


def cost_matrix(sub: Sequence[str], asr: Sequence[str], homophone_cost: float = 0.3,
                near_miss_max: float = 0.5, sub_times: Sequence[float] | None = None,
                asr_times: Sequence[float] | None = None, time_window: float = 3.0,
                time_penalty: float = 1.0, sub_fixed: Sequence[bool] | None = None) -> np.ndarray:
    """Vectorized graded substitution costs, shape (len(sub), len(asr))."""
    n, m = len(sub), len(asr)
    if n == 0 or m == 0:
        return np.zeros((n, m))
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist
    cs = [T.canon(t) for t in sub]
    ca = [T.canon(t) for t in asr]
    d = cdist(cs, ca, scorer=Levenshtein.normalized_distance, dtype=np.float32, workers=1)
    d = np.asarray(d, dtype=np.float64)
    cost = np.where(d < near_miss_max, 0.4 + 0.6 * d, 1.0)
    codes: dict[str, int] = {}
    ms = np.array([codes.setdefault(metaphone(t), len(codes)) if metaphone(t) else -1 for t in sub])
    ma = np.array([codes.setdefault(metaphone(t), len(codes)) if metaphone(t) else -2 for t in asr])
    cost = np.where(ms[:, None] == ma[None, :], np.minimum(cost, homophone_cost), cost)
    exact = np.array(cs, dtype=object)[:, None] == np.array(ca, dtype=object)[None, :]
    cost = np.where(exact, 0.0, cost)
    if sub_times is not None and asr_times is not None:
        ts = np.asarray(sub_times, dtype=np.float64)
        ta = np.asarray(asr_times, dtype=np.float64)
        far = np.abs(ts[:, None] - ta[None, :]) > time_window
        cost = cost + time_penalty * far
    if sub_fixed is not None:
        fixed = np.asarray(sub_fixed, dtype=bool)
        cost[fixed, :] = EVENT_COST
    return cost


def nw_align(cost: np.ndarray, gap: float = 0.8, equal: np.ndarray | None = None) -> list[Op]:
    """Needleman-Wunsch with row-vectorized DP. Tie order: diagonal > deletion > insertion."""
    n, m = cost.shape
    if n == 0:
        return [("ins", None, j) for j in range(m)]
    if m == 0:
        return [("del", i, None) for i in range(n)]
    tb = np.empty((n + 1, m + 1), dtype=np.int8)   # 0 diag, 1 up (del), 2 left (ins)
    tb[0, 0] = 0
    tb[0, 1:] = 2
    idx = np.arange(m + 1, dtype=np.float64) * gap
    prev = idx.copy()
    U = np.empty(m + 1)
    for i in range(1, n + 1):
        diag = prev[:-1] + cost[i - 1]
        up = prev[1:] + gap
        U[0] = prev[0] + gap
        U[1:] = np.minimum(diag, up)
        C = np.minimum.accumulate(U - idx) + idx
        row = np.empty(m + 1, dtype=np.int8)
        row[0] = 1
        row[1:] = np.where(diag <= up, 0, 1)
        row[C < U - 1e-9] = 2
        tb[i] = row
        prev = C
    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        c = tb[i, j]
        if i > 0 and j > 0 and c == 0:
            same = bool(equal[i - 1, j - 1]) if equal is not None else cost[i - 1, j - 1] == 0.0
            ops.append(("match" if same else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and (c == 1 or j == 0):
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def _lis_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Longest chain strictly increasing in both coordinates."""
    if not pairs:
        return []
    pairs = sorted(set(pairs), key=lambda p: (p[0], -p[1]))
    tails: list[int] = []
    tail_idx: list[int] = []
    prev = [-1] * len(pairs)
    for k, (_, j) in enumerate(pairs):
        pos = bisect.bisect_left(tails, j)
        if pos == len(tails):
            tails.append(j)
            tail_idx.append(k)
        else:
            tails[pos] = j
            tail_idx[pos] = k
        prev[k] = tail_idx[pos - 1] if pos > 0 else -1
    out = []
    k = tail_idx[-1]
    while k >= 0:
        out.append(pairs[k])
        k = prev[k]
    return out[::-1]


def find_anchors(sub_canon: Sequence[str], asr_canon: Sequence[str], n: int = 3,
                 sub_times: Sequence[float] | None = None, asr_times: Sequence[float] | None = None,
                 max_dt: float | None = None) -> list[tuple[int, int]]:
    """Token pairs covered by n-grams that occur exactly once in both sequences."""
    def unique_grams(seq: Sequence[str]) -> dict[tuple[str, ...], int]:
        seen: dict[tuple[str, ...], int] = {}
        dup: set[tuple[str, ...]] = set()
        for k in range(len(seq) - n + 1):
            g = tuple(seq[k:k + n])
            if not all(g):
                continue
            if g in seen:
                dup.add(g)
            else:
                seen[g] = k
        return {g: k for g, k in seen.items() if g not in dup}

    if len(sub_canon) < n or len(asr_canon) < n:
        return []
    gs, ga = unique_grams(sub_canon), unique_grams(asr_canon)
    pairs = []
    for g, i in gs.items():
        j = ga.get(g)
        if j is None:
            continue
        for k in range(n):
            if max_dt is not None and sub_times is not None and asr_times is not None:
                if abs(sub_times[i + k] - asr_times[j + k]) > max_dt:
                    continue
            pairs.append((i + k, j + k))
    return _lis_pairs(pairs)


def _align_block(sub: Sequence[str], asr: Sequence[str], params: dict,
                 sub_times=None, asr_times=None, sub_fixed=None) -> list[Op]:
    n, m = len(sub), len(asr)
    if n * m > MAX_BLOCK_CELLS and sub_times is not None and asr_times is not None and n > 1:
        mid = n // 2
        jm = bisect.bisect_left(list(asr_times), sub_times[mid])
        left = _align_block(sub[:mid], asr[:jm], params, sub_times[:mid], asr_times[:jm],
                            None if sub_fixed is None else sub_fixed[:mid])
        right = _align_block(sub[mid:], asr[jm:], params, sub_times[mid:], asr_times[jm:],
                             None if sub_fixed is None else sub_fixed[mid:])
        return left + [(op, None if i is None else i + mid, None if j is None else j + jm)
                       for op, i, j in right]
    cost = cost_matrix(sub, asr, params.get("homophone_cost", 0.3), params.get("near_miss_max", 0.5),
                       sub_times, asr_times, params.get("time_window", 3.0),
                       params.get("time_penalty", 1.0), sub_fixed)
    cs = np.array([T.canon(t) for t in sub], dtype=object)
    ca = np.array([T.canon(t) for t in asr], dtype=object)
    equal = (cs[:, None] == ca[None, :]) if n and m else None
    if equal is not None and sub_fixed is not None:
        equal[np.asarray(sub_fixed, dtype=bool), :] = False
    return nw_align(cost, params.get("gap", 0.8), equal)


def align(sub: Sequence[str], asr: Sequence[str], gap: float = 0.8, homophone_cost: float = 0.3,
          near_miss_max: float = 0.5, anchor_ngram: int = 3, sub_times: Sequence[float] | None = None,
          asr_times: Sequence[float] | None = None, time_window: float = 3.0,
          time_penalty: float = 1.0, sub_fixed: Sequence[bool] | None = None,
          use_anchors: bool = True) -> list[Op]:
    """Align two token lists -> ops [(op, i, j)], op in {match, sub, del, ins}."""
    params = dict(gap=gap, homophone_cost=homophone_cost, near_miss_max=near_miss_max,
                  time_window=time_window, time_penalty=time_penalty)
    n, m = len(sub), len(asr)
    if not use_anchors or n * m <= 40_000:
        return _align_block(sub, asr, params, sub_times, asr_times, sub_fixed)
    cs = [T.canon(t) for t in sub]
    ca = [T.canon(t) for t in asr]
    if sub_fixed is not None:
        cs = ["" if f else c for c, f in zip(cs, sub_fixed)]
    max_dt = 4.0 * time_window if sub_times is not None and asr_times is not None else None
    anchors = find_anchors(cs, ca, anchor_ngram, sub_times, asr_times, max_dt)
    ops: list[Op] = []
    pi, pj = -1, -1
    for ai, aj in anchors + [(n, m)]:
        i0, j0 = pi + 1, pj + 1
        if ai > i0 or aj > j0:
            block = _align_block(sub[i0:ai], asr[j0:aj], params,
                                 None if sub_times is None else sub_times[i0:ai],
                                 None if asr_times is None else asr_times[j0:aj],
                                 None if sub_fixed is None else sub_fixed[i0:ai])
            ops.extend((op, None if i is None else i + i0, None if j is None else j + j0)
                       for op, i, j in block)
        if ai < n and aj < m:
            ops.append(("match", ai, aj))
        pi, pj = ai, aj
    return ops


def align_words(sub_words, asr_words, cfg=None, use_time: bool = True) -> list[Op]:
    """Align subtitle Words to ASR Words (uses times when `use_time`)."""
    p = cfg.align if cfg is not None else None
    sub = [w.text for w in sub_words]
    asr = [w.text for w in asr_words]
    fixed = [bool(w.meta.get("event")) for w in sub_words]
    st = at = None
    if use_time and all(w.start is not None for w in sub_words) and all(w.start is not None for w in asr_words):
        st = [0.5 * (w.start + (w.end if w.end is not None else w.start)) for w in sub_words]
        at = [0.5 * (w.start + (w.end if w.end is not None else w.start)) for w in asr_words]
    kw = {}
    if p is not None:
        kw = dict(gap=p.gap, homophone_cost=p.homophone_cost, near_miss_max=p.near_miss_max,
                  anchor_ngram=p.anchor_ngram, time_window=p.time_window, time_penalty=p.time_penalty)
    return align(sub, asr, sub_times=st, asr_times=at, sub_fixed=fixed, **kw)


def ops_to_pairs(ops: list[Op]) -> dict[int, int]:
    """i -> j for diagonal (match/sub) ops."""
    return {i: j for op, i, j in ops if op in ("match", "sub")}


def opcodes_tokens(a: Sequence[str], b: Sequence[str]) -> list[Op]:
    """Plain (unit-cost) token alignment on canonical forms, for metrics and re-alignment."""
    return align(a, b, gap=1.0, homophone_cost=1.0, near_miss_max=0.0)

"""Diarization error rate (NIST md-eval style, frame-based).

DER = (missed speech + false alarm + speaker confusion) / reference speech, after mapping
hypothesis speakers to reference speakers one-to-one so that their overlap is largest
(Hungarian algorithm). Frames within `collar` seconds of a reference turn boundary are not scored.
Turns are (start, end, speaker) in seconds; turns of different speakers may overlap.
"""
from __future__ import annotations

import numpy as np


def _matrix(turns, names: list, n: int, step: float) -> np.ndarray:
    m = np.zeros((n, len(names)), dtype=bool)
    col = {s: k for k, s in enumerate(names)}
    for a, b, s in turns:
        m[int(round(a / step)):int(round(b / step)), col[s]] = True
    return m


def der(ref, hyp, collar: float = 0.25, step: float = 0.01) -> dict:
    """{der, miss, fa, conf, ref_speech} with the error parts in seconds."""
    from scipy.optimize import linear_sum_assignment
    end = max([b for _, b, _ in ref] + [b for _, b, _ in hyp] + [0.0])
    n = int(np.ceil(end / step)) + 1
    rn, hn = sorted({s for *_, s in ref}), sorted({s for *_, s in hyp})
    R, H = _matrix(ref, rn, n, step), _matrix(hyp, hn, n, step)
    keep = np.ones(n, dtype=bool)
    c = int(round(collar / step))
    if c > 0:
        for a, b, _ in ref:
            for t in (int(round(a / step)), int(round(b / step))):
                keep[max(0, t - c):t + c] = False
    R, H = R[keep], H[keep]
    correct = np.zeros(len(R), dtype=int)
    if rn and hn:
        overlap = R.T.astype(np.int64) @ H.astype(np.int64)
        ri, hi = linear_sum_assignment(-overlap)
        for i, j in zip(ri, hi):
            correct += R[:, i] & H[:, j]
    nr, nh = R.sum(1), H.sum(1)
    miss = np.maximum(nr - nh, 0).sum() * step
    fa = np.maximum(nh - nr, 0).sum() * step
    conf = (np.minimum(nr, nh) - correct).sum() * step
    total = nr.sum() * step
    return {"der": float((miss + fa + conf) / total) if total else 0.0, "miss": float(miss), "fa": float(fa),
            "conf": float(conf), "ref_speech": float(total)}

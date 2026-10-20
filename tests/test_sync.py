import random

import pytest

from subfix.config import Config
from subfix.subio import Cue, Word, cues_to_words
from subfix.sync import apply, estimate, fit_offset_drift, sync_cues


def _asr_words(n=700, seed=0):
    rng = random.Random(seed)
    vocab = [f"word{k}" for k in range(250)]
    words, t = [], 1.0
    for _ in range(n):
        dur = rng.uniform(0.2, 0.45)
        words.append(Word(rng.choice(vocab), t, t + dur, 0.95, "asr"))
        t += dur + rng.uniform(0.02, 0.25)
    return words


def _subtitle_from(words, to_sub, per_cue=8, jump_after=None, jump=0.0):
    cues = []
    for k in range(0, len(words), per_cue):
        chunk = words[k:k + per_cue]
        s, e = to_sub(chunk[0].start), to_sub(chunk[-1].end)
        if jump_after is not None and k >= jump_after:
            s, e = s + jump, e + jump
        cues.append(Cue(len(cues) + 1, s, e, " ".join(w.text for w in chunk)))
    return cues


def test_recover_offset_and_fps_drift():
    """imp.md 13: inject +2.0 s offset and 1.0427 drift; recover a,b within 0.001 / 50 ms."""
    a_true, b_true = 1.0427, 2.0          # audio = a * subtitle_time + b
    asr = _asr_words()
    cues = _subtitle_from(asr, lambda t: (t - b_true) / a_true)
    res = estimate(cues_to_words(cues), asr, Config())
    assert res.method == "ransac"
    assert abs(res.a - a_true) < 0.001
    assert abs(res.b - b_true) < 0.05


def test_fit_offset_drift_exact_pairs():
    pairs = [(t, 1.0427 * t + 2.0) for t in range(0, 300, 3)]
    a, b, mask = fit_offset_drift(pairs)
    assert a == pytest.approx(1.0427, abs=1e-6) and b == pytest.approx(2.0, abs=1e-4) and mask.all()


def test_piecewise_offset_jump_is_detected():
    asr = _asr_words(seed=1)
    cues = _subtitle_from(asr, lambda t: t - 3.0, jump_after=350, jump=6.0)
    synced, res = sync_cues(cues, asr, Config())
    assert res.method == "piecewise"
    # every cue start should land within 0.6 s of its first word in the audio
    for k, c in enumerate(synced):
        assert abs(c.start - asr[8 * k].start) < 0.6


def test_too_few_anchors_is_identity():
    cues = [Cue(1, 0, 1, "completely unrelated text")]
    asr = [Word("nothing", 0, 0.5), Word("matches", 0.5, 1.0), Word("here", 1.0, 1.5)]
    out, res = sync_cues(cues, asr, Config())
    assert res.method == "none" and out[0].start == 0


def test_apply_keeps_cue_order():
    from subfix.sync import SyncResult
    cues = [Cue(1, 1.0, 2.0, "a"), Cue(2, 3.0, 4.0, "b")]
    out = apply(cues, SyncResult(a=2.0, b=1.0, method="manual"))
    assert [(c.start, c.end) for c in out] == [(3.0, 5.0), (7.0, 9.0)]

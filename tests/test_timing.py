import numpy as np
import pytest

from subfix.subio import Cue, Word
from subfix.timing import ctc_viterbi, finalize_cue_times, plan_chunks, provisional_times, spans_from_path


def cue(i, s, e, words=None):
    c = Cue(i, s, e, "x")
    c.words = words or []
    return c


def check_invariants(cues, min_gap=0.083):
    for c in cues:
        assert c.end > c.start
    for a, b in zip(cues, cues[1:]):
        assert a.start < b.start
        assert a.end <= b.start - min_gap + 1e-9


def test_min_duration_extension_and_gap():
    cues = [cue(1, 0.0, 0.3, [Word("a", 0.0, 0.3)]), cue(2, 5.0, 6.0, [Word("b", 5.0, 6.0)])]
    out = finalize_cue_times(cues, min_dur=0.833, max_dur=7, min_gap=0.083)
    assert out[0].end == pytest.approx(0.833)
    check_invariants(out)


def test_max_duration_cap():
    out = finalize_cue_times([cue(1, 0.0, 9.0, [Word("a", 0.0, 9.0)])], max_dur=7.0)
    assert out[0].end == pytest.approx(7.0)


def test_overlapping_cues_are_separated():
    cues = [cue(1, 0, 0, [Word("a", 1.0, 2.5)]), cue(2, 0, 0, [Word("b", 2.0, 3.0)]),
            cue(3, 0, 0, [Word("c", 2.05, 2.1)])]
    out = finalize_cue_times(cues, min_dur=0.833, min_gap=0.083, linger=0.5)
    check_invariants(out)


def test_linger_respects_next_cue():
    cues = [cue(1, 0, 0, [Word("a", 0.0, 1.0)]), cue(2, 0, 0, [Word("b", 1.2, 2.0)])]
    out = finalize_cue_times(cues, linger=0.5, min_gap=0.083)
    assert out[0].end == pytest.approx(1.2 - 0.083)


def test_ctc_viterbi_recovers_token_spans():
    V, blank = 4, 0
    T_ = 15
    logp = np.full((T_, V), -10.0)
    logp[:, blank] = -0.1
    logp[2:5, 1] = -0.01        # token 1 at frames 2-4
    logp[8:12, 2] = -0.01       # token 2 at frames 8-11
    path = ctc_viterbi(logp, [1, 2], blank)
    spans = spans_from_path(path, [0, 1], 2)
    assert spans == [(2, 5), (8, 12)]


def test_ctc_viterbi_repeated_tokens_need_blank():
    logp = np.log(np.full((3, 3), 1 / 3))
    assert ctc_viterbi(logp, [1, 1], 0) is not None
    assert ctc_viterbi(logp[:2], [1, 1], 0) is None     # needs 3 frames (1, blank, 1)


def test_provisional_times_interpolates_and_is_monotonic():
    ws = [Word("one", 0.0, 0.5), Word("two"), Word("three"), Word("four", 2.0, 2.5), Word("five")]
    t = provisional_times(ws, 4.0)
    assert t[1, 0] == pytest.approx(0.5) and t[2, 1] == pytest.approx(2.0)
    assert np.all(np.diff(t[:, 0]) >= 0) and t[-1, 1] <= 4.0 + 1e-9


def test_plan_chunks_bounded():
    t = np.array([[k * 0.5, k * 0.5 + 0.3] for k in range(400)])
    chunks = plan_chunks(t, 200.0, max_s=60.0)
    assert chunks[0][0] == 0 and chunks[-1][1] == 400
    for a, b in chunks:
        assert t[b - 1, 1] - t[a, 0] <= 60.0
    assert all(c1[1] == c2[0] for c1, c2 in zip(chunks, chunks[1:]))

import random

from subfix import text as T
from subfix.subio import Cue, tokenize
from tools.corrupt import DEFAULT_RATES, corrupt_clip, typo

SENT = ("When Mister Bennet arrived in London the weather was cold, and their friends were waiting "
        "for him at the station. He said that it was too late to go to the theatre, so they went home.")


def ref_cues(n_rep=12):
    words = (SENT + " ") * n_rep
    toks = words.split()
    cues, t = [], 0.0
    for k in range(0, len(toks), 8):
        chunk = toks[k:k + 8]
        cues.append(Cue(len(cues) + 1, t, t + 2.4, " ".join(chunk)))
        t += 2.6
    return cues


def test_deterministic_with_seed():
    a = corrupt_clip(ref_cues(), "clip", True, None, seed=1)
    b = corrupt_clip(ref_cues(), "clip", True, None, seed=1)
    c = corrupt_clip(ref_cues(), "clip", True, None, seed=2)
    assert [(x.start, x.text) for x in a.cues] == [(x.start, x.text) for x in b.cues]
    assert a.log == b.log
    assert a.log != c.log


def test_log_matches_corrupted_tokens():
    res = corrupt_clip(ref_cues(), "clip", True, None, seed=4)
    src = [t for c in res.cues for t in tokenize(c.text.replace("\n", " "))]
    assert len(src) == res.log["n_src_tokens"]
    for e in res.log["text_errors"]:
        if e["type"] in ("spelling", "homophone", "entity"):
            assert T.canon(src[e["src_index"]]) == T.canon(e["new"])
            assert T.canon(e["new"]) != T.canon(e["orig"])
        if e["type"] == "dup":
            assert T.canon(src[e["src_index"]]) == T.canon(src[e["src_index"] + 1])


def test_rates_are_close_to_targets():
    res = corrupt_clip(ref_cues(30), "clip", True, None, seed=0)
    n = res.log["n_ref_tokens"]
    counts = res.log["counts"]
    for kind in ("homophone", "spelling", "drop", "dup"):
        assert abs(counts[kind] - DEFAULT_RATES[kind] * n) <= max(3, 0.25 * DEFAULT_RATES[kind] * n), kind


def test_timing_corruption_is_logged():
    res = corrupt_clip(ref_cues(), "clip", True, None, seed=7)
    tm = res.log["timing"]
    assert -5.0 <= tm["offset"] <= 5.0
    assert tm["drift"] in (1.0, 25 / 23.976, 23.976 / 25)
    assert all(c.end > c.start >= 0 for c in res.cues)


def test_typo_changes_word():
    rng = random.Random(0)
    for w in ["station", "weather", "theatre", "friends"]:
        assert typo(w, rng).lower() != w

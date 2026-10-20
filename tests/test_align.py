import random

import pytest

from subfix.align import _lis_pairs, align, find_anchors, sub_cost
from subfix import text as T


def ops(a: str, b: str, **kw):
    return align(a.split(), b.split(), **kw)


def kinds(o):
    return [k for k, _, _ in o]


def test_identical_all_match():
    o = ops("the cat sat on the mat", "the cat sat on the mat")
    assert kinds(o) == ["match"] * 6


def test_substitution():
    o = ops("i saw the cat", "i saw a cat")
    assert ("sub", 2, 2) in o and kinds(o).count("match") == 3


def test_deletion_duplicate_word():
    o = ops("the the cat", "the cat")
    assert kinds(o).count("del") == 1 and kinds(o).count("match") == 2


def test_insertion():
    o = ops("i want go home", "i want to go home")
    assert ("ins", None, 2) in o and kinds(o).count("match") == 4


def test_homophone_and_graded_costs():
    assert sub_cost("their", "there") == pytest.approx(0.3)
    assert sub_cost("Their,", "there") == pytest.approx(0.3)        # punctuation/case ignored
    assert sub_cost("kitten", "mitten") == pytest.approx(0.4 + 0.6 / 6)
    assert sub_cost("apple", "zebra") == 1.0
    assert sub_cost("Hello.", "hello") == 0.0


def test_homophone_preferred_over_gap():
    o = ops("over their now", "over there now")
    assert ("sub", 1, 1) in o


def test_numbers_match_spelled_out():
    assert T.canon("25") == T.canon("twenty-five")
    o = align(["it", "costs", "25", "dollars"], ["it", "costs", "twenty-five", "dollars"])
    assert kinds(o) == ["match"] * 4


def test_event_tokens_never_match():
    o = align(["[music]", "hello"], ["music", "hello"], sub_fixed=[True, False])
    assert ("del", 0, None) in o and ("ins", None, 0) in o


def test_time_penalty_prefers_nearby_word():
    o = align(["x"], ["x", "y", "x"], sub_times=[10.0], asr_times=[0.0, 5.0, 10.0], time_window=2.0)
    assert ("match", 0, 2) in o


def test_lis_pairs():
    assert _lis_pairs([(0, 0), (1, 5), (2, 1), (3, 2), (3, 3)]) == [(0, 0), (2, 1), (3, 2)]


def test_anchor_fill_equivalent_to_plain_nw():
    rng = random.Random(3)
    vocab = [f"w{k}" for k in range(400)]
    ref = [rng.choice(vocab) for _ in range(600)]
    hyp = []
    for w in ref:
        r = rng.random()
        if r < 0.05:
            hyp.append(rng.choice(vocab))      # substitution
        elif r < 0.08:
            continue                           # deletion
        elif r < 0.11:
            hyp.extend([w, rng.choice(vocab)])  # insertion
        else:
            hyp.append(w)
    fast = align(ref, hyp, use_anchors=True)
    slow = align(ref, hyp, use_anchors=False)
    assert kinds(fast).count("match") == kinds(slow).count("match")
    assert sorted(i for _, i, _ in fast if i is not None) == list(range(len(ref)))
    assert sorted(j for _, _, j in fast if j is not None) == list(range(len(hyp)))
    assert len(find_anchors([T.canon(x) for x in ref], [T.canon(x) for x in hyp])) > 100

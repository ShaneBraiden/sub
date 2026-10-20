"""Property tests for the segmenters (imp.md 13)."""
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from subfix.config import Config
from subfix.segment import best_two_line_split, segment
from subfix.subio import Word, tokenize

CFG = Config()
ALPHA = "abcdefghijklmnopqrstuvwxyz"

word_st = st.tuples(
    st.text(alphabet=ALPHA, min_size=1, max_size=14),
    st.sampled_from(["", "", "", ",", ".", "?"]),
    st.floats(min_value=0.08, max_value=1.2),      # word duration
    st.floats(min_value=0.0, max_value=1.5),       # pause after
)


def build(spec, long_word: bool = False):
    words, t = [], 0.3
    for k, (w, p, dur, gap) in enumerate(spec):
        if long_word and k == 0:
            w = "x" * 60
        words.append(Word(w + p, t, t + dur))
        t += dur + gap
    return words


def check(words, cues, cfg=CFG):
    seg = cfg.segment
    out_tokens = [t for c in cues for t in tokenize(c.text.replace("\n", " "))]
    assert out_tokens == [w.text for w in words]                  # all words, in order
    for c in cues:
        lines = c.text.split("\n")
        assert len(lines) <= seg.max_lines
        for ln in lines:
            assert len(ln) <= seg.max_chars or len(ln.split()) == 1
        assert c.end > c.start
    for a, b in zip(cues, cues[1:]):
        assert a.start < b.start
        assert a.end <= b.start - seg.min_gap + 1e-9               # monotonic, non-overlapping


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(word_st, min_size=1, max_size=70), st.booleans())
def test_dp_segmenter_properties(spec, long_word):
    words = build(spec, long_word)
    check(words, segment(words, CFG, None, "dp"))


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(word_st, min_size=1, max_size=70), st.booleans())
def test_greedy_segmenter_properties(spec, long_word):
    words = build(spec, long_word)
    check(words, segment(words, CFG, None, "greedy"))


def test_two_line_split_prefers_punctuation_and_bottom_heavy():
    toks = "I went to the market, and then I came straight back home".split()
    lines, _ = best_two_line_split(toks, 42)
    assert lines[0].endswith("market,")
    toks = "the quick brown fox jumps over the lazy dog near the river".split()
    lines, _ = best_two_line_split(toks, 42)
    assert not lines[0].endswith("the")                         # never split article-noun


def test_dp_breaks_at_sentence_end_and_respects_cps():
    text = ("This is the first sentence of the test. Here comes a second sentence that is longer. "
            "And finally a third one.").split()
    words, t = [], 0.0
    for w in text:
        words.append(Word(w, t, t + 0.3))
        t += 0.38
    cues = segment(words, CFG, None, "dp")
    assert any(c.text.replace("\n", " ").endswith("test.") for c in cues)

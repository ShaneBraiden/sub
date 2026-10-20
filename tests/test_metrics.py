import pytest

from metrics.format_metrics import format_counts, summarize_format
from metrics.text_metrics import edit_scores, normalize_text, punct_pairs, punct_scores, wer_cer
from metrics.timing import cue_timing, match_cues, summarize_cue_timing
from subfix.config import Config
from subfix.subio import Cue


def test_wer_cer_known_values():
    assert wer_cer(["the cat sat"], ["the cat sat"])["wer"] == 0.0
    r = wer_cer(["the cat sat on the mat"], ["the cat sit on mat"])
    assert r["wer"] == pytest.approx(2 / 6)
    assert normalize_text("Hello, World!") == "hello world"


def test_edit_scores_perfect_missing_and_overcorrection():
    src, ref = "i saw there car".split(), "i saw their car".split()
    perfect = edit_scores([src], [ref], [ref])
    assert perfect["edit_p"] == perfect["edit_r"] == perfect["edit_f05"] == 1.0
    noop = edit_scores([src], [ref], [src])
    assert noop["edit_r"] == 0.0 and noop["edit_f05"] == 0.0
    over = edit_scores([src], [ref], ["you saw their car".split()])
    assert over["edit_tp"] == 1 and over["edit_fp"] == 1
    assert over["overcorrection"] == pytest.approx(1 / 3)


def test_punct_scores_per_class():
    ref = ["Hello,", "world.", "How", "are", "you?"]
    hyp = ["hello", "world.", "how", "are", "you."]
    s = punct_scores(punct_pairs(ref, hyp))
    assert s["punct_._p"] == pytest.approx(0.5) and s["punct_._r"] == pytest.approx(1.0)
    assert s["punct_,_r"] == 0.0 and s["punct_?_r"] == 0.0
    assert s["casing_acc"] == pytest.approx(3 / 5)


REF_WORDS = [{"text": t, "start": s, "end": s + 0.5} for t, s in
             [("one", 0.0), ("two", 0.6), ("three", 1.2), ("four", 3.0), ("five", 3.6), ("six", 4.2)]]


def test_cue_timing_shift():
    ref = [Cue(1, 0.0, 2.0, "one two three"), Cue(2, 3.0, 5.0, "four five six")]
    hyp = [Cue(1, 0.2, 2.2, "one two three"), Cue(2, 3.2, 5.2, "four five six")]
    assert match_cues(ref, hyp) == [(0, 0), (1, 1)]
    s = summarize_cue_timing([cue_timing(ref, hyp, REF_WORDS)])
    assert s["t_mean_ms"] == pytest.approx(200, abs=1e-6)          # onset error
    assert s["pair_mean_ms"] == pytest.approx(200, abs=1e-6)
    assert s["t_within_250"] == 1.0 and s["t_within_100"] == 0.0 and s["cue_match_rate"] == 1.0


def test_onset_error_ignores_segmentation_differences():
    ref = [Cue(1, 0.0, 2.0, "one two three"), Cue(2, 3.0, 5.0, "four five six")]
    hyp = [Cue(1, 0.0, 1.0, "one two"), Cue(2, 1.2, 3.5, "three four"), Cue(3, 3.6, 5.0, "five six")]
    s = summarize_cue_timing([cue_timing(ref, hyp, REF_WORDS)])
    assert s["t_mean_ms"] == pytest.approx(0.0, abs=1e-6)


def test_cue_matching_is_one_to_one():
    ref = [Cue(1, 0.0, 4.0, "a b c d e f")]
    hyp = [Cue(1, 0.0, 2.0, "a b c"), Cue(2, 2.1, 4.0, "d e f")]
    assert len(match_cues(ref, hyp)) <= 1


def test_format_counts():
    cfg = Config()
    good = Cue(1, 0.0, 3.0, "A short line\nand another")
    long_line = Cue(2, 3.2, 4.0, "This line is definitely longer than forty-two characters")
    s = summarize_format([format_counts([good, long_line], cfg)])
    assert s["fmt_lines_ok"] == 0.5 and s["fmt_nlines_ok"] == 1.0
    assert s["fmt_gap_ok"] == 1.0
    assert s["fmt_cps17_ok"] == 0.5

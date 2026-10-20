import pytest

from subfix.subio import (Cue, compose_vtt, cue_words, cues_to_words, is_caseless, parse_srt_text,
                          parse_vtt_text, read_subs, tokenize, write_srt, write_vtt)

SRT = """1
00:00:01,000 --> 00:00:03,500
<i>Hello there,</i> how are you?

2
00:00:04,000 --> 00:00:06,000
- I'm fine.
- Good [laughs] thanks !
"""


def _tuples(cues):
    return [(round(c.start, 3), round(c.end, 3), c.text) for c in cues]


def test_srt_parse_strips_tags_and_dashes():
    cues = parse_srt_text(SRT)
    assert len(cues) == 2
    assert cues[0].start == pytest.approx(1.0) and cues[0].end == pytest.approx(3.5)
    assert cues[0].text == "Hello there, how are you?"
    assert cues[1].text == "I'm fine.\nGood [laughs] thanks !"


def test_srt_roundtrip(tmp_path):
    cues = parse_srt_text(SRT)
    p = tmp_path / "a.srt"
    write_srt(cues, p)
    assert _tuples(read_subs(p)) == _tuples(cues)
    write_srt(read_subs(p), tmp_path / "b.srt")
    assert (tmp_path / "b.srt").read_text(encoding="utf-8") == p.read_text(encoding="utf-8")


def test_vtt_roundtrip(tmp_path):
    cues = parse_srt_text(SRT)
    p = tmp_path / "a.vtt"
    write_vtt(cues, p)
    assert p.read_text(encoding="utf-8").startswith("WEBVTT")
    assert _tuples(read_subs(p)) == _tuples(cues)


def test_vtt_parse_header_note_and_short_timestamps():
    vtt = "WEBVTT\n\nNOTE a comment\n\n00:01.000 --> 00:02.500 align:start\nHi <b>there</b>\n\n" \
          "intro\n00:03.000 --> 00:04.000\nSecond\nline\n"
    cues = parse_vtt_text(vtt)
    assert _tuples(cues) == [(1.0, 2.5, "Hi there"), (3.0, 4.0, "Second\nline")]
    assert "00:00:01.000 --> 00:00:02.500" in compose_vtt(cues)


def test_tokenize_events_and_punctuation():
    assert tokenize("Good [door slams] thanks !") == ["Good", "[door slams]", "thanks!"]
    assert tokenize("well -- okay") == ["well--", "okay"]


def test_cue_words_times_lines_and_events():
    c = Cue(1, 10.0, 14.0, "Hello there,\n[music] friend")
    ws = cue_words(c, 0)
    assert [w.text for w in ws] == ["Hello", "there,", "[music]", "friend"]
    assert [w.meta["line"] for w in ws] == [0, 0, 1, 1]
    assert ws[2].meta.get("event") is True
    assert ws[0].start == pytest.approx(10.0) and ws[-1].end == pytest.approx(14.0)
    assert all(a.end <= b.start + 1e-9 for a, b in zip(ws, ws[1:]))


def test_cues_to_words_indices_and_caseless():
    cues = [Cue(1, 0, 2, "the quick brown fox jumps over"), Cue(2, 2, 4, "the lazy dog again and again")]
    ws = cues_to_words(cues)
    assert [w.meta["orig_index"] for w in ws] == list(range(len(ws)))
    assert {w.meta["cue"] for w in ws} == {0, 1}
    assert is_caseless(ws)
    cased = cues_to_words([Cue(1, 0, 2, "The quick brown Fox jumps over the lazy dog again and again")])
    assert not is_caseless(cased)


def test_asr_merges_split_numbers_and_punctuation():
    from subfix.asr import merge_pieces
    from subfix.subio import Word
    ws = [Word("about", 0, 0.3), Word("10", 0.3, 0.5), Word(",000", 0.5, 0.9, 0.8), Word("people", 1.0, 1.4),
          Word("6", 1.5, 1.6), Word(".8", 1.6, 1.8), Word("!", 1.8, 1.8)]
    out = merge_pieces(ws)
    assert [w.text for w in out] == ["about", "10,000", "people", "6.8!"]
    assert out[1].end == 0.9 and out[1].prob == 0.8 and [w.meta["asr_index"] for w in out] == [0, 1, 2, 3]

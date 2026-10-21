"""Chapter markers (src/subfix/chapters.py): stemming, both segmenters on a document with known
topic changes, timing, titles, output formats and the export."""
from __future__ import annotations

import json
import random

import pytest

from subfix.chapters import (Background, Chapter, Document, light_stem, make_chapters, sentence_spans,
                             to_ffmetadata, to_vtt, to_youtube)
from subfix.config import Config
from subfix.paths import ROOT
from subfix.subio import Cue, Word, parse_vtt_text

TOPICS = {
    "whales": "whale ocean harpoon ship sailor captain voyage sea deck mast".split(),
    "baking": "bread oven flour dough yeast baker loaf crust butter sugar".split(),
    "chess": "chess knight bishop pawn queen board opening endgame checkmate rook".split(),
}
GLUE = "the a of and with on in was very".split()


def document(order=("whales", "baking", "chess"), sentences=30, seed=0, punct=True):
    """Timed words: `sentences` sentences per topic, 0.3 s per word, 0.6 s pauses between sentences."""
    rng = random.Random(seed)
    words, t, starts = [], 0.0, []
    for topic in order:
        starts.append(len(words))
        for _ in range(sentences):
            toks = []
            for k in range(rng.randint(8, 12)):
                toks.append(rng.choice(TOPICS[topic]) if k % 2 == 0 else rng.choice(GLUE))
            toks[0] = toks[0].capitalize()
            if punct:
                toks[-1] += "."
            for tok in toks:
                words.append(Word(tok, round(t, 3), round(t + 0.25, 3)))
                t += 0.3
            t += 0.6
    return words, starts


def cfg(**chapters):
    return Config.load(overrides={f"chapters.{k}": v for k, v in chapters.items()})


def test_light_stem_merges_inflections_only():
    assert [light_stem(w) for w in ("whales", "voyages", "studies", "baking", "hopping", "agreed")] == \
        ["whale", "voyage", "studi", "bake", "hop", "agree"]
    assert light_stem("organization") == "organization"     # no derivational stripping
    assert light_stem("sea") == "sea" and light_stem("chess") == "chess"


@pytest.mark.parametrize("method", ["dp", "tiling"])
def test_finds_the_topic_changes(method):
    words, starts = document()
    chs = make_chapters(words, cfg(method=method, min_duration=30.0, cutoff=-1.0), duration=words[-1].end + 1)
    assert len(chs) == 3
    assert [ch.words[0] for ch in chs] == starts
    assert chs[0].start == 0.0 and chs[-1].end == pytest.approx(words[-1].end + 1)
    for a, b in zip(chs, chs[1:]):
        assert a.end == b.start
    assert [ch.score > 0 for ch in chs] == [False, True, True]
    for ch, topic in zip(chs, ("whales", "baking", "chess")):
        assert set(ch.title.lower().replace(",", " ").replace(" and ", " ").split()) <= set(TOPICS[topic])
        assert ch.summary and ch.summary.split()[0].lower() in TOPICS[topic]


def test_chapter_starts_in_the_pause_or_at_the_cue_it_opens():
    words, starts = document()
    chs = make_chapters(words, cfg(min_duration=30.0))
    w = words[starts[1]]
    assert chs[1].start == pytest.approx(w.start - 0.25)
    cue = Cue(1, w.start - 0.05, w.end, w.text, [w])
    chs = make_chapters(words, cfg(min_duration=30.0), cues=[cue])
    assert chs[1].start == pytest.approx(w.start - 0.05)


def test_exact_count_and_minimum_length():
    words, _ = document()
    assert len(make_chapters(words, cfg(num_chapters=2, min_duration=30.0))) == 2
    # 3 x 30 sentences of ~3.9 s: no room for 3 chapters of at least 2 minutes each
    chs = make_chapters(words, cfg(num_chapters=3, min_duration=120.0))
    assert len(chs) == 2 and all(ch.end - ch.start >= 120.0 for ch in chs)
    one = make_chapters(words, cfg(min_duration=1000.0))
    assert len(one) == 1 and one[0].score == 0.0
    assert make_chapters([], cfg()) == []


def test_unpunctuated_text_is_cut_at_long_pauses():
    words, _ = document(punct=False, sentences=3)
    st, en = [w.start for w in words], [w.end for w in words]
    spans = sentence_spans(words, st, en, max_words=12)
    assert all(b - a <= 12 for a, b in spans)
    assert [a for a, _ in spans][:3] == [0, *[a for a, _ in spans][1:3]]
    # every cut lands on one of the 0.6 s pauses between generated sentences
    assert all(st[a] - en[a - 1] > 0.5 for a, _ in spans[1:])


def test_generic_words_are_ignored():
    words = [Word(t, k * 0.3, k * 0.3 + 0.2) for k, t in enumerate("Things and stuff about the whale.".split())]
    doc = Document(words, None, cfg().chapters)
    assert [doc.words[k].text for k in range(len(words)) if doc.term(k, {"NOUN"}, True, 0.0)] == ["whale."]


def test_output_formats():
    chs = [Chapter(1, 0.0, 61.5, "Whales; ships = sea"), Chapter(2, 61.5, 3725.0, "Baking #2")]
    ff = to_ffmetadata(chs)
    assert ff.startswith(";FFMETADATA1\n")
    assert "START=61500\nEND=3725000\ntitle=Baking \\#2" in ff
    assert "title=Whales\\; ships \\= sea" in ff
    assert to_youtube(chs) == "0:00 Whales; ships = sea\n1:01 Baking #2\n"
    cues = parse_vtt_text(to_vtt(chs))
    assert [(c.start, c.end, c.text) for c in cues] == [(0.0, 61.5, "Whales; ships = sea"), (61.5, 3725.0, "Baking #2")]


def test_export_writes_chapter_files(tmp_path):
    from subfix.export import write_outputs
    from subfix.pipeline import RunResult
    words, _ = document()
    chs = make_chapters(words, cfg(min_duration=30.0))
    cues = [Cue(1, 0.0, 2.0, "Whale.", words[:1])]
    res = RunResult(mode="B", cues=cues, words=words, orig_cues=None, orig_words=None, asr=None, sync=None,
                    chapters=chs)
    paths = write_outputs(res, tmp_path, "clip", html=False)
    assert {"chapters_vtt", "chapters_txt", "ffmetadata", "chapters_json"} <= set(paths)
    rows = json.loads((tmp_path / "clip.chapters.json").read_text(encoding="utf-8"))
    assert [r["index"] for r in rows] == [1, 2, 3]
    assert json.loads((tmp_path / "clip.changes.json").read_text(encoding="utf-8"))["chapters"] == rows
    res.chapters = chs[:1]                       # a single chapter: no chapter files
    assert "chapters_vtt" not in write_outputs(res, tmp_path / "one", "clip", html=False)


@pytest.mark.skipif(not (ROOT / "models/lm/kn4.npz").exists(), reason="LM not trained")
def test_background_unigrams_from_our_lm():
    bg = Background(ROOT / "models/lm/kn4.npz")
    assert bg.logp("the") > bg.logp("whale") > bg.logp("uranium") >= bg.logp("qzxqzx")
    assert bg.logp("qzxqzx") == bg.unk


def test_dp_bonus_pulls_a_chapter_start_to_strong_evidence():
    import numpy as np
    words, _ = document()
    c = cfg(min_duration=30.0).chapters
    doc = Document(words, None, c)
    plain = doc.segment_dp(c.min_duration, c.penalty, 3)
    assert plain == doc.segment_dp(c.min_duration, c.penalty, 3, bonus=np.zeros(len(doc.sents)))
    bonus = np.zeros(len(doc.sents))
    k = plain[1] + 3                              # three sentences after the true change
    bonus[k] = 1e4
    moved = doc.segment_dp(c.min_duration, c.penalty, 3, bonus=bonus)
    assert moved == [0, k, plain[2], plain[3]]
    assert k in doc.segment_dp(c.min_duration, c.penalty, 0, bonus=bonus)


def test_texttiling_depth_matches_the_sentence_loop():
    """The prefix-sum similarity + one-pass peaks equal Hearst's block-by-block computation."""
    import math
    from collections import Counter

    import numpy as np
    words, _ = document(sentences=12, seed=3)
    doc = Document(words, None, cfg().chapters)
    S, nw = len(doc.sents), [b - a for a, b in doc.sents]
    for window, smoothing in ((15, 0), (40, 1), (100, 2)):
        sims = []
        for g in range(1, S):
            left, right, k, n = Counter(), Counter(), g - 1, 0
            while k >= 0 and n < window:
                left.update(doc.terms[k])
                n, k = n + nw[k], k - 1
            k, n = g, 0
            while k < S and n < window:
                right.update(doc.terms[k])
                n, k = n + nw[k], k + 1
            norm = math.sqrt(sum(v * v for v in left.values()) * sum(v * v for v in right.values()))
            sims.append(sum(v * right[t] for t, v in left.items()) / norm if norm else 0.0)
        s = np.asarray(sims)
        if smoothing:
            s = np.convolve(np.pad(s, smoothing, mode="edge"), np.ones(2 * smoothing + 1), "valid") / (2 * smoothing + 1)
        ref = []
        for g in range(len(s)):
            lo = hi = g
            while lo > 0 and s[lo - 1] >= s[lo]:
                lo -= 1
            while hi < len(s) - 1 and s[hi + 1] >= s[hi]:
                hi += 1
            ref.append((s[lo] - s[g]) + (s[hi] - s[g]))
        assert np.allclose(doc.depth(window, smoothing), ref)


def meeting(num: int = 0, **kw):
    """Chapters config in meeting mode with these chapters.meeting settings."""
    return Config.load(overrides={"chapters.mode": "meeting", "chapters.min_duration": 30.0, "chapters.num_chapters": num,
                                  **{f"chapters.meeting.{k}": v for k, v in kw.items()}})


def test_meeting_mode_finds_a_change_marked_by_silence_and_a_discourse_marker():
    """Both halves draw on the same words; only a long silence and "Okay," mark where the second starts."""
    rng = random.Random(5)
    vocab = TOPICS["whales"] + TOPICS["baking"]
    words, t, change = [], 0.0, 0
    for s in range(60):
        if s == 30:
            t += 4.0
            change = len(words)
        toks = [rng.choice(vocab) if k % 2 == 0 else rng.choice(GLUE) for k in range(rng.randint(8, 12))]
        toks[0] = "Okay," if s == 30 else toks[0].capitalize()
        toks[-1] += "."
        for tok in toks:
            words.append(Word(tok, round(t, 3), round(t + 0.25, 3)))
            t += 0.3
        t += 0.6
    two = make_chapters(words, meeting(2, pause_weight=4.0, cue_weight=16.0), duration=t)
    assert [ch.words[0] for ch in two] == [0, change]
    talk = make_chapters(words, cfg(min_duration=30.0, num_chapters=2), duration=t)
    assert [ch.words[0] for ch in talk] != [0, change]           # the words alone cannot see it
    doc = Document(words, None, meeting().chapters)
    k = next(i for i, (a, _) in enumerate(doc.sents) if a == change)
    bonus = doc.boundary_bonus(4.0, 16.0, 3.0, ["okay", "so"])
    assert bonus[k] == pytest.approx(4.0 * 3.0 + 16.0)          # silence capped at 3 s, plus the marker
    assert bonus[0] == 0.0 and bonus[1] == pytest.approx(4.0 * 0.65)   # ordinary 0.65 s pause, no marker


def test_overlapping_speech_counts_as_no_silence():
    words = [Word("Right.", 0.0, 5.0), Word("Yes.", 1.0, 1.5), Word("So", 5.5, 5.8), Word("then.", 5.8, 6.0)]
    doc = Document(words, None, cfg().chapters)
    assert doc.silence_before().tolist() == pytest.approx([0.0, 0.0, 0.5])
    assert doc.boundary_bonus(1.0, 2.0, 3.0, ["so"]).tolist() == pytest.approx([0.0, 0.0, 2.5])


def test_unknown_mode_is_an_error():
    words, _ = document(sentences=5)
    with pytest.raises(ValueError, match="chapters.mode"):
        make_chapters(words, Config.load(overrides={"chapters.mode": "lecture"}))

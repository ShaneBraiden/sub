"""Speaker diarization: word assignment, overlaps, speaker-aware segmentation, labels, statistics.

pyannote itself is never loaded here: a fake pipeline module stands in for it, so these tests run
without the model, a GPU or a Hugging Face token.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import soundfile as sf
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from metrics.format_metrics import cue_cps, format_report
from subfix import pipeline
from subfix.config import Config
from subfix.diarize import (Diarization, PyannoteDiarizer, Turn, assign_speakers, diarize, exclusive_from_turns,
                            line_prefixes, overlap_regions, parse_labels, speaker_report, speaker_turns, to_rttm,
                            union_length)
from subfix.export import write_outputs
from subfix.segment import keep_original, segment
from subfix.subio import Cue, Word, compose_srt, compose_vtt, parse_srt_text, parse_vtt_text, spoken_text, tokenize

CFG = Config().override({"diarize.enabled": True})


def timed(text: str, t0: float = 0.0, dur: float = 0.3, gap: float = 0.05) -> list[Word]:
    out, t = [], t0
    for tok in text.split():
        out.append(Word(tok, t, t + dur, 0.9, "asr"))
        t += dur + gap
    return out


def diar(*turns) -> Diarization:
    return Diarization.build([Turn(a, b, s) for a, b, s in turns])


# ------------------------------------------------------------------ diarization output
def test_build_renames_by_first_appearance_and_derives_exclusive():
    d = diar((5.0, 9.0, "B"), (0.0, 6.0, "A"), (8.0, 8.0, "C"))      # zero-length turn dropped
    assert d.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert [(t.start, t.end, t.speaker) for t in d.turns] == [(0.0, 6.0, "SPEAKER_00"), (5.0, 9.0, "SPEAKER_01")]
    # overlap 5-6 goes to the longer turn (A: 6 s vs B: 4 s); exclusive turns never overlap
    assert [(t.start, t.end, t.speaker) for t in d.exclusive] == [(0.0, 6.0, "SPEAKER_00"), (6.0, 9.0, "SPEAKER_01")]
    assert overlap_regions(d.turns) == [(5.0, 6.0, ("SPEAKER_00", "SPEAKER_01"))]
    assert union_length([(0, 6), (5, 9), (10, 11)]) == pytest.approx(10.0)
    assert Diarization.from_json(d.to_json()).to_json() == d.to_json()


def test_exclusive_merges_same_speaker_and_keeps_gaps():
    ex = exclusive_from_turns([Turn(0, 2, "A"), Turn(2, 3, "A"), Turn(4, 5, "B")])
    assert [(t.start, t.end, t.speaker) for t in ex] == [(0, 3, "A"), (4, 5, "B")]


# ------------------------------------------------------------------ words <- speakers
def test_assign_by_largest_overlap_nearest_turn_and_neighbours():
    d = diar((0.0, 1.0, "A"), (1.0, 2.0, "B"), (6.0, 7.0, "B"))
    words = [Word("one", 0.1, 0.5), Word("two", 0.8, 1.4),      # 0.2 s A vs 0.4 s B -> B
             Word("three", 2.3, 2.6),                           # 0.3 s after B's turn -> B (max_gap 1 s)
             Word("four", 4.0, 4.2),                            # 2 s from any turn -> previous word's speaker
             Word("five", 6.2, 6.5)]
    out = assign_speakers(words, d, max_gap=1.0, smooth_cover=0.0)
    assert [w.meta["speaker"] for w in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_01", "SPEAKER_01"]
    assert out[0].meta["speaker_cover"] == 1.0 and out[2].meta["speaker_cover"] == 0.0
    assert words[0].meta == {}                                  # inputs are not modified


def test_assign_without_turns_before_the_first_word_takes_the_next_speaker():
    d = diar((5.0, 6.0, "A"))
    out = assign_speakers([Word("x", 0.0, 0.2), Word("y", 5.1, 5.4)], d, max_gap=1.0)
    assert [w.meta["speaker"] for w in out] == ["SPEAKER_00", "SPEAKER_00"]


def test_weak_one_word_islands_are_smoothed_but_real_interjections_stay():
    d = diar((0.0, 1.62, "A"), (1.62, 2.0, "B"), (2.0, 4.0, "A"))
    words = [Word("so", 0.2, 0.5), Word("then", 1.3, 1.7),      # mostly A, edge of B's turn
             Word("we", 1.75, 1.95), Word("went", 2.1, 2.4)]    # fully inside B's short turn
    strong = assign_speakers(words, d, smooth_cover=0.5)
    assert [w.meta["speaker"] for w in strong] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    weak = diar((0.0, 1.75, "A"), (1.88, 2.0, "B"), (2.0, 4.0, "A"))  # "we" only 35 % inside B
    out = assign_speakers(words, weak, smooth_cover=0.5)
    assert [w.meta["speaker"] for w in out] == ["SPEAKER_00"] * 4


def test_overlapping_speech_is_flagged():
    d = diar((0.0, 3.0, "A"), (2.0, 4.0, "B"))
    words = [Word("a", 0.5, 1.0), Word("b", 1.9, 2.5), Word("c", 2.2, 2.4), Word("d", 3.2, 3.5), Word("e", 2.5, 2.5)]
    out = assign_speakers(words, d, overlap_min=0.5)
    flags = [w.meta.get("overlap", False) for w in out]
    assert flags == [False, True, True, False, True]            # b: 0.5 of 0.6 s; e: zero-length, inside
    # b and c sit between A ("a") and B ("d"): handed over at the longest pause, before b
    assert out[1].meta["speaker"] == "SPEAKER_01" and out[1].meta["overlap_with"] == ["SPEAKER_00"]


def test_handover_inside_overlapping_speech_splits_at_the_sentence_end():
    # A talks until 3.0, B starts at 2.0; the exclusive view gives the whole overlap to A (longer turn),
    # but Whisper's "Captain Fryer" in the overlap is already B speaking
    d = diar((0.0, 3.0, "A"), (2.0, 5.0, "B"))
    words = [Word("to", 1.2, 1.5), Word("their", 1.55, 1.85), Word("minds.", 2.05, 2.35), Word("Captain", 2.4, 2.7),
             Word("Fryer", 2.75, 2.95), Word("says", 3.1, 3.4), Word("that", 3.45, 3.75)]
    out = assign_speakers(words, d, overlap_min=0.5)
    assert [w.meta["speaker"] for w in out] == ["SPEAKER_00"] * 3 + ["SPEAKER_01"] * 4
    assert [w.text for w in out if w.meta.get("overlap")] == ["minds.", "Captain", "Fryer"]
    assert [w.meta["speaker"] for w in assign_speakers(words[:5], d)] == ["SPEAKER_00"] * 5   # no handover seen
    # an interjection (same speaker on both sides) keeps the plain assignment
    inter = diar((0.0, 5.0, "A"), (2.0, 2.6, "B"))
    words = timed("one two three", 1.0, 0.3, 0.05) + timed("four five", 2.6, 0.3, 0.05)
    assert {w.meta["speaker"] for w in assign_speakers(words, inter)} == {"SPEAKER_00"}


# ------------------------------------------------------------------ labels
def test_line_prefixes_modes():
    A, B = "SPEAKER_00", "SPEAKER_01"
    assert line_prefixes([A, A], None, "every") == ["SPEAKER_00: ", ""]
    assert line_prefixes([A], A, "every") == ["SPEAKER_00: "]
    assert line_prefixes([A], A, "change") == [""]
    assert line_prefixes([A, B], A, "change") == ["", "SPEAKER_01: "]
    assert line_prefixes([A, B], None, "none") == ["- ", "- "]
    assert line_prefixes([A, A], None, "none") == ["", ""]
    assert line_prefixes([A], None, "every", "[{speaker}] ") == ["[SPEAKER_00] "]
    with pytest.raises(ValueError):
        line_prefixes([A], None, "sometimes")


def test_parse_labels_of_edited_text():
    spk, labs = parse_labels(["SPEAKER_01: Hello there", "and more", "SPEAKER_00: hi"], ["SPEAKER_00", "SPEAKER_01"])
    assert spk == ["SPEAKER_01", "SPEAKER_01", "SPEAKER_00"] and labs == ["SPEAKER_01: ", "", "SPEAKER_00: "]
    assert parse_labels(["no label"], ["SPEAKER_00"], prev="SPEAKER_00") == (["SPEAKER_00"], [""])


# ------------------------------------------------------------------ speaker-aware segmentation
def two_speakers(n_a: int = 14, n_b: int = 3, n_c: int = 12):
    words = timed(" ".join(f"alpha{k}" for k in range(n_a)), 0.0)
    t = words[-1].end + 0.05
    words += timed(" ".join(f"beta{k}" for k in range(n_b)), t)
    t = words[-1].end + 0.05
    words += timed(" ".join(f"gamma{k}" for k in range(n_c)), t)
    for k, w in enumerate(words):
        w.meta["speaker"] = "SPEAKER_01" if n_a <= k < n_a + n_b else "SPEAKER_00"
    return words


def check_speaker_cues(words, cues, cfg, dialogue=True):
    seg = cfg.segment
    said = [t for c in cues for t in tokenize(spoken_text(c).replace("\n", " "))]
    assert said == [w.text for w in words]                      # all words, in order, labels removed
    for c in cues:
        lines = c.text.split("\n")
        assert len(c.speakers) == len(lines) == len(c.labels)
        assert len(lines) <= seg.max_lines
        for ln, lab, sp in zip(lines, c.labels, c.speakers):
            assert ln.startswith(lab)
            assert len(ln) <= seg.max_chars or len(ln[len(lab):].split()) == 1
        start = 0
        for ln, lab, sp in zip(lines, c.labels, c.speakers):     # each line is one speaker
            n = len(tokenize(ln[len(lab):]))
            assert {w.meta["speaker"] for w in c.words[start:start + n]} == {sp}
            start += n
        distinct = len(set(c.speakers))
        assert distinct <= (2 if dialogue else 1)
        if distinct == 2:
            assert len(lines) == 2 and c.speakers[0] != c.speakers[1]


def test_cues_never_mix_speakers_and_carry_labels():
    words = two_speakers()
    cues = segment(words, CFG)
    check_speaker_cues(words, cues, CFG)
    assert cues[0].text.startswith("SPEAKER_00: ")
    assert any(c.speakers[0] == "SPEAKER_01" or "SPEAKER_01" in c.speakers for c in cues)
    assert all(c.labels[0] for c in cues)                       # "every": each cue starts labelled


def test_quick_exchange_becomes_a_dialogue_cue():
    words = timed("Really?", 0.0) + timed("Yes.", 0.4) + timed("Wow.", 0.8)
    words[1].meta["speaker"] = "SPEAKER_01"
    words[0].meta["speaker"] = words[2].meta["speaker"] = "SPEAKER_00"
    cues = segment(words, CFG)
    check_speaker_cues(words, cues, CFG)
    assert cues[0].text == "SPEAKER_00: Really?\nSPEAKER_01: Yes."
    off = CFG.override({"diarize.dialogue": False})
    cues = segment(words, off)
    check_speaker_cues(words, cues, off, dialogue=False)
    assert len(cues) == 3


def test_label_mode_change_and_none():
    words = two_speakers(4, 2, 30)
    change = CFG.override({"diarize.label": "change"})
    cues = segment(words, change)
    check_speaker_cues(words, cues, change)
    labelled = [c for c in cues if any(c.labels)]
    assert 2 <= len(labelled) < len(cues)                       # later SPEAKER_00 cues are unlabelled
    none = CFG.override({"diarize.label": "none"})
    cues = segment(words, none)
    check_speaker_cues(words, cues, none)
    assert all(lab in ("", "- ") for c in cues for lab in c.labels)
    assert all(not c.text.startswith("SPEAKER") for c in cues)


def test_greedy_segmenter_never_spans_a_speaker_change():
    words = two_speakers()
    cues = segment(words, CFG, method="greedy")
    check_speaker_cues(words, cues, CFG, dialogue=False)


def test_keep_original_labels_each_line_by_majority():
    orig = [Cue(1, 0.0, 2.0, "hello there\nhi"), Cue(2, 2.5, 4.0, "how are you")]
    words = []
    for pos, c in enumerate(orig):
        for ln, line in enumerate(c.text.split("\n")):
            for tok in line.split():
                words.append(Word(tok, c.start, c.end, meta={"cue": pos, "line": ln}))
    for w, s in zip(words, ["A", "A", "B", "A", "A", "A"]):
        w.meta["speaker"] = s
    cues = keep_original(words, orig, CFG, keep_times=True)
    assert cues[0].text == "A: hello there\nB: hi" and cues[0].speakers == ["A", "B"]
    assert cues[1].text == "A: how are you"


word_st = st.tuples(st.text(alphabet="abcdefghij", min_size=1, max_size=12), st.floats(0.08, 1.0),
                    st.floats(0.0, 1.2), st.integers(0, 2))


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(word_st, min_size=1, max_size=60), st.sampled_from(["every", "change", "none"]),
       st.booleans(), st.sampled_from(["dp", "greedy"]))
def test_speaker_segmentation_properties(spec, mode, dialogue, method):
    words, t = [], 0.3
    for w, dur, gap, spk in spec:
        words.append(Word(w, t, t + dur, meta={"speaker": f"SPEAKER_0{spk}"}))
        t += dur + gap
    cfg = CFG.override({"diarize.label": mode, "diarize.dialogue": dialogue})
    cues = segment(words, cfg, method=method)
    check_speaker_cues(words, cues, cfg, dialogue=dialogue and method == "dp")
    for a, b in zip(cues, cues[1:]):
        assert a.end <= b.start - cfg.segment.min_gap + 1e-9


# ------------------------------------------------------------------ file formats + metrics
def test_labels_count_for_lines_but_not_for_reading_speed():
    c = Cue(1, 0.0, 1.0, "SPEAKER_00: abc", speakers=["SPEAKER_00"], labels=["SPEAKER_00: "])
    assert spoken_text(c) == "abc" and cue_cps(c) == pytest.approx(3.0)
    long = Cue(1, 0.0, 2.0, "SPEAKER_00: " + "x" * 35 + " y", speakers=["SPEAKER_00"], labels=["SPEAKER_00: "])
    assert format_report([long], CFG)["fmt_lines_ok"] == 0.0     # 49 visible characters > 42


def test_srt_and_vtt_with_speakers_round_trip_without_labels():
    cues = [Cue(1, 0.0, 1.5, "SPEAKER_00: Hello there,\nmy friend.", speakers=["SPEAKER_00"] * 2,
                labels=["SPEAKER_00: ", ""]),
            Cue(2, 2.0, 3.0, "SPEAKER_01: Hi.\nSPEAKER_00: Bye.", speakers=["SPEAKER_01", "SPEAKER_00"],
                labels=["SPEAKER_01: ", "SPEAKER_00: "])]
    vtt = compose_vtt(cues)
    assert "<v SPEAKER_00>SPEAKER_00: Hello there,\nmy friend.</v>" in vtt
    assert "<v SPEAKER_01>SPEAKER_01: Hi.</v>\n<v SPEAKER_00>SPEAKER_00: Bye.</v>" in vtt
    for back in (parse_vtt_text(vtt), parse_srt_text(compose_srt(cues))):
        assert [c.text for c in back] == ["Hello there,\nmy friend.", "Hi.\nBye."]
    assert compose_vtt([Cue(1, 0, 1, "plain")]).endswith("plain\n")


def test_speaker_turns_report_and_rttm():
    d = diar((0.0, 3.0, "A"), (2.5, 4.0, "B"), (4.5, 6.0, "A"))
    words = assign_speakers(timed("one two three four five six seven", 0.1, 0.5, 0.35), d)
    turns = speaker_turns(words)
    assert [t["speaker"] for t in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    cues = segment(words, CFG)
    rep = speaker_report(words, cues, d, 6.5)
    tot = rep["totals"]
    assert tot["speakers"] == 2 and tot["speaker_changes"] == 2
    assert tot["speech_time"] == pytest.approx(5.5) and tot["overlap_time"] == pytest.approx(0.5)
    a, b = rep["speakers"]
    assert a["speaker"] == "SPEAKER_00" and a["talk_time"] == pytest.approx(4.5) and a["turns"] == 2
    assert b["interruptions"] == 1 and a["interruptions"] == 0   # B starts while A is still talking
    assert a["words"] + b["words"] == 7 and a["wpm"] == pytest.approx(60 * a["words"] / 4.5, abs=0.1)
    assert rep["overlaps"] == [{"start": 2.5, "end": 3.0, "speakers": ["SPEAKER_00", "SPEAKER_01"]}]
    rttm = to_rttm(d, "my clip").splitlines()
    assert rttm[0] == "SPEAKER my_clip 1 0.000 3.000 <NA> <NA> SPEAKER_00 <NA> <NA>" and len(rttm) == 3


# ------------------------------------------------------------------ pyannote wrapper (faked) + cache + pipeline
class _Seg:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _Ann:
    def __init__(self, tracks):
        self.tracks = tracks

    def itertracks(self, yield_label=False):
        for a, b, s in self.tracks:
            yield _Seg(a, b), None, s


class _Out:
    def __init__(self, regular, exclusive):
        self.speaker_diarization, self.exclusive_speaker_diarization = regular, exclusive


@pytest.fixture
def fake_pyannote(monkeypatch):
    calls: dict = {}

    class Pipeline:
        @staticmethod
        def from_pretrained(source, token=None):
            calls["source"], calls["token"] = source, token
            if source == "gated/model":
                raise PermissionError("403 gated")
            return Pipeline()

        def to(self, device):
            calls["device"] = str(device)
            return self

        def __call__(self, file, **kw):
            calls["kw"], calls["shape"] = kw, tuple(file["waveform"].shape)
            return _Out(_Ann([(0.0, 2.0, "spk_b"), (1.5, 3.0, "spk_a")]),
                        _Ann([(0.0, 1.7, "spk_b"), (1.7, 3.0, "spk_a")]))

    audio = types.ModuleType("pyannote.audio")
    audio.Pipeline = Pipeline
    root = types.ModuleType("pyannote")
    root.audio = audio
    monkeypatch.setitem(sys.modules, "pyannote", root)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)
    return calls


def test_pyannote_wrapper(fake_pyannote, monkeypatch):
    import os
    monkeypatch.setenv("MY_TOKEN", "hf_test")
    dz = PyannoteDiarizer("some/model", "cpu", "MY_TOKEN")
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "false"     # pyannote telemetry stays off
    assert fake_pyannote["token"] == "hf_test" and fake_pyannote["device"] == "cpu"
    res = dz(np.zeros(16000 * 3, np.float32), 16000, num_speakers=2, max_speakers=5)
    assert fake_pyannote["kw"] == {"num_speakers": 2} and fake_pyannote["shape"] == (1, 48000)
    assert res.speakers == ["SPEAKER_00", "SPEAKER_01"]           # spk_b spoke first
    assert [(t.start, t.end) for t in res.exclusive] == [(0.0, 1.7), (1.7, 3.0)]
    assert res.info["exclusive"] == "pyannote"
    with pytest.raises(RuntimeError, match="huggingface.co/gated/model.*MY_TOKEN"):
        PyannoteDiarizer("gated/model", "cpu", "MY_TOKEN")


def write_wav(path, seconds=8.0):
    sf.write(str(path), np.zeros(int(16000 * seconds), np.float32), 16000, subtype="PCM_16")
    return str(path)


class FakeDiarizer:
    def __init__(self, turns):
        self.turns, self.calls = turns, 0

    def __call__(self, audio, sr, num_speakers=0, min_speakers=0, max_speakers=0):
        self.calls += 1
        return Diarization.build([Turn(*t) for t in self.turns], info={"model": "fake"})


class FakeModels:
    def __init__(self, diarizer):
        self.d = diarizer
        self.released: list[str] = []

    def diarizer(self, cfg=None):
        return self.d

    def pos_tagger(self, cfg=None):
        return None

    def background(self, cfg=None):
        return None

    def release(self, *kinds):
        self.released.extend(kinds)


def test_diarize_is_cached_by_audio_and_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBFIX_CACHE", str(tmp_path / "cache"))
    wav = write_wav(tmp_path / "a.wav", 3.0)
    fake = FakeDiarizer([(0.0, 1.0, "x"), (1.0, 3.0, "y")])
    first = diarize(wav, CFG, FakeModels(fake))
    again = diarize(wav, CFG, FakeModels(fake))
    assert fake.calls == 1 and again.to_json()["turns"] == first.to_json()["turns"]
    diarize(wav, CFG.override({"diarize.num_speakers": 2}), FakeModels(fake))
    assert fake.calls == 2


def test_pipeline_finish_labels_cues_and_writes_speaker_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBFIX_CACHE", str(tmp_path / "cache"))
    wav = write_wav(tmp_path / "talk.wav")
    words = timed("So what did you think of the film last night? Honestly it was far too long.", 0.2, 0.25, 0.1)
    words += timed("Fair enough.", words[-1].end + 0.5)
    cut = next(k for k, w in enumerate(words) if w.text == "Honestly")
    t_cut = words[cut].start - 0.05
    fake = FakeDiarizer([(0.0, t_cut, "host"), (t_cut, words[-3].end + 0.1, "guest"), (words[-2].start - 0.1, 8.0, "host")])
    cfg = CFG.override({"timing.backend": "keep", "keep_models_loaded": False})
    models = FakeModels(fake)
    times: dict = {}
    res = pipeline._finish("B", words, [], None, None, None, None, None, times, pipeline._Timer(times, None),
                           8.0, wav, cfg, models)
    assert "diarize" in times and models.released == ["diarizer"]
    assert {w.meta["speaker"] for w in res.words[:cut]} == {"SPEAKER_00"}
    assert {w.meta["speaker"] for w in res.words[cut:-2]} == {"SPEAKER_01"}
    assert res.cues[0].text.startswith("SPEAKER_00: So what")
    assert res.speakers["totals"]["speakers"] == 2
    log = res.to_log()
    assert log["words"][0]["speaker"] == "SPEAKER_00" and log["cues"][0]["speakers"][0] == "SPEAKER_00"
    paths = write_outputs(res, tmp_path / "out", "talk")
    assert set(paths) >= {"srt", "vtt", "json", "html", "rttm", "speakers"}
    srt = (tmp_path / "out" / "talk.srt").read_text(encoding="utf-8")
    assert "SPEAKER_01: Honestly" in srt
    assert "<v SPEAKER_01>" in (tmp_path / "out" / "talk.vtt").read_text(encoding="utf-8")
    assert (tmp_path / "out" / "talk.rttm").read_text(encoding="utf-8").count("SPEAKER talk 1") == 3


def test_pipeline_without_diarization_has_no_speaker_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBFIX_CACHE", str(tmp_path / "cache"))
    wav = write_wav(tmp_path / "talk.wav", 3.0)
    cfg = Config().override({"timing.backend": "keep"})
    times: dict = {}
    res = pipeline._finish("B", timed("hello there my friend"), [], None, None, None, None, None, times,
                           pipeline._Timer(times, None), 3.0, wav, cfg, FakeModels(None))
    assert res.speakers is None and res.diarization is None and "diarize" not in times
    assert all(not c.speakers and not c.labels for c in res.cues)
    assert "speakers" not in res.to_log() and "speaker" not in res.to_log()["words"][0]
    assert set(write_outputs(res, tmp_path / "out", "x")) == {"srt", "vtt", "json", "html"}

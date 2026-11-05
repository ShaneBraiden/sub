"""asr.degrade: the opt-in "bad Whisper" used for demos and stress tests."""
from subfix.asr import AsrResult, degrade
from subfix.subio import Word


def asr(n: int = 400) -> AsrResult:
    words = [Word("Their," if i % 10 == 0 else "Nation.", i * 0.5, i * 0.5 + 0.4, 0.99, "asr", {"asr_index": i})
             for i in range(n)]
    segs = [{"start": 0.0, "end": n * 0.25, "text": "x", "words": [0, n // 2]},
            {"start": n * 0.25, "end": n * 0.5, "text": "y", "words": [n // 2, n]}]
    return AsrResult(words, segs, {"model": "fake"})


def test_strips_punctuation_and_case_even_without_errors():
    out = degrade(asr(), 0.0)
    assert {w.text for w in out.words} == {"their", "nation"}
    assert out.info["degraded"]["rate"] == 0.0
    assert all(w.prob == 0.99 for w in out.words)


def test_is_seeded_and_leaves_the_input_alone():
    src = asr()
    a, b = degrade(src, 0.3), degrade(src, 0.3)
    assert a.text == b.text
    assert degrade(src, 0.3, seed=1).text != a.text
    assert src.words[0].text == "Their,"
    assert "degraded" not in src.info


def test_rate_counts_and_lowered_probability():
    out = degrade(asr(), 0.3)
    d = out.info["degraded"]
    done = d["homophone"] + d["typo"] + d["drop"] + d["dup"]
    assert 60 < done < 180
    assert sum(w.prob <= 0.3 for w in out.words) == d["homophone"] + d["typo"] + 2 * d["dup"]


def test_segments_stay_consistent():
    out = degrade(asr(), 0.3)
    assert out.segments[0]["words"][1] == out.segments[1]["words"][0]
    assert out.segments[1]["words"][1] == len(out.words)
    assert " ".join(s["text"] for s in out.segments) == out.text
    assert [w.meta["asr_index"] for w in out.words] == list(range(len(out.words)))


def test_server_flag_sets_the_rate_and_is_off_by_default(monkeypatch):
    from app.backend.runner import PipelineRunner
    from app.backend.schemas import option_defaults

    monkeypatch.delenv("SUBFIX_SERVER_DEGRADE_WHISPER", raising=False)
    assert PipelineRunner().base_config().asr.degrade == 0.0
    monkeypatch.setenv("SUBFIX_SERVER_DEGRADE_WHISPER", "0.25")
    cfg = PipelineRunner().base_config()
    assert cfg.asr.degrade == 0.25
    assert "asr_degrade" not in option_defaults(cfg)

"""Golden-file integration tests (imp.md 13): 3 short LibriSpeech dev-clean clips.

Needs Whisper (scripts/prefetch_models.py) and our trained models (experiments/train_*.py);
run with  pytest -m integration
Thresholds are regression guards, not targets: Mode B WER < 10 %, Mode A removes >= 30 % of the
injected word errors (objective O1), recovers the injected offset within 0.6 s (coarse) and puts >= 70 %
of cue onsets within ±250 ms of the MFA reference; formatting rules always hold.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from metrics.format_metrics import format_report
from metrics.text_metrics import wer_cer
from metrics.timing import cue_timing, summarize_cue_timing
from subfix.config import Config
from subfix.models import Models
from subfix.pipeline import run
from subfix.subio import read_srt

pytestmark = pytest.mark.integration
FIX = Path(__file__).parent / "fixtures"
CLIPS = [1, 2, 3]


@pytest.fixture(scope="module")
def env():
    cfg = Config.load()
    return cfg, Models(cfg)


def text(cues) -> str:
    return " ".join(c.text.replace("\n", " ") for c in cues)


def check_format_and_order(cues, cfg, duration):
    fmt = format_report(cues, cfg)
    assert fmt["fmt_lines_ok"] == 1.0 and fmt["fmt_nlines_ok"] == 1.0 and fmt["fmt_gap_ok"] == 1.0
    for a, b in zip(cues, cues[1:]):
        assert a.start < a.end <= b.start
    assert cues[-1].end <= duration + 1.0


@pytest.mark.parametrize("n", CLIPS)
def test_mode_b_audio_only(n, env):
    cfg, models = env
    res = run(FIX / f"clip{n}.flac", None, cfg, models)
    ref = read_srt(FIX / f"clip{n}.ref.srt")
    assert wer_cer([text(ref)], [text(res.cues)])["wer"] < 0.10
    check_format_and_order(res.cues, cfg, res.audio_duration)


@pytest.mark.parametrize("n", CLIPS)
def test_mode_a_corrects_text_and_timing(n, env):
    cfg, models = env
    src = FIX / f"clip{n}.corrupted.srt"
    log = json.loads((FIX / f"clip{n}.corrupted.log.json").read_text(encoding="utf-8"))
    res = run(FIX / f"clip{n}.flac", src, cfg, models)
    ref = read_srt(FIX / f"clip{n}.ref.srt")
    wer_in = wer_cer([text(ref)], [text(read_srt(src))])["wer"]
    wer_out = wer_cer([text(ref)], [text(res.cues)])["wer"]
    assert wer_out <= 0.7 * wer_in, (wer_in, wer_out)
    assert res.sync is not None and res.sync.method != "none"
    # coarse sync only: its audio-side anchor times are Whisper DTW timestamps (often 0.1-0.4 s early)
    # and these fixtures carry N(0, 300 ms) cue jitter; final timing comes from forced alignment below
    assert abs(res.sync.b + log["timing"]["offset"]) < 0.6
    ref_words = json.loads((FIX / f"clip{n}.words.json").read_text(encoding="utf-8"))
    t = summarize_cue_timing([cue_timing(ref, res.cues, ref_words)])
    assert t["t_within_250"] >= 0.7, t
    check_format_and_order(res.cues, cfg, res.audio_duration)

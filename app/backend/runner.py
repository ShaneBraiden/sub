"""Runs the subfix pipeline for one job and turns its RunResult into the JSON the UI reads.

Heavy imports (torch, faster-whisper, the text models) happen on the first job, so the server
starts instantly and the API tests can swap in a fake runner.
"""
from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any, Callable

from .schemas import JobOptions, diarization_status, option_defaults, to_overrides

log = logging.getLogger(__name__)

# pipeline timer names in run order -> what the UI shows
STAGES: list[tuple[str, str]] = [
    ("preprocess", "Converting the audio"),
    ("asr", "Transcribing the audio"),
    ("sync", "Syncing the subtitles to the audio"),
    ("fusion", "Comparing the subtitles with the audio"),
    ("punct", "Restoring punctuation"),
    ("truecase", "Fixing capital letters"),
    ("spell", "Checking spelling"),
    ("languagetool", "Checking grammar rules"),
    ("gec", "Grammar correction"),
    ("timing", "Timing every word"),
    ("diarize", "Identifying the speakers"),
    ("segment", "Splitting into subtitle cues"),
    ("chapters", "Finding the chapters"),
    ("export", "Writing the subtitle files"),
]
STAGE_LABELS = dict(STAGES)

Progress = Callable[[str, float], None]


class Cancelled(BaseException):
    """Raised from the progress callback to stop a run. A BaseException on purpose: the pipeline's
    timer swallows ordinary exceptions raised by progress callbacks."""


# ------------------------------------------------------------------ JSON helpers
def jsonable(x: Any) -> Any:
    """Plain JSON types only: numpy scalars -> Python, NaN/inf -> None, tuples -> lists."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return float(x) if math.isfinite(x) else None
    if hasattr(x, "item"):                       # numpy scalar
        return jsonable(x.item())
    return str(x)


def cue_json(c, with_words: bool = False) -> dict:
    d = {"index": c.index, "start": round(c.start, 3), "end": round(c.end, 3), "text": c.text}
    if getattr(c, "speakers", None):
        d["speakers"] = list(c.speakers)
        d["labels"] = list(c.labels)
    if with_words:
        d["words"] = [[w.text, None if w.start is None else round(w.start, 3),
                       None if w.end is None else round(w.end, 3)] for w in c.words]
    return d


def _tokens(toks) -> list[dict]:
    out = []
    for tok in toks:
        d = {"k": tok[0], "t": tok[1]}
        if len(tok) > 2 and tok[2]:
            d["tip"] = tok[2]
        out.append(d)
    return out


def sync_note(s) -> str | None:
    """One line on what the coarse sync changed (Mode A)."""
    if s is None or s.method == "none":
        return None
    drift = f", speed x{s.a:.4f} (frame-rate drift)" if abs(s.a - 1) > 5e-4 else ""
    if len(s.pieces) > 1:
        return f"Re-synced in {len(s.pieces)} parts (offset jumps found){drift}"
    if abs(s.b) < 0.05 and not drift:
        return "Already in sync"
    return f"Shifted {s.b:+.2f} s{drift}"


def build_result(res, cfg) -> dict:
    from metrics.format_metrics import format_report
    from subfix.diff import diff_rows, summary_counts
    from subfix.pipeline import raw_whisper_cues

    seg = cfg.segment
    rows = diff_rows(res, seg.max_cps)
    if res.mode == "A":
        before = res.orig_cues or []
    else:
        before = raw_whisper_cues(res.asr) if res.asr else []
    sync = None
    if res.sync is not None:
        sync = dict(res.sync.to_json(), note=sync_note(res.sync))
    asr_info = dict(res.asr.info) if res.asr else {}
    asr_info["words"] = len(res.asr.words) if res.asr else 0
    return jsonable({
        "mode": res.mode,
        "audio_duration": res.audio_duration,
        "processing_time": res.total_time,
        "rtf": res.rtf,
        "stage_times": res.stage_times,
        "sync": sync,
        "timing": res.timing_stats,
        "asr": asr_info,
        "counts": summary_counts(rows),
        "edit_counts": res.edit_counts(),
        "format_before": format_report(before, cfg) if before else {},
        "format_after": format_report(res.cues, cfg) if res.cues else {},
        "limits": {"max_cps": seg.max_cps, "lenient_cps": seg.lenient_cps, "max_chars": seg.max_chars,
                   "max_lines": seg.max_lines, "min_dur": seg.min_dur, "max_dur": seg.max_dur,
                   "min_gap": seg.min_gap, "cps_count_spaces": seg.cps_count_spaces},
        "rows": [{"index": r["index"], "start": r["start"], "end": r["end"], "old_start": r["old_start"],
                  "old_end": r["old_end"], "moved": r["moved"], "cps": r["cps"], "cps_bad": r["cps_bad"],
                  "orig": _tokens(r["orig"]), "new": _tokens(r["new"]),
                  **({"speakers": r["speakers"], "overlap": r["overlap"]} if r.get("speakers") else {})}
                 for r in rows],
        "original_cues": [cue_json(c) for c in before],
        "cues": [cue_json(c, with_words=True) for c in res.cues],
        "edits": res.edits,
        "speakers": getattr(res, "speakers", None),
        "chapters": None if getattr(res, "chapters", None) is None else [ch.to_json() for ch in res.chapters],
        "config": res.config,
    })


def waveform_peaks(wav_path: str | Path, bins: int = 2000) -> dict:
    """Max |amplitude| per bin, normalised to the loudest bin (for the timeline)."""
    import numpy as np
    from subfix.preprocess import SR, load_audio

    audio = np.abs(load_audio(str(wav_path)))
    n = len(audio)
    bins = max(1, min(bins, n))
    edges = np.linspace(0, n, bins + 1).astype(np.int64)
    peaks = np.maximum.reduceat(audio, edges[:-1]) if n else np.zeros(bins)
    top = float(peaks.max()) if len(peaks) else 0.0
    if top > 0:
        peaks = peaks / top
    return {"duration": n / SR, "peaks": [round(float(p), 3) for p in peaks]}


# ------------------------------------------------------------------ runner
class PipelineRunner:
    """Keeps one model registry alive between jobs (only the first job pays the loading time)."""

    def __init__(self):
        self._models = None
        self._device: dict | None = None
        self._lock = threading.Lock()

    def base_config(self):
        from subfix.config import Config
        return Config.load()

    def defaults(self) -> dict:
        return option_defaults(self.base_config())

    def diarization(self) -> dict:
        """Can speaker diarization run here? (our speaker model is trained)"""
        return diarization_status(self.base_config().diarize.model)

    def run(self, job: dict, media: Path, subs: Path | None, out_dir: Path, progress: Progress) -> dict:
        from subfix.export import write_outputs
        from subfix.models import Models
        from subfix.pipeline import run

        options = JobOptions(**job["options"])
        cfg = self.base_config()
        cfg = cfg.override(to_overrides(options, cfg))
        with self._lock:
            if self._models is None:
                self._models = Models(cfg)
            self._models.cfg = cfg
            models = self._models
        res = run(media, subs, cfg, models, lexicon=options.terms or None,
                  progress=lambda frac, desc="": progress(desc, frac))
        progress("export", 0.97)
        stem = Path(job["media_name"]).stem + ".corrected"
        paths = write_outputs(res, out_dir, stem)
        result = build_result(res, cfg)
        result["files"] = {kind: Path(p).name for kind, p in paths.items()}
        result["wav"] = str(res.wav)
        return result

    def loaded_models(self) -> list[str]:
        if self._models is None:
            return []
        return [f"{kind}: {key}" for kind, (key, inst) in self._models._slots.items() if inst is not None]

    def device_info(self) -> dict:
        """GPU name and memory (torch is imported on first use only)."""
        if self._device is None:
            info: dict[str, Any] = {"device": "cpu", "name": None}
            try:
                import torch
                if torch.cuda.is_available():
                    info = {"device": "cuda", "name": torch.cuda.get_device_name(0)}
            except Exception as exc:          # torch missing or broken: report CPU
                log.warning("torch unavailable: %s", exc)
            self._device = info
        info = dict(self._device)
        if info["device"] == "cuda":
            try:
                import torch
                free, total = torch.cuda.mem_get_info(0)
                info.update(memory_free_mb=free // 2**20, memory_total_mb=total // 2**20)
            except Exception:
                pass
        return info

"""ASR with word timestamps + word probabilities (faster-whisper) — imp.md 8.2.

Results are cached as JSON (keyed by audio content + decoding params) so experiments never
re-run ASR.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import text as T
from .paths import cache_dir
from .subio import Word
from .utils.cache import file_hash, load_json, save_json, text_hash
from .utils.device import resolve_device, setup_cuda_dlls

log = logging.getLogger(__name__)

_MODELS: dict[tuple[str, str, str], object] = {}
_NUM_CONT = re.compile(r"^[.,]\d")       # Whisper splits "10,000" into "10" + ",000"


def _continues_number(prev: Word | None, tok: str) -> bool:
    return prev is not None and bool(_NUM_CONT.match(tok)) and prev.text[-1:].isdigit()


def merge_pieces(words: list[Word]) -> list[Word]:
    """Re-attach punctuation-only pieces and number continuations to the previous word."""
    out: list[Word] = []
    for w in words:
        if out and (not T.norm(w.text) or _continues_number(out[-1], w.text)):
            prev = out[-1]
            prev.text += w.text
            prev.end = w.end
            prev.prob = min(prev.prob, w.prob)
            continue
        out.append(w)
    for i, w in enumerate(out):
        w.meta["asr_index"] = i
    return out
DEFAULT_PROMPT = "Hello, and welcome. Today, we will talk about it."


@dataclass
class AsrResult:
    words: list[Word]
    segments: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "words": [[w.text, w.start, w.end, w.prob] for w in self.words],
            "segments": self.segments,
            "info": self.info,
        }

    @classmethod
    def from_json(cls, d: dict) -> "AsrResult":
        words = [Word(t, s, e, p, "asr", {"asr_index": i}) for i, (t, s, e, p) in enumerate(d["words"])]
        return cls(merge_pieces(words), d.get("segments", []), d.get("info", {}))

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def resolve_model(model: str, device: str, compute_type: str) -> tuple[str, str]:
    if model == "auto":
        model = "large-v3-turbo" if device == "cuda" else "small.en"
    if compute_type == "auto":
        compute_type = "int8_float16" if device == "cuda" else "int8"
    return model, compute_type


def load_model(model: str, device: str, compute_type: str):
    key = (model, device, compute_type)
    if key not in _MODELS:
        if device == "cuda":
            setup_cuda_dlls()
        from faster_whisper import WhisperModel
        log.info("loading faster-whisper %s on %s (%s)", model, device, compute_type)
        _MODELS[key] = WhisperModel(model, device=device, compute_type=compute_type)
    return _MODELS[key]


def unload_models() -> None:
    _MODELS.clear()
    from .utils.device import free_gpu
    free_gpu()


def _run(wav: str, model: str, device: str, compute_type: str, beam_size: int, vad_filter: bool,
         language: str, hotwords: str | None, initial_prompt: str | None = None) -> AsrResult:
    wm = load_model(model, device, compute_type)
    t0 = time.perf_counter()
    kwargs = dict(language=language, beam_size=beam_size, word_timestamps=True,
                  vad_filter=vad_filter, condition_on_previous_text=False)
    if hotwords:
        kwargs["hotwords"] = hotwords
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    segments, info = wm.transcribe(wav, **kwargs)
    words: list[Word] = []
    segs: list[dict] = []
    for seg in segments:  # generator -> iterate once
        first = len(words)
        for w in seg.words or []:
            tok = w.word.strip()
            if not tok:
                continue
            if words and (not T.norm(tok) or _continues_number(words[-1], tok)):
                words[-1].text += tok          # punctuation-only piece / ",000" of a split number
                words[-1].end = float(w.end)
                words[-1].prob = min(words[-1].prob, float(w.probability))
                continue
            words.append(Word(tok, float(w.start), float(w.end), float(w.probability), "asr",
                              {"asr_index": len(words)}))
        segs.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip(),
                     "avg_logprob": float(seg.avg_logprob), "no_speech_prob": float(seg.no_speech_prob),
                     "words": [first, len(words)]})
    elapsed = time.perf_counter() - t0
    info_d = {"model": model, "device": device, "compute_type": compute_type,
              "language": getattr(info, "language", language),
              "duration": float(getattr(info, "duration", 0.0) or 0.0),
              "duration_after_vad": float(getattr(info, "duration_after_vad", 0.0) or 0.0),
              "elapsed": elapsed, "beam_size": beam_size, "vad_filter": vad_filter}
    return AsrResult(words, segs, info_d)


def transcribe(wav: str | Path, model: str = "auto", device: str = "auto", compute_type: str = "auto",
               beam_size: int = 5, vad_filter: bool = True, language: str = "en",
               hotwords: str | None = None, use_cache: bool = True,
               initial_prompt: str | None = DEFAULT_PROMPT) -> AsrResult:
    """Transcribe a 16 kHz WAV and return words with (start, end, prob)."""
    wav = str(wav)
    dev = resolve_device(device)
    model, ctype = resolve_model(model, dev, compute_type)
    key = text_hash(file_hash(wav), model, ctype, beam_size, vad_filter, language, hotwords or "",
                    initial_prompt or "")
    cpath = cache_dir(f"asr/{model.replace('/', '_')}") / f"{Path(wav).stem}-{key}.json"
    if use_cache and cpath.exists():
        return AsrResult.from_json(load_json(cpath))
    try:
        res = _run(wav, model, dev, ctype, beam_size, vad_filter, language, hotwords, initial_prompt)
    except RuntimeError as exc:  # missing CUDA DLLs etc. -> CPU fallback
        if dev != "cuda":
            raise
        log.warning("CUDA ASR failed (%s); falling back to CPU int8", exc)
        _MODELS.pop((model, dev, ctype), None)
        res = _run(wav, model, "cpu", "int8", beam_size, vad_filter, language, hotwords, initial_prompt)
    if use_cache:
        save_json(cpath, res.to_json())
    return res


def cached_result(wav: str | Path, model: str, **kw) -> AsrResult | None:
    """Return a cached ASR result if one exists for these params (no decoding)."""
    try:
        return transcribe(wav, model=model, use_cache=True, **kw) if _cache_exists(wav, model, **kw) else None
    except Exception:
        return None


def _cache_exists(wav, model, device="auto", compute_type="auto", beam_size=5, vad_filter=True,
                  language="en", hotwords=None, initial_prompt=DEFAULT_PROMPT) -> bool:
    dev = resolve_device(device)
    model, ctype = resolve_model(model, dev, compute_type)
    key = text_hash(file_hash(str(wav)), model, ctype, beam_size, vad_filter, language, hotwords or "",
                    initial_prompt or "")
    return (cache_dir(f"asr/{model.replace('/', '_')}") / f"{Path(wav).stem}-{key}.json").exists()

"""Request / response models of the HTTP API."""
from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]
TERMINAL = frozenset({"done", "failed", "cancelled"})


class JobOptions(BaseModel):
    """Per-job settings. None = use the server configuration (configs/default.yaml + tuned.yaml)."""

    model_config = ConfigDict(extra="forbid")

    max_cps: float | None = Field(None, ge=10, le=30, description="Reading speed limit (characters per second)")
    max_chars: int | None = Field(None, ge=20, le=80, description="Characters per line")
    max_lines: int | None = Field(None, ge=1, le=3, description="Lines per cue")
    min_duration: float | None = Field(None, ge=0.3, le=3.0, description="Shortest cue (s)")
    max_duration: float | None = Field(None, ge=2.0, le=15.0, description="Longest cue (s)")
    segmentation: Literal["dp", "greedy", "keep"] | None = None
    sync: bool | None = None
    fusion: bool | None = None
    punctuation: bool | None = None
    truecase: bool | None = None
    spelling: bool | None = None
    grammar_rules: bool | None = None
    grammar_model: bool | None = None
    timing: Literal["align", "whisper"] | None = None
    asr_model: Literal["auto", "large-v3-turbo", "small.en", "base.en", "tiny.en"] | None = None
    diarize: bool | None = None
    num_speakers: int | None = Field(None, ge=0, le=20, description="Number of speakers (0 = estimate)")
    speaker_labels: Literal["every", "change", "none"] | None = None
    chapters: bool | None = None
    num_chapters: int | None = Field(None, ge=0, le=50, description="Number of chapters (0 = choose automatically)")
    min_chapter: float | None = Field(None, ge=10, le=900, description="Shortest chapter (s)")
    terms: str = Field("", max_length=4000, description="Domain terms, comma or newline separated")

    @field_validator("terms")
    @classmethod
    def _strip_terms(cls, v: str) -> str:
        return v.strip()


# option -> dotted config key (options whose key is missing from the config are reported as unavailable)
CONFIG_KEYS: dict[str, str] = {
    "max_cps": "segment.max_cps",
    "max_chars": "segment.max_chars",
    "max_lines": "segment.max_lines",
    "min_duration": "segment.min_dur",
    "max_duration": "segment.max_dur",
    "segmentation": "segment.method",
    "sync": "sync.enabled",
    "fusion": "fusion.enabled",
    "punctuation": "punct.enabled",
    "truecase": "truecase.enabled",
    "spelling": "spell.symspell",
    "grammar_rules": "spell.languagetool",
    "grammar_model": "gec.enabled",
    "timing": "timing.backend",
    "asr_model": "asr.model",
    "diarize": "diarize.enabled",
    "num_speakers": "diarize.num_speakers",
    "speaker_labels": "diarize.label",
    "chapters": "chapters.enabled",
    "num_chapters": "chapters.num_chapters",
    "min_chapter": "chapters.min_duration",
}


def config_get(cfg: Any, dotted: str) -> Any:
    target = cfg
    for part in dotted.split("."):
        if not hasattr(target, part):
            raise AttributeError(dotted)
        target = getattr(target, part)
    return target


def has_key(cfg: Any, dotted: str) -> bool:
    try:
        config_get(cfg, dotted)
        return True
    except AttributeError:
        return False


def option_defaults(cfg: Any) -> dict[str, Any]:
    """Current configuration expressed as option values (what the UI shows before any change)."""
    out: dict[str, Any] = {}
    for opt, key in CONFIG_KEYS.items():
        if not has_key(cfg, key):
            continue
        value = config_get(cfg, key)
        if opt == "timing":
            value = "whisper" if value == "whisper" else "align"
        out[opt] = value
    return out


def to_overrides(options: JobOptions, cfg: Any) -> dict[str, Any]:
    """Dotted config overrides for the options that were set (and exist in this configuration)."""
    out: dict[str, Any] = {}
    for opt, key in CONFIG_KEYS.items():
        value = getattr(options, opt)
        if value is None:
            continue
        if not has_key(cfg, key):
            log.warning("option %s ignored: config key %s does not exist", opt, key)
            continue
        if opt == "timing":
            if value == "align":
                continue          # keep the configured forced aligner
            value = "whisper"
        out[key] = value
    return out


def option_fields() -> list[dict[str, Any]]:
    """Field metadata (type, range, choices) for the UI."""
    schema = JobOptions.model_json_schema()["properties"]
    fields = []
    for name, prop in schema.items():
        variants = prop.get("anyOf", [prop])
        main = next((v for v in variants if v.get("type") != "null"), variants[0])
        f: dict[str, Any] = {"key": name, "type": main.get("type", "string")}
        if "enum" in main:
            f["choices"] = main["enum"]
        for src, dst in (("minimum", "min"), ("maximum", "max"), ("maxLength", "max_length")):
            if src in main:
                f[dst] = main[src]
        if prop.get("description"):
            f["description"] = prop["description"]
        fields.append(f)
    return fields


def diarization_status(token_env: str = "HF_TOKEN") -> dict[str, bool]:
    """Whether speaker diarization can run: pyannote.audio installed, a Hugging Face token available
    (checked without importing pyannote or touching the network)."""
    import importlib.util
    import os

    try:
        installed = importlib.util.find_spec("pyannote.audio") is not None
    except (ImportError, ValueError):
        installed = False
    token = bool(token_env and os.environ.get(token_env))
    if not token:
        try:
            from huggingface_hub import get_token
            token = bool(get_token())
        except Exception:
            token = False
    return {"installed": installed, "token": token}


class CueIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(max_length=500)


class CuesIn(BaseModel):
    cues: list[CueIn] = Field(max_length=20000)

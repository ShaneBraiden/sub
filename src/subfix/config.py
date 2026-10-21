"""Typed configuration loaded from configs/default.yaml (+ configs/tuned.yaml)."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR


@dataclass
class AsrCfg:
    model: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    language: str = "en"
    initial_prompt: str = "Hello, and welcome. Today, we will talk about it."
    use_lexicon_hotwords: bool = True


@dataclass
class SyncCfg:
    enabled: bool = True
    ngram: int = 3
    min_anchors: int = 8
    residual_threshold: float = 1.0
    max_slope_dev: float = 0.1
    piecewise: bool = True
    piecewise_trigger: float = 0.85
    piecewise_min_anchors: int = 6
    piecewise_penalty: float = 0.3


@dataclass
class AlignCfg:
    gap: float = 0.8
    homophone_cost: float = 0.3
    near_miss_max: float = 0.5
    time_window: float = 3.0
    time_penalty: float = 1.0
    anchor_ngram: int = 3


@dataclass
class FusionCfg:
    enabled: bool = True
    w_conf: float = 1.0
    w_phon: float = 0.5
    w_lm: float = 0.3
    w_oov: float = 0.5
    w_lex: float = 1.0
    tau_sub: float = 1.0
    tau_ins: float = 0.7
    tau_lm_ins: float = 0.0
    tau_lm_del: float = 0.0
    del_span_conf: float = 0.9
    lm_context: int = 5
    lm_model: str = "models/lm/kn4.npz"
    lm_clip: float = 5.0


@dataclass
class PunctCfg:
    enabled: bool = True
    model: str = "models/punct"
    strategy_a: str = "fill"
    strategy_b: str = "replace"
    min_conf: float = 0.0
    chunk_words: int = 230
    overlap: int = 30


@dataclass
class TruecaseCfg:
    enabled: bool = True


@dataclass
class SpellCfg:
    symspell: bool = True
    max_edit_distance: int = 2
    min_count: int = 100000
    languagetool: bool = True
    lt_language: str = "en-US"
    lt_categories: list[str] = field(default_factory=lambda: ["TYPOS", "GRAMMAR", "CONFUSED_WORDS"])
    lt_rules: list[str] = field(default_factory=lambda: ["EN_A_VS_AN"])
    guard_prob: float = 0.8


@dataclass
class GecCfg:
    enabled: bool = True
    model: str = "models/corrector"
    max_ratio: float = 0.25
    batch_size: int = 16
    num_beams: int = 4
    max_length: int = 128
    max_sentence_words: int = 40
    guard_prob: float = 0.8


@dataclass
class TimingCfg:
    backend: str = "ctc"
    model: str = "models/aligner"
    star_penalty: float = 0.5
    chunk_max_s: float = 60.0
    chunk_margin_s: float = 1.0


@dataclass
class SegmentCfg:
    method: str = "dp"
    max_chars: int = 42
    max_lines: int = 2
    max_cps: float = 17.0
    lenient_cps: float = 20.0
    min_dur: float = 0.833
    max_dur: float = 7.0
    min_gap: float = 0.083
    lead_in: float = 0.0
    linger: float = 0.5
    max_words: int = 30
    cps_count_spaces: bool = False
    use_pos: bool = True
    pos_model: str = "models/pos/perceptron.json.gz"
    pause_bonus: float = 1.0
    cue_cost: float = 1.0


@dataclass
class DiarizeCfg:
    enabled: bool = False
    model: str = "pyannote/speaker-diarization-community-1"
    token_env: str = "HF_TOKEN"
    num_speakers: int = 0
    min_speakers: int = 0
    max_speakers: int = 0
    max_gap: float = 1.0
    overlap_min: float = 0.5
    smooth_cover: float = 0.5
    label: str = "every"
    label_format: str = "{speaker}: "
    dialogue: bool = True
    dialogue_cost: float = 0.5


@dataclass
class MeetingChaptersCfg:
    """chapters.mode = meeting: evidence besides the words, and its own (finer) chapter count."""
    penalty: float = 3.0
    pause_weight: float = 4.0
    cue_weight: float = 8.0
    pause_cap: float = 3.0
    cue_words: list[str] = field(default_factory=lambda: ["okay", "ok", "so", "now", "well", "right", "alright",
                                                          "anyway"])


@dataclass
class ChaptersCfg:
    enabled: bool = True
    mode: str = "talk"
    method: str = "dp"
    min_duration: float = 60.0
    num_chapters: int = 0
    penalty: float = 6.0
    window_words: int = 400
    smoothing: int = 4
    cutoff: float = 0.0
    content_pos: list[str] = field(default_factory=lambda: ["NOUN", "PROPN", "VERB", "ADJ"])
    stem: bool = True
    generic_logp: float = 0.0
    title_terms: int = 3
    meeting: MeetingChaptersCfg = field(default_factory=MeetingChaptersCfg)


@dataclass
class SummaryCfg:
    enabled: bool = True
    method: str = "embed"
    headline_words: int = 25
    overview_words: int = 150
    chapter_words: int = 50
    min_words: int = 10
    max_sentence_words: int = 40
    mmr_lambda: float = 0.5
    redundancy: float = 0.5
    damping: float = 0.85
    lsa_model: str = "models/summarizer/lsa.npz"
    abstractive: bool = False
    kind: str = "auto"


@dataclass
class LlmCfg:
    provider: str = "ollama"
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b-instruct"
    temperature: float = 0.0
    seed: int = 0
    num_ctx: int = 8192
    timeout: float = 600.0
    keep_alive: str = "5m"
    chunk_words: int = 2500
    unload_after: bool = True


@dataclass
class SentimentCfg:
    enabled: bool = True
    model: str = "models/sentiment"
    unit: str = "sentence"
    bandwidth: float = 0.0
    points: int = 400


# Flat aliases accepted by Config.replace(...) (e.g. from the Gradio UI).
ALIASES = {
    "use_gec": "gec.enabled",
    "use_punct": "punct.enabled",
    "use_fusion": "fusion.enabled",
    "use_sync": "sync.enabled",
    "use_languagetool": "spell.languagetool",
    "use_symspell": "spell.symspell",
    "asr_model": "asr.model",
    "gec_model": "gec.model",
    "timing_backend": "timing.backend",
    "segmenter": "segment.method",
    "use_diarize": "diarize.enabled",
}


@dataclass
class Config:
    device: str = "auto"
    cache_dir: str = "cache"
    keep_models_loaded: bool = True
    asr: AsrCfg = field(default_factory=AsrCfg)
    sync: SyncCfg = field(default_factory=SyncCfg)
    align: AlignCfg = field(default_factory=AlignCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    punct: PunctCfg = field(default_factory=PunctCfg)
    truecase: TruecaseCfg = field(default_factory=TruecaseCfg)
    spell: SpellCfg = field(default_factory=SpellCfg)
    gec: GecCfg = field(default_factory=GecCfg)
    timing: TimingCfg = field(default_factory=TimingCfg)
    segment: SegmentCfg = field(default_factory=SegmentCfg)
    diarize: DiarizeCfg = field(default_factory=DiarizeCfg)
    chapters: ChaptersCfg = field(default_factory=ChaptersCfg)
    summary: SummaryCfg = field(default_factory=SummaryCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    sentiment: SentimentCfg = field(default_factory=SentimentCfg)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path | None = None, use_tuned: bool = True,
             overrides: dict[str, Any] | None = None) -> "Config":
        cfg = cls()
        cfg._merge_yaml(CONFIG_DIR / "default.yaml")
        tuned = CONFIG_DIR / "tuned.yaml"
        if use_tuned and tuned.exists():
            cfg._merge_yaml(tuned)
        if path is not None:
            cfg._merge_yaml(Path(path))
        if overrides:
            cfg = cfg.override(overrides)
        return cfg

    def _merge_yaml(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _merge_into(self, data, str(path))

    # ------------------------------------------------------------------ updates
    def override(self, dotted: dict[str, Any]) -> "Config":
        """Return a copy with dotted-key overrides applied, e.g. {"segment.max_cps": 20}."""
        new = copy.deepcopy(self)
        for key, value in dotted.items():
            key = ALIASES.get(key, key)
            target: Any = new
            parts = key.split(".")
            for p in parts[:-1]:
                if not hasattr(target, p):
                    raise KeyError(f"unknown config section '{p}' in '{key}'")
                target = getattr(target, p)
            leaf = parts[-1]
            if not hasattr(target, leaf):
                raise KeyError(f"unknown config key '{key}'")
            setattr(target, leaf, _coerce(getattr(target, leaf), value))
        return new

    def replace(self, **kwargs: Any) -> "Config":
        """Flat overrides: section-unique field names, aliases, or section__field."""
        dotted: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in ALIASES:
                dotted[ALIASES[k]] = v
            elif "__" in k:
                dotted[k.replace("__", ".")] = v
            elif k in {f.name for f in fields(self)} and not is_dataclass(getattr(self, k)):
                dotted[k] = v
            else:
                owners = [f.name for f in fields(self)
                          if is_dataclass(getattr(self, f.name)) and hasattr(getattr(self, f.name), k)]
                if len(owners) != 1:
                    raise KeyError(f"ambiguous or unknown key '{k}' (sections: {owners})")
                dotted[f"{owners[0]}.{k}"] = v
        return self.override(dotted)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_into(obj: Any, data: dict[str, Any], src: str) -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"{src}: unknown config key '{key}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{src}: section '{key}' must be a mapping")
            _merge_into(current, value, src)
        else:
            setattr(obj, key, _coerce(current, value))


def _coerce(current: Any, value: Any) -> Any:
    if value is None or current is None:
        return value
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(value)) if isinstance(value, str) else int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list) and isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"

"""Lazy model registry: one instance per model kind, so a 6 GB GPU never holds two GEC models."""
from __future__ import annotations

import logging
from typing import Any, Callable

from .utils.device import free_gpu, resolve_device

log = logging.getLogger(__name__)


class Models:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self._slots: dict[str, tuple[Any, Any]] = {}   # kind -> (key, instance)

    def _get(self, kind: str, key: Any, factory: Callable[[], Any]) -> Any:
        slot = self._slots.get(kind)
        if slot is not None and slot[0] == key:
            return slot[1]
        if slot is not None:
            self.release(kind)
        log.info("loading %s: %s", kind, key)
        inst = factory()
        self._slots[kind] = (key, inst)
        return inst

    def release(self, *kinds: str) -> None:
        for kind in kinds or list(self._slots):
            slot = self._slots.pop(kind, None)
            if slot is not None and hasattr(slot[1], "close"):
                try:
                    slot[1].close()
                except Exception:
                    pass
        free_gpu()

    # ------------------------------------------------------------ factories
    def lm(self, cfg=None):
        cfg = cfg or self.cfg
        from .lm import LMScorer
        return self._get("lm", cfg.fusion.lm_model, lambda: LMScorer(cfg.fusion.lm_model))

    def spell(self, cfg=None):
        cfg = cfg or self.cfg
        from .textfix.spell import SpellChecker
        key = (cfg.spell.max_edit_distance, cfg.spell.min_count)
        return self._get("spell", key, lambda: SpellChecker(cfg.spell.max_edit_distance, cfg.spell.min_count))

    def languagetool(self, cfg=None):
        cfg = cfg or self.cfg
        from .textfix.spell import LanguageToolChecker
        key = (cfg.spell.lt_language, tuple(cfg.spell.lt_categories), tuple(cfg.spell.lt_rules))
        try:
            return self._get("languagetool", key,
                             lambda: LanguageToolChecker(cfg.spell.lt_language, list(cfg.spell.lt_categories),
                                                         list(cfg.spell.lt_rules)))
        except Exception as exc:
            log.warning("LanguageTool unavailable: %s", exc)
            self._slots["languagetool"] = (key, None)
            return None

    def punct(self, cfg=None):
        cfg = cfg or self.cfg
        from .textfix.punct import PunctuationRestorer
        return self._get("punct", cfg.punct.model, lambda: PunctuationRestorer(cfg.punct.model, self.device))

    def gec(self, cfg=None):
        cfg = cfg or self.cfg
        from .textfix.gec import GEC
        g = cfg.gec
        key = (g.model, g.num_beams, g.max_length)
        return self._get("gec", key, lambda: GEC(g.model, self.device, g.num_beams, g.max_length, g.batch_size))

    def aligner(self, backend: str, cfg=None):
        cfg = cfg or self.cfg
        from .timing import make_aligner
        key = (backend, cfg.timing.model, cfg.timing.star_penalty)
        return self._get("aligner", key, lambda: make_aligner(backend, cfg, self.device))

    def diarizer(self, cfg=None):
        """pyannote speaker-diarization pipeline (opt-in; needs pyannote.audio and a Hugging Face token)."""
        cfg = cfg or self.cfg
        from .diarize import PyannoteDiarizer
        d = cfg.diarize
        return self._get("diarizer", (d.model, d.token_env),
                         lambda: PyannoteDiarizer(d.model, self.device, d.token_env))

    def background(self, cfg=None):
        """Unigram log-probabilities of our KN LM for chapter segmentation and titles (read from
        the LM file without its higher orders), or None if the LM is missing."""
        cfg = cfg or self.cfg

        def load():
            from .chapters import Background
            from .lm import resolve_model_path
            return Background(resolve_model_path(cfg.fusion.lm_model))
        try:
            return self._get("background", cfg.fusion.lm_model, load)
        except Exception as exc:
            log.warning("LM unigrams unavailable (%s); chapter titles use word frequency only", exc)
            return None

    def pos_tagger(self, cfg=None):
        """Our averaged-perceptron POS tagger, or None (lexical fallback) if it is not trained."""
        cfg = cfg or self.cfg
        path = cfg.segment.pos_model

        def load():
            from .lm import resolve_model_path
            from .postag import PerceptronTagger
            return PerceptronTagger.load(resolve_model_path(path))
        try:
            return self._get("pos", path, load)
        except Exception as exc:
            log.warning("POS tagger unavailable (%s); using lexical POS fallback", exc)
            return None

    def summarizer(self, cfg=None):
        """Our LSA sentence embeddings for the extractive summaries, or None (TF-IDF fallback) if not trained."""
        cfg = cfg or self.cfg
        path = cfg.summary.lsa_model

        def load():
            from .lm import resolve_model_path
            from .summarize import LSA
            return LSA(resolve_model_path(path))
        try:
            return self._get("summarizer", path, load)
        except Exception as exc:
            log.warning("LSA model unavailable (%s); summaries use TF-IDF (python experiments/train_summarizer.py)", exc)
            return None

    def sentiment(self, cfg=None):
        """Our sentiment classifier, or None if it is not trained (python experiments/train_sentiment.py)."""
        cfg = cfg or self.cfg
        path = cfg.sentiment.model

        def load():
            from .lm import resolve_model_path
            from .sentiment import SentimentModel
            return SentimentModel(resolve_model_path(path))
        try:
            return self._get("sentiment", path, load)
        except Exception as exc:
            log.warning("sentiment model unavailable (%s): python experiments/train_sentiment.py", exc)
            return None

    def llm(self, cfg=None):
        """Client of the local LLM (Ollama) for the opt-in abstractive summaries."""
        cfg = cfg or self.cfg
        from .llm import Ollama
        c = cfg.llm
        if c.provider != "ollama":
            raise ValueError(f"unknown llm.provider {c.provider!r} (only ollama)")
        key = (c.host, c.model, c.temperature, c.seed, c.num_ctx, c.timeout, c.keep_alive)
        return self._get("llm", key, lambda: Ollama.from_config(c))

    def free_gpu_for_llm(self) -> None:
        """Unload Whisper and our GPU models so a local LLM gets the (6 GB) GPU to itself; they reload on the
        next job."""
        if self.device != "cuda":
            return
        from .asr import unload_models
        unload_models()
        loaded = [k for k in ("punct", "gec", "aligner", "diarizer") if k in self._slots]
        if loaded:                                    # release() without names would drop every model
            self.release(*loaded)
        else:
            free_gpu()
        log.info("freed GPU memory for the LLM")

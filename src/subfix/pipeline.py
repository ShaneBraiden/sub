"""Mode A / Mode B orchestration — imp.md 2 / 8.8.

Mode A: media + faulty subtitles -> ASR -> coarse sync -> alignment -> evidence fusion
Mode B: media only               -> ASR draft
then (both): punctuation -> truecasing -> SymSpell/LanguageTool -> our corrector (guarded)
          -> forced alignment of the corrected words -> [speaker diarization, opt-in]
          -> DP re-segmentation (speaker-aware, labelled) -> cue timing rules
          -> chapter markers from topic segmentation (second output, cfg.chapters)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import text as T
from .align import align_words
from .asr import AsrResult, transcribe
from .config import Config
from .diarize import Diarization
from .fuse import Fuser, passthrough
from .models import Models
from .preprocess import audio_duration, load_audio, to_wav
from .segment import keep_original, pos_tags, segment
from .subio import Cue, Word, cues_to_words, is_caseless, read_subs
from .sync import SyncResult, sync_cues
from .timing import word_times

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    mode: str
    cues: list[Cue]
    words: list[Word]
    orig_cues: list[Cue] | None
    orig_words: list[Word] | None
    asr: AsrResult | None
    sync: SyncResult | None
    edits: list[dict] = field(default_factory=list)
    stage_times: dict = field(default_factory=dict)
    audio_duration: float = 0.0
    wav: str = ""
    timing_stats: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    diarization: Diarization | None = None   # pyannote turns (cfg.diarize.enabled)
    speakers: dict | None = None             # diarize.speaker_report(): per-speaker stats, turns, overlaps
    chapters: list | None = None             # chapters.Chapter list (cfg.chapters.enabled)

    @property
    def total_time(self) -> float:
        return float(sum(self.stage_times.values()))

    @property
    def rtf(self) -> float:
        return self.total_time / max(self.audio_duration, 1e-6)

    def edit_counts(self) -> dict:
        out: dict[str, int] = {}
        for e in self.edits:
            if e.get("decision") == "reject" or e.get("op") == "guard":
                continue
            key = f"{e['stage']}:{e['op']}"
            out[key] = out.get(key, 0) + 1
        return out

    def to_log(self) -> dict:
        log = {
            "mode": self.mode,
            "audio_duration": self.audio_duration,
            "stage_times": self.stage_times,
            "rtf": self.rtf,
            "sync": self.sync.to_json() if self.sync else None,
            "timing": self.timing_stats,
            "edit_counts": self.edit_counts(),
            "edits": self.edits,
            "cues": [{"index": c.index, "start": round(c.start, 3), "end": round(c.end, 3), "text": c.text}
                     for c in self.cues],
            "words": [{"text": w.text, "start": None if w.start is None else round(w.start, 3),
                       "end": None if w.end is None else round(w.end, 3), "source": w.source,
                       "op": w.meta.get("op"), "orig_index": w.meta.get("orig_index")} for w in self.words],
            "config": self.config,
        }
        if self.speakers is not None:
            for row, c in zip(log["cues"], self.cues):
                row["speakers"] = c.speakers
            for row, w in zip(log["words"], self.words):
                row["speaker"] = w.meta.get("speaker")
                if w.meta.get("overlap"):
                    row["overlap_with"] = w.meta.get("overlap_with", [])
            log["speakers"] = self.speakers
        if self.chapters is not None:
            log["chapters"] = [ch.to_json() for ch in self.chapters]
        return log


class _Timer:
    def __init__(self, store: dict, progress: Callable | None):
        self.store = store
        self.progress = progress

    def __call__(self, name: str, frac: float | None = None):
        timer = self

        class _Ctx:
            def __enter__(self_inner):
                if timer.progress and frac is not None:
                    try:
                        timer.progress(frac, desc=name)
                    except Exception:
                        pass
                self_inner.t0 = time.perf_counter()

            def __exit__(self_inner, *exc):
                timer.store[name] = timer.store.get(name, 0.0) + time.perf_counter() - self_inner.t0
        return _Ctx()


def read_lexicon(lexicon) -> list[str]:
    if not lexicon:
        return []
    if isinstance(lexicon, (list, tuple)):
        return [t.strip() for t in lexicon if t and t.strip()]
    p = Path(str(lexicon))
    raw = p.read_text(encoding="utf-8") if p.exists() and p.is_file() else str(lexicon)
    return [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]


TEXT_SECTIONS = ("asr", "sync", "align", "fusion", "punct", "truecase", "spell", "gec")


def _words_to_json(words: list[Word]) -> list[dict]:
    return [{"text": w.text, "start": w.start, "end": w.end, "prob": w.prob, "source": w.source, "meta": w.meta}
            for w in words]


def _words_from_json(rows: list[dict]) -> list[Word]:
    return [Word(r["text"], r["start"], r["end"], r["prob"], r["source"], r["meta"]) for r in rows]


def _sync_from_json(d: dict | None) -> SyncResult | None:
    if not d:
        return None
    d = dict(d)
    d["pieces"] = [dict(p, until=float("inf") if p.get("until") is None else p["until"]) for p in d.get("pieces", [])]
    return SyncResult(**d)


def run(media_path: str | Path, sub_path: str | Path | None = None, cfg: Config | None = None,
        models: Models | None = None, lexicon=None, asr_result: AsrResult | None = None,
        progress: Callable | None = None, use_text_cache: bool = True) -> RunResult:
    cfg = cfg or Config.load()
    models = models or Models(cfg)
    stage_times: dict[str, float] = {}
    timer = _Timer(stage_times, progress)
    edits: list[dict] = []
    terms = read_lexicon(lexicon)

    with timer("preprocess", 0.02):
        wav = to_wav(media_path)
        dur = audio_duration(wav)

    with timer("asr", 0.05):
        if asr_result is None:
            hot = ", ".join(terms) if (terms and cfg.asr.use_lexicon_hotwords) else None
            asr_result = transcribe(wav, cfg.asr.model, cfg.device, cfg.asr.compute_type, cfg.asr.beam_size,
                                    cfg.asr.vad_filter, cfg.asr.language, hot,
                                    initial_prompt=cfg.asr.initial_prompt or None)
        asr_words = [w.copy() for w in asr_result.words]

    mode = "A" if sub_path else "B"
    orig_cues = orig_words = None
    sync_res: SyncResult | None = None
    synced = None
    caseless = False
    # text-stage cache: identical text configurations (e.g. timing/segmenter ablations) reuse the
    # corrected words and only redo forced alignment + segmentation
    from .paths import cache_dir
    from .utils.cache import file_hash, load_json, save_json, text_hash
    cfg_d = cfg.to_dict()
    tkey = text_hash(file_hash(wav), file_hash(sub_path) if sub_path else None,
                     {k: cfg_d[k] for k in TEXT_SECTIONS}, terms, asr_result.info.get("model"))
    tcache = cache_dir("text") / f"{Path(wav).stem}-{tkey}.json"
    if mode == "A":
        orig_cues = read_subs(sub_path)
        orig_words = cues_to_words(orig_cues)
    if use_text_cache and tcache.exists():
        d = load_json(tcache)
        words = _words_from_json(d["words"])
        edits = d["edits"]
        sync_res = _sync_from_json(d.get("sync"))
        synced = [Cue(c["index"], c["start"], c["end"], c["text"]) for c in d["synced"]] if d.get("synced") else None
        caseless = d.get("caseless", False)
        stage_times.update(d.get("stage_times", {}))
        return _finish(mode, words, edits, orig_cues, orig_words, synced, sync_res, asr_result, stage_times, timer,
                       dur, wav, cfg, models)
    if mode == "A":
        with timer("sync", 0.25):
            if cfg.sync.enabled:
                synced, sync_res = sync_cues(orig_cues, asr_words, cfg)
            else:
                synced = orig_cues
            synced = [Cue(c.index, c.start, c.end, c.text) for c in synced]
            sub_words = cues_to_words(synced)
            caseless = is_caseless(sub_words)
        with timer("fusion", 0.35):
            if cfg.fusion.enabled:
                use_time = sync_res is not None and sync_res.method != "none"
                ops = align_words(sub_words, asr_words, cfg, use_time=use_time)
                lm = models.lm() if cfg.fusion.w_lm != 0 else None
                spell = models.spell() if (cfg.fusion.w_oov != 0 or cfg.spell.symspell) else None
                fz = Fuser(cfg, lm=lm, spell=spell, lexicon=terms).fuse(sub_words, asr_words, ops, caseless)
                words = fz.words
                edits.extend(fz.edits)
            else:
                words = passthrough(sub_words)
    else:
        words = [w.copy(meta=dict(w.meta, op="asr", asr_text=w.text, asr_prob=w.prob)) for w in asr_words]
        from .textfix.truecase import mark_sentence_starts
        mark_sentence_starts(words)

    if cfg.punct.enabled and words:
        with timer("punct", 0.45):
            from .textfix.punct import restore
            strategy = cfg.punct.strategy_a if mode == "A" else cfg.punct.strategy_b
            words, ed = restore(words, models.punct(), strategy, cfg.punct.min_conf, cfg.punct.chunk_words,
                                cfg.punct.overlap)
            edits.extend(ed)
    if cfg.truecase.enabled and words:
        with timer("truecase", 0.5):
            from .textfix.truecase import truecase
            words, ed = truecase(words, caseless, models.spell())
            edits.extend(ed)
    if cfg.spell.symspell and words:
        with timer("spell", 0.55):
            from .textfix.common import protector
            words, ed = models.spell().correct_words(words, protect=protector(cfg.spell.guard_prob))
            edits.extend(ed)
    if cfg.spell.languagetool and words:
        with timer("languagetool", 0.6):
            lt = models.languagetool()
            if lt is not None:
                from .textfix.common import protector
                words, ed = lt.correct_words(words, protect=protector(cfg.spell.guard_prob))
                edits.extend(ed)
    if cfg.gec.enabled and words:
        with timer("gec", 0.65):
            from .textfix.gec import apply_gec
            from .textfix.common import protector
            words, ed = apply_gec(words, models.gec(), cfg.gec.max_ratio, cfg.gec.max_sentence_words,
                                  protect=protector(cfg.gec.guard_prob))
            edits.extend(ed)
            if not cfg.keep_models_loaded:
                models.release("gec")

    if use_text_cache:
        text_times = {k: v for k, v in stage_times.items() if k not in ("preprocess", "asr")}
        save_json(tcache, {"words": _words_to_json(words), "edits": edits,
                           "sync": sync_res.to_json() if sync_res else None,
                           "synced": [{"index": c.index, "start": c.start, "end": c.end, "text": c.text}
                                      for c in synced] if synced else None,
                           "caseless": caseless, "stage_times": text_times}, indent=None)
    return _finish(mode, words, edits, orig_cues, orig_words, synced, sync_res, asr_result, stage_times, timer,
                   dur, wav, cfg, models)


def _finish(mode, words, edits, orig_cues, orig_words, synced, sync_res, asr_result, stage_times, timer,
            dur, wav, cfg, models) -> RunResult:
    """Forced alignment + diarization + segmentation (never cached; diarization has its own cache)."""
    timing_stats: dict = {}
    backend = cfg.timing.backend
    audio = None
    with timer("timing", 0.75):
        if backend == "ctc":
            audio = load_audio(wav)
            words, timing_stats = word_times(words, wav, cfg, aligner=models.aligner(backend), audio=audio)
            if not cfg.keep_models_loaded:
                models.release("aligner")
        elif backend in ("whisper", "keep"):
            words, timing_stats = word_times(words, wav, cfg, backend="whisper")
            timing_stats["backend"] = backend
        else:
            raise ValueError(f"unknown timing backend {backend!r} (use ctc, whisper or keep)")

    diar = None
    if cfg.diarize.enabled and words:
        with timer("diarize", 0.83):
            from .diarize import assign_speakers, diarize
            d = cfg.diarize
            diar = diarize(wav, cfg, models, audio=audio)
            words = assign_speakers(words, diar, d.max_gap, d.overlap_min, d.smooth_cover)
            if not cfg.keep_models_loaded:
                models.release("diarizer")

    pos = None
    with timer("segment", 0.9):
        if cfg.segment.method == "keep" and mode == "A":
            cues = keep_original(words, synced or orig_cues, cfg, keep_times=(backend == "keep"))
        else:
            tagger = models.pos_tagger() if cfg.segment.use_pos else None
            pos = pos_tags(words, tagger)
            cues = segment(words, cfg, pos)
            if tagger is None:
                pos = None

    chapters = None
    if cfg.chapters.enabled and words:
        with timer("chapters", 0.93):
            from .chapters import make_chapters
            if pos is None:
                tagger = models.pos_tagger()
                pos = pos_tags(words, tagger) if tagger is not None else None
            chapters = make_chapters(words, cfg, pos, models.background(), cues, dur)

    speakers = None
    if diar is not None:
        from .diarize import speaker_report
        speakers = speaker_report(words, cues, diar, dur)
    return RunResult(mode=mode, cues=cues, words=words, orig_cues=orig_cues, orig_words=orig_words,
                     asr=asr_result, sync=sync_res, edits=edits, stage_times=stage_times,
                     audio_duration=dur, wav=wav, timing_stats=timing_stats, config=cfg.to_dict(),
                     diarization=diar, speakers=speakers, chapters=chapters)


def raw_whisper_cues(asr: AsrResult) -> list[Cue]:
    """Baseline B3 / Mode-B raw output: Whisper segments as cues, text and times untouched."""
    cues = []
    for seg in asr.segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        cues.append(Cue(len(cues) + 1, float(seg["start"]), float(seg["end"]), text))
    return cues


def words_text(words: list[Word]) -> str:
    return T.detokenize([w.text for w in words])

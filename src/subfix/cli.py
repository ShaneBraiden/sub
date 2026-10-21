"""Command line.

  subfix correct --media clip.mp4 [--subs clip.srt] --out outdir     # the full pipeline
  subfix summarize --subs talk.srt [--llm]                          # summaries of any subtitle file
  subfix sentiment --subs interview.srt                             # sentiment over time of any subtitle file
  subfix transcribe --media clip.mp4                                # raw Whisper SRT
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _setup() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _overrides(pairs: list[str] | None) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got '{p}'")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def cmd_correct(args) -> int:
    from .config import Config
    from .export import write_outputs
    from .pipeline import run
    overrides = _overrides(args.set)
    if args.diarize or args.speakers is not None or args.labels:
        overrides.setdefault("diarize.enabled", True)
    if args.speakers is not None:
        overrides["diarize.num_speakers"] = args.speakers
    if args.labels:
        overrides["diarize.label"] = args.labels
    _chapter_overrides(args, overrides)
    if args.no_summary:
        overrides["summary.enabled"] = False
    elif args.llm:
        overrides.update({"summary.enabled": True, "summary.abstractive": True})
    if args.no_sentiment:
        overrides["sentiment.enabled"] = False
    cfg = Config.load(args.config, use_tuned=not args.no_tuned, overrides=overrides)
    res = run(args.media, args.subs, cfg, lexicon=args.lexicon)
    stem = args.stem or Path(args.media).stem + ".corrected"
    paths = write_outputs(res, args.out, stem)
    from metrics.format_metrics import format_report
    fmt = format_report(res.cues, cfg)
    out = {"mode": res.mode, "cues": len(res.cues), "edits": res.edit_counts(),
           "rtf": round(res.rtf, 3), "stage_times": {k: round(v, 2) for k, v in res.stage_times.items()},
           "sync": res.sync.to_json() if res.sync else None, "format": fmt, "outputs": paths}
    if res.speakers is not None:
        out["speakers"] = {"totals": res.speakers["totals"],
                           "per_speaker": [{k: r[k] for k in ("speaker", "talk_time", "share", "words", "wpm",
                                                               "turns", "overlap_time", "interruptions")}
                                           for r in res.speakers["speakers"]]}
    if res.chapters is not None:
        out["chapters"] = [{"start": round(ch.start, 2), "title": ch.title} for ch in res.chapters]
    if res.summary is not None:
        out["summary"] = _summary_brief(res.summary)
    if res.sentiment is not None:
        from .sentiment import summary_line
        out["sentiment"] = summary_line(res.sentiment)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


def _chapter_overrides(args, overrides: dict) -> None:
    if args.no_chapters:
        overrides["chapters.enabled"] = False
        return
    if args.chapters is not None:
        overrides.update({"chapters.enabled": True, "chapters.num_chapters": args.chapters})
    if args.chapter_mode:
        overrides.update({"chapters.enabled": True, "chapters.mode": args.chapter_mode})


def _summary_brief(summary) -> dict:
    main = summary.abstractive or summary.extractive
    out = {"headline": main["headline"]["text"], "overview": main["overview"]["text"],
           "by": summary.abstractive["model"] if summary.abstractive else summary.extractive["method"]}
    if summary.llm_error:
        out["llm_error"] = summary.llm_error
    return out


def _read_transcript(path: str, cfg, with_chapters: bool):
    """Words (times interpolated inside the cues, speakers from labels), chapters and duration of a subtitle file."""
    from .chapters import make_chapters
    from .models import Models
    from .segment import pos_tags
    from .sentiment import labeled_words, read_labeled_subs
    cues = read_labeled_subs(path)
    if not cues:
        raise SystemExit(f"{path}: no cues")
    words = labeled_words(cues)
    duration = max(c.end for c in cues)
    models = Models(cfg)
    chapters = None
    if with_chapters and words:
        pos = pos_tags(words, models.pos_tagger())
        chapters = make_chapters(words, cfg, pos, models.background(), cues, duration)
    return cues, words, chapters, duration, models


def cmd_summarize(args) -> int:
    from .config import Config
    from .pipeline import summarize_words
    from .summarize import write_summary_files
    overrides = _overrides(args.set)
    _chapter_overrides(args, overrides)
    overrides["summary.abstractive"] = bool(args.llm)
    if args.method:
        overrides["summary.method"] = args.method
    if args.words:
        overrides["summary.overview_words"] = args.words
    cfg = Config.load(args.config, overrides=overrides)
    cues, words, chapters, duration, models = _read_transcript(args.subs, cfg, cfg.chapters.enabled)
    res = summarize_words(words, cfg, models, chapters, duration)
    stem = Path(args.subs).stem
    paths = write_summary_files(res, args.out, stem)
    print(json.dumps({**_summary_brief(res), "chapters": len(chapters or []), "outputs": paths},
                     indent=1, ensure_ascii=False))
    return 0


def cmd_sentiment(args) -> int:
    from .config import Config
    from .sentiment import summary_line, track, units_from_cues, write_sentiment_files
    overrides = _overrides(args.set)
    if args.unit:
        overrides["sentiment.unit"] = args.unit
    if args.bandwidth is not None:
        overrides["sentiment.bandwidth"] = args.bandwidth
    cfg = Config.load(args.config, overrides=overrides)
    cues, words, chapters, duration, models = _read_transcript(args.subs, cfg, args.chapters)
    model = models.sentiment()
    if model is None:
        raise SystemExit("no sentiment model: python experiments/train_sentiment.py")
    s = cfg.sentiment
    res = track(units_from_cues(cues, s.unit), model, duration, s.bandwidth, s.points, chapters)
    paths = write_sentiment_files(res, args.out, Path(args.subs).stem, png=not args.no_plot)
    print(json.dumps({**summary_line(res), "speakers": res.speakers, "outputs": paths}, indent=1, ensure_ascii=False))
    return 0


def cmd_transcribe(args) -> int:
    from .asr import transcribe
    from .pipeline import raw_whisper_cues
    from .preprocess import to_wav
    from .subio import write_srt
    wav = to_wav(args.media)
    res = transcribe(wav, model=args.model, device=args.device)
    out = Path(args.out or Path(args.media).with_suffix(".whisper.srt"))
    write_srt(raw_whisper_cues(res), out)
    print(json.dumps({"words": len(res.words), "info": res.info, "srt": str(out)}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    _setup()
    ap = argparse.ArgumentParser(prog="subfix", description="Audio-aware subtitle correction")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("correct", help="correct subtitles (Mode A) or generate them (Mode B)")
    c.add_argument("--media", required=True, help="audio/video file")
    c.add_argument("--subs", default=None, help="existing .srt/.vtt (omit for Mode B)")
    c.add_argument("--out", default="outputs", help="output directory")
    c.add_argument("--stem", default=None, help="output file stem")
    c.add_argument("--config", default=None, help="extra YAML layered over configs/default.yaml")
    c.add_argument("--set", nargs="*", help="dotted overrides, e.g. segment.max_cps=20 gec.enabled=false")
    c.add_argument("--lexicon", default=None, help="comma-separated domain terms or a text file")
    c.add_argument("--no-tuned", action="store_true", help="ignore configs/tuned.yaml")
    c.add_argument("--diarize", action="store_true",
                   help="label speakers with pyannote.audio (needs requirements-diarize.txt and HF_TOKEN)")
    c.add_argument("--speakers", type=int, default=None, metavar="N",
                   help="number of speakers, if known (implies --diarize; 0 = estimate)")
    c.add_argument("--labels", choices=["every", "change", "none"], default=None,
                   help="speaker labels on every cue, only at speaker changes, or none (implies --diarize)")
    c.add_argument("--chapters", type=int, default=None, metavar="N",
                   help="number of chapter markers (0 = choose automatically; chapters are on by default)")
    c.add_argument("--chapter-mode", choices=["talk", "meeting"], default=None,
                   help="talk: talks, lectures, audiobooks (default); meeting: meetings and discussions "
                        "(also uses silences and words like 'okay'/'so' that open a topic, finer chapters)")
    c.add_argument("--no-chapters", action="store_true", help="do not write chapter markers")
    c.add_argument("--llm", action="store_true",
                   help="also write abstractive summaries with the local LLM (Ollama, see llm: in the config)")
    c.add_argument("--no-summary", action="store_true", help="do not write summaries")
    c.add_argument("--no-sentiment", action="store_true", help="do not track sentiment over time")
    c.set_defaults(fn=cmd_correct)
    s = sub.add_parser("summarize", help="summaries (headline, overview, chapters) of an existing .srt/.vtt")
    s.add_argument("--subs", required=True, help=".srt/.vtt file (speaker labels such as 'SPEAKER_00:' are used)")
    s.add_argument("--out", default="outputs", help="output directory")
    s.add_argument("--llm", action="store_true", help="also write abstractive summaries with the local LLM")
    s.add_argument("--method", choices=["embed", "textrank", "tfidf"], default=None, help="extractive method")
    s.add_argument("--words", type=int, default=None, help="length of the overview (words)")
    s.add_argument("--chapters", type=int, default=None, metavar="N", help="number of chapters (0 = automatic)")
    s.add_argument("--chapter-mode", choices=["talk", "meeting"], default=None)
    s.add_argument("--no-chapters", action="store_true", help="no per-chapter summaries")
    s.add_argument("--config", default=None)
    s.add_argument("--set", nargs="*", help="dotted overrides, e.g. summary.chapter_words=40")
    s.set_defaults(fn=cmd_summarize)
    e = sub.add_parser("sentiment", help="sentiment over time of an existing .srt/.vtt (JSON, CSV, PNG plot)")
    e.add_argument("--subs", required=True, help=".srt/.vtt file (speaker labels give per-speaker curves)")
    e.add_argument("--out", default="outputs", help="output directory")
    e.add_argument("--unit", choices=["cue", "sentence"], default=None, help="score every cue or every sentence")
    e.add_argument("--bandwidth", type=float, default=None, help="smoothing width in seconds (0 = automatic)")
    e.add_argument("--chapters", action="store_true", help="find chapter markers and report each chapter's mean")
    e.add_argument("--no-plot", action="store_true", help="skip the PNG plot")
    e.add_argument("--config", default=None)
    e.add_argument("--set", nargs="*", help="dotted overrides, e.g. sentiment.points=600")
    e.set_defaults(fn=cmd_sentiment)
    t = sub.add_parser("transcribe", help="raw faster-whisper transcription to SRT")
    t.add_argument("--media", required=True)
    t.add_argument("--model", default="auto")
    t.add_argument("--device", default="auto")
    t.add_argument("--out", default=None)
    t.set_defaults(fn=cmd_transcribe)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

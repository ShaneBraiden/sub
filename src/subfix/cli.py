"""Command line: `subfix correct --media clip.mp4 [--subs clip.srt] --out outdir`."""
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
    if args.no_chapters:
        overrides["chapters.enabled"] = False
    elif args.chapters is not None:
        overrides.update({"chapters.enabled": True, "chapters.num_chapters": args.chapters})
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
    print(json.dumps(out, indent=1, ensure_ascii=False))
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
    c.add_argument("--no-chapters", action="store_true", help="do not write chapter markers")
    c.set_defaults(fn=cmd_correct)
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

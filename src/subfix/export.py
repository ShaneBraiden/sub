"""Export corrected cues (.srt / .vtt) + a JSON change log (+ RTTM and speaker statistics when
the run was diarized, + chapter markers as WebVTT / FFmpeg metadata / a timestamp list / JSON when
the media has at least two chapters, + the summary as Markdown / JSON, + sentiment over time as JSON,
CSV and a PNG plot)."""
from __future__ import annotations

from pathlib import Path

from .subio import Cue, write_srt, write_vtt
from .utils.cache import save_json


def write_both_formats(cues: list[Cue], out_dir: str | Path, stem: str = "corrected") -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    srt, vtt = out / f"{stem}.srt", out / f"{stem}.vtt"
    write_srt(cues, srt)
    write_vtt(cues, vtt)
    return {"srt": str(srt), "vtt": str(vtt)}


def write_outputs(result, out_dir: str | Path, stem: str = "corrected", html: bool = True) -> dict[str, str]:
    paths = write_both_formats(result.cues, out_dir, stem)
    log_path = Path(out_dir) / f"{stem}.changes.json"
    save_json(log_path, result.to_log())
    paths["json"] = str(log_path)
    if getattr(result, "diarization", None) is not None:
        from .diarize import to_rttm
        rttm = Path(out_dir) / f"{stem}.rttm"
        rttm.write_text(to_rttm(result.diarization, stem), encoding="utf-8", newline="\n")
        paths["rttm"] = str(rttm)
        sp = Path(out_dir) / f"{stem}.speakers.json"
        save_json(sp, result.speakers)
        paths["speakers"] = str(sp)
    chapters = getattr(result, "chapters", None)
    if chapters and len(chapters) >= 2:
        from .chapters import write_chapter_files
        paths.update(write_chapter_files(chapters, out_dir, stem))
    if getattr(result, "summary", None) is not None:
        from .summarize import write_summary_files
        paths.update(write_summary_files(result.summary, out_dir, stem))
    if getattr(result, "sentiment", None) is not None:
        from .sentiment import write_sentiment_files
        paths.update(write_sentiment_files(result.sentiment, out_dir, stem))
    if html:
        from .diff import html_side_by_side
        hp = Path(out_dir) / f"{stem}.diff.html"
        hp.write_text("<!doctype html><meta charset='utf-8'><title>Subtitle diff</title>"
                      + html_side_by_side(result), encoding="utf-8")
        paths["html"] = str(hp)
    return paths

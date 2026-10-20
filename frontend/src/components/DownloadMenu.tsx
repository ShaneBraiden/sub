import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { Result } from "../api/types";
import { IconChevron, IconDownload } from "./Icons";

export default function DownloadMenu({ jobId, result }: { jobId: string; result: Result }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const edited = !!result.edited_cues;

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);

  const item = (kind: string, label: string, sub: string, inline = false) =>
    result.files[kind] || kind.startsWith("edited_") ? (
      <a href={api.fileUrl(jobId, kind, inline)} target={inline ? "_blank" : undefined} rel="noreferrer" onClick={() => setOpen(false)}>
        <IconDownload />
        {label}
        <span className="m-sub">{sub}</span>
      </a>
    ) : null;

  return (
    <div className="menu-wrap" ref={ref}>
      <div className="row" style={{ gap: 0 }}>
        <a
          className="btn primary"
          style={{ borderTopRightRadius: 0, borderBottomRightRadius: 0 }}
          href={api.fileUrl(jobId, edited ? "edited_srt" : "srt")}
        >
          <IconDownload /> Download .srt
        </a>
        <button
          className="btn primary icon"
          style={{ borderTopLeftRadius: 0, borderBottomLeftRadius: 0, borderLeft: "1px solid rgba(255,255,255,.25)" }}
          onClick={() => setOpen((o) => !o)}
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label="More download formats"
        >
          <IconChevron style={{ transform: "rotate(90deg)" }} />
        </button>
      </div>
      {open && (
        <div className="menu" role="menu">
          <div className="m-label">{edited ? "With your edits" : "Subtitles"}</div>
          {item(edited ? "edited_srt" : "srt", "SubRip", ".srt")}
          {item(edited ? "edited_vtt" : "vtt", "WebVTT", ".vtt")}
          {edited && (
            <>
              <div className="m-label">Automatic version</div>
              {item("srt", "SubRip", ".srt")}
              {item("vtt", "WebVTT", ".vtt")}
            </>
          )}
          <hr />
          {item("json", "Change log", ".json")}
          {item("html", "Diff report", "opens in a tab", true)}
          {(result.files.speakers || result.files.rttm) && <div className="m-label">Speakers</div>}
          {item("speakers", "Speaker statistics", ".json")}
          {item("rttm", "Diarization", ".rttm")}
          {(result.files.chapters_vtt || result.files.ffmetadata) && <div className="m-label">Chapters</div>}
          {item("chapters_vtt", "WebVTT chapters", ".vtt")}
          {item("chapters_txt", "Timestamp list", "for a video description")}
          {item("ffmetadata", "FFmpeg chapters", ".ffmetadata")}
          {item("chapters_json", "Chapter details", ".json")}
        </div>
      )}
    </div>
  );
}

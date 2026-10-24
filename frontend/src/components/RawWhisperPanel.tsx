import { useState } from "react";
import type { AsrSegment, Result } from "../api/types";
import { formatClock } from "../lib/format";

interface Props {
  result: Result;
  onSeek: (t: number) => void;
}

type View = "paragraph" | "segments";

/** Whisper's untouched transcript, exactly as decoded (before sync, fusion, punctuation, casing,
 * spelling, grammar and segmentation), next to the final corrected text for comparison. */
export default function RawWhisperPanel({ result, onSeek }: Props) {
  const [view, setView] = useState<View>("paragraph");
  const [copied, setCopied] = useState(false);

  // older Mode-B jobs: the "before" cues are Whisper's segments unchanged
  const segments: AsrSegment[] | null =
    result.asr_raw?.segments ??
    (result.mode === "B"
      ? result.original_cues.map((c) => ({ start: c.start, end: c.end, text: c.text, avg_logprob: null, no_speech_prob: null }))
      : null);
  if (!segments) return <div className="empty">This job was run before the raw Whisper output was saved. Re-run it to see it here.</div>;
  if (!segments.length) return <div className="empty">Whisper heard no speech in this recording.</div>;

  const raw = result.asr_raw?.text ?? segments.map((s) => s.text).join(" ");
  const final = (result.edited_cues ?? result.cues).map((c) => c.text.replace(/\s*\n\s*/g, " ")).join(" ");
  const words = raw.split(/\s+/).filter(Boolean).length;
  const model = result.asr.model as string | undefined;

  const copy = () => {
    navigator.clipboard?.writeText(raw).then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      },
      () => {},
    );
  };

  return (
    <div className="raw-panel">
      <div className="spk-totals">
        {model && (
          <span>
            model <b>{model}</b>
          </span>
        )}
        <span>
          <b>{words.toLocaleString()}</b> words
        </span>
        <span>
          <b>{segments.length}</b> {segments.length === 1 ? "segment" : "segments"}
        </span>
      </div>
      <div className="raw-bar">
        <div className="raw-switch" role="group" aria-label="View">
          <button className={`btn sm${view === "paragraph" ? " primary" : ""}`} onClick={() => setView("paragraph")}>
            Paragraph
          </button>
          <button className={`btn sm${view === "segments" ? " primary" : ""}`} onClick={() => setView("segments")}>
            Segments
          </button>
        </div>
        <button className="btn sm" onClick={copy}>
          {copied ? "Copied" : "Copy text"}
        </button>
      </div>

      {view === "paragraph" ? (
        <>
          <h4 className="raw-h">Raw Whisper output</h4>
          <p className="raw-text">{raw}</p>
          <h4 className="raw-h">After our pipeline</h4>
          <p className="raw-text final">{final}</p>
        </>
      ) : (
        <ol className="raw-segs">
          {segments.map((s, i) => (
            <li key={i}>
              <button className="btn ghost sm mono" onClick={() => onSeek(s.start)} title="Jump here">
                {formatClock(s.start)}–{formatClock(s.end)}
              </button>
              <span className="raw-seg-text">{s.text}</span>
              {s.avg_logprob != null && (
                <span className="small muted" title="Whisper's average token probability for this segment">
                  {Math.round(Math.exp(s.avg_logprob) * 100)}%
                </span>
              )}
            </li>
          ))}
        </ol>
      )}
      <p className="small muted raw-note">
        Exactly as Whisper decoded it, before any of our models touched it (punctuation, casing, spelling, grammar, timing, speakers and cue
        splitting).
      </p>
    </div>
  );
}

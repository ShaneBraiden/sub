import type { Result } from "../api/types";
import { chapterAt, minChapter } from "../lib/chapters";
import { formatClock, formatDuration, plural } from "../lib/format";

interface Props {
  result: Result;
  time: number;
  onSeek: (t: number) => void;
}

/** Chapter markers: a strip along the recording and the list of chapters (title, length,
 * keywords and the sentence that best represents each one). Clicking jumps to the chapter. */
export default function ChaptersPanel({ result, time, onSeek }: Props) {
  const chapters = result.chapters;
  if (!chapters) return <div className="empty">This job was run without chapter markers.</div>;
  if (!chapters.length) return <div className="empty">No speech was found to split into chapters.</div>;
  const total = Math.max(result.audio_duration, chapters[chapters.length - 1].end, 0.1);
  const pct = (t: number) => `${(Math.max(0, Math.min(total, t)) / total) * 100}%`;
  const cur = chapterAt(chapters, time);
  const min = minChapter(result.config);
  const single = chapters.length < 2;

  return (
    <div className="ch-panel">
      {single ? (
        <p className="ch-note">
          Too short to split into chapters{min != null ? ` (each chapter must last at least ${formatDuration(min)})` : ""}. What it is about:
        </p>
      ) : (
        <>
          <div className="spk-totals">
            <span>
              <b>{chapters.length}</b> chapters
            </span>
            <span>
              average <b>{formatDuration(total / chapters.length)}</b>
            </span>
            {min != null && (
              <span>
                shortest allowed <b>{formatDuration(min)}</b>
              </span>
            )}
          </div>
          <div className="ch-strip" role="group" aria-label="Chapters along the recording">
            {chapters.map((c, i) => (
              <button
                key={c.index}
                className={`ch-seg${i === cur ? " on" : ""}${i % 2 ? " alt" : ""}`}
                style={{ left: pct(c.start), width: `calc(${pct(c.end)} - ${pct(c.start)})` }}
                title={`${c.index}. ${c.title} · ${formatClock(c.start)}–${formatClock(c.end)}`}
                aria-label={`Chapter ${c.index}: ${c.title}, at ${formatClock(c.start)}`}
                onClick={() => onSeek(c.start)}
              >
                {c.index}
              </button>
            ))}
            <span className="spk-head" style={{ left: pct(time) }} />
          </div>
        </>
      )}
      <ol className="ch-list">
        {chapters.map((c, i) => (
          <li key={c.index} className={i === cur ? "on" : undefined} aria-current={i === cur ? "true" : undefined}>
            <button className="ch-item" onClick={() => onSeek(c.start)} title={`Play from ${formatClock(c.start)}`}>
              <span className="ch-time num">{formatClock(c.start)}</span>
              <span className="ch-body">
                <span className="ch-title">
                  {!single && <span className="ch-num">{c.index}</span>}
                  {c.title}
                </span>
                <span className="ch-meta">
                  {formatDuration(c.end - c.start)}
                  {c.keywords.length > 0 && <> · {c.keywords.slice(0, 6).join(", ")}</>}
                </span>
                {c.summary && <span className="ch-summary">“{c.summary}”</span>}
              </span>
            </button>
          </li>
        ))}
      </ol>
      <p className="small muted">
        Chapters start where the vocabulary changes (topic segmentation of the corrected text); titles are the phrases most
        typical of each chapter. {single ? "" : `Download the ${plural(chapters.length, "chapter")} from the download menu: WebVTT chapters, FFmpeg metadata, or a timestamp list for a video description.`}
      </p>
    </div>
  );
}

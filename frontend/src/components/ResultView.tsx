import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Job, Result } from "../api/types";
import { nextChapterStart, previousChapterStart } from "../lib/chapters";
import { formatDuration, formatPercent, plural } from "../lib/format";
import { speakerColors } from "../lib/speakers";
import ChangesTable from "./ChangesTable";
import ChaptersPanel from "./ChaptersPanel";
import DiffTable from "./DiffTable";
import Player from "./Player";
import QualityPanel from "./QualityPanel";
import SpeakersPanel from "./SpeakersPanel";
import Timeline from "./Timeline";

type Tab = "subs" | "changes" | "speakers" | "chapters" | "quality";

function Stat({ label, value, sub, delta }: { label: string; value: string; sub?: string; delta?: string }) {
  return (
    <div className="card stat">
      <span className="s-label">{label}</span>
      <span className="s-value">
        {value}
        {delta && <span className="delta">{delta}</span>}
      </span>
      {sub && <span className="s-sub">{sub}</span>}
    </div>
  );
}

function gain(before?: number, after?: number): string | undefined {
  if (before == null || after == null) return undefined;
  const d = Math.round((after - before) * 100);
  return d > 0 ? `+${d} pts` : undefined;
}

export default function ResultView({ job, result, onSaved }: { job: Job; result: Result; onSaved: () => void }) {
  const mediaRef = useRef<HTMLVideoElement>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [tab, setTab] = useState<Tab>("subs");

  const seek = useCallback((t: number) => {
    const m = mediaRef.current;
    if (m) m.currentTime = t;
    setTime(t);
  }, []);

  const chapters = result.chapters && result.chapters.length >= 2 ? result.chapters : null;

  // keyboard: space = play/pause, arrows = +-2 s, [ ] = previous / next chapter (not while typing)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.closest("input, textarea, select, button, a, [contenteditable]") || e.ctrlKey || e.metaKey || e.altKey) return;
      const m = mediaRef.current;
      if (!m) return;
      if (e.key === " ") {
        e.preventDefault();
        if (m.paused) m.play().catch(() => {});
        else m.pause();
      } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        seek(Math.max(0, m.currentTime + (e.key === "ArrowLeft" ? -2 : 2)));
      } else if (chapters && (e.key === "[" || e.key === "]")) {
        e.preventDefault();
        const t = e.key === "[" ? previousChapterStart(chapters, m.currentTime) : nextChapterStart(chapters, m.currentTime);
        if (t != null) seek(t);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [seek, chapters]);

  const c = result.counts;
  const fb = result.format_before;
  const fa = result.format_after;
  // manual edits replace their cue; untouched cues keep their word times (for the word highlight)
  const cues = useMemo(() => {
    const auto = result.cues;
    const edited = result.edited_cues;
    if (!edited) return auto;
    if (edited.length !== auto.length) return edited;
    return edited.map((e, i) => (e.text === auto[i].text && e.start === auto[i].start && e.end === auto[i].end ? auto[i] : e));
  }, [result.cues, result.edited_cues]);
  const beforeLabel = result.mode === "A" ? "Original" : "Whisper";
  const appliedEdits = result.edits.filter((e) => e.decision !== "reject" && e.op !== "guard" && e.op !== "blocked").length;

  return (
    <>
      <div className="stats">
        {result.mode === "A" ? (
          <>
            <Stat label="Words corrected" value={c.sub.toLocaleString()} sub={`${c.ins} inserted · ${c.del} deleted`} />
            <Stat label="Punctuation & casing" value={c.fmt.toLocaleString()} sub="marks and capitals fixed" />
            <Stat label="Cues re-timed" value={c.moved_cues.toLocaleString()} sub={result.sync?.note ?? `of ${result.cues.length}`} />
          </>
        ) : (
          <>
            <Stat label="Subtitle cues" value={result.cues.length.toLocaleString()} sub={`${Number(result.asr.words ?? 0).toLocaleString()} words heard`} />
            <Stat label="Punctuation & casing" value={c.fmt.toLocaleString()} sub="changed vs. raw Whisper" />
            <Stat label="Words changed" value={(c.sub + c.ins + c.del).toLocaleString()} sub="spelling and grammar fixes" />
          </>
        )}
        <Stat label={`Reading speed ≤ ${result.limits.max_cps}`} value={formatPercent(fa.fmt_cps17_ok)} delta={gain(fb.fmt_cps17_ok, fa.fmt_cps17_ok)} sub={`${beforeLabel.toLowerCase()}: ${formatPercent(fb.fmt_cps17_ok)}`} />
        <Stat label={`Lines ≤ ${result.limits.max_chars} chars`} value={formatPercent(fa.fmt_lines_ok)} delta={gain(fb.fmt_lines_ok, fa.fmt_lines_ok)} sub={`${beforeLabel.toLowerCase()}: ${formatPercent(fb.fmt_lines_ok)}`} />
        <Stat label="Processing time" value={formatDuration(result.processing_time)} sub={`for ${formatDuration(result.audio_duration)} of audio`} />
      </div>

      <div className="workspace">
        <div className="left">
          <div className="card" style={{ overflow: "hidden" }}>
            <Player
              jobId={job.id}
              mediaName={job.media_name}
              mediaRef={mediaRef}
              time={time}
              duration={result.audio_duration}
              onTime={setTime}
              onPlaying={setPlaying}
              cues={cues}
              originalCues={result.original_cues}
              originalLabel={beforeLabel}
              wordTimes={result.cues.some((c) => !!c.words?.length)}
              speakerColors={speakerColors(result)}
              chapters={chapters}
              onSeek={seek}
            />
          </div>
          <div className="card">
            <Timeline
              jobId={job.id}
              duration={result.audio_duration}
              before={result.original_cues}
              after={cues}
              beforeLabel={beforeLabel}
              time={time}
              onSeek={seek}
              chapters={chapters}
            />
          </div>
          <p className="small muted">
            <span className="kbd">Space</span> play / pause · <span className="kbd">←</span> <span className="kbd">→</span> 2 s
            {chapters && (
              <>
                {" "}
                · <span className="kbd">[</span> <span className="kbd">]</span> chapter
              </>
            )}
          </p>
        </div>

        <div className="right card">
          <div className="tabs" role="tablist">
            <button role="tab" aria-selected={tab === "subs"} className={tab === "subs" ? "on" : ""} onClick={() => setTab("subs")}>
              Subtitles<span className="count">{result.cues.length}</span>
            </button>
            <button role="tab" aria-selected={tab === "changes"} className={tab === "changes" ? "on" : ""} onClick={() => setTab("changes")}>
              Change log<span className="count">{appliedEdits}</span>
            </button>
            {result.speakers && (
              <button role="tab" aria-selected={tab === "speakers"} className={tab === "speakers" ? "on" : ""} onClick={() => setTab("speakers")}>
                Speakers<span className="count">{result.speakers.totals.speakers}</span>
              </button>
            )}
            {result.chapters && (
              <button role="tab" aria-selected={tab === "chapters"} className={tab === "chapters" ? "on" : ""} onClick={() => setTab("chapters")}>
                Chapters<span className="count">{result.chapters.length}</span>
              </button>
            )}
            <button role="tab" aria-selected={tab === "quality"} className={tab === "quality" ? "on" : ""} onClick={() => setTab("quality")}>
              Quality & timing
            </button>
          </div>
          {tab === "subs" && <DiffTable jobId={job.id} result={result} time={time} playing={playing} onSeek={seek} onSaved={onSaved} />}
          {tab === "changes" && <ChangesTable edits={result.edits} onSeek={seek} mode={result.mode} />}
          {tab === "speakers" && <SpeakersPanel result={result} time={time} onSeek={seek} />}
          {tab === "chapters" && <ChaptersPanel result={result} time={time} onSeek={seek} />}
          {tab === "quality" && <QualityPanel result={result} />}
        </div>
      </div>
      <p className="small muted" style={{ marginTop: 14 }}>
        {plural(result.cues.length, "cue")} · {plural(appliedEdits, "change")} logged · job {job.id}
      </p>
    </>
  );
}

import { useEffect, useRef, useState, type RefObject } from "react";
import { api } from "../api/client";
import type { Chapter, CueData } from "../api/types";
import { chapterAt, nextChapterStart, previousChapterStart } from "../lib/chapters";
import { findCueAt, isVideo } from "../lib/cues";
import { formatClock } from "../lib/format";
import { IconPause, IconPlay, IconSkipBack, IconSkipForward, IconWave } from "./Icons";

export type CaptionSource = "corrected" | "original" | "off";

interface Props {
  jobId: string;
  mediaName: string;
  mediaRef: RefObject<HTMLVideoElement>;
  time: number;
  duration: number;
  onTime: (t: number) => void;
  onPlaying: (p: boolean) => void;
  cues: CueData[];
  originalCues: CueData[];
  originalLabel: string;
  wordTimes: boolean;          // highlight words as they are spoken (automatic cues only)
  speakerColors?: Map<string, string>;
  chapters?: Chapter[] | null;   // 2+ chapter markers: previous / next buttons and the current chapter's title
  onSeek?: (t: number) => void;
}

function Caption({ cue, time, karaoke, original, colors }: { cue: CueData; time: number; karaoke: boolean; original: boolean; colors?: Map<string, string> }) {
  const all = cue.text.split("\n");
  // speaker label prefixes are drawn apart from the spoken words (and are not part of the word timing)
  const prefix = all.map((l, i) => {
    const lab = cue.labels?.[i];
    return lab && l.startsWith(lab) ? lab : "";
  });
  const lines = all.map((l, i) => l.slice(prefix[i].length));
  const label = (i: number) => {
    if (!prefix[i]) return null;
    const who = cue.speakers?.[i];
    return (
      <span className="spk-label">
        {who && <span className="spk-dot" style={{ background: colors?.get(who) }} />}
        {prefix[i]}
      </span>
    );
  };
  const words = cue.words;
  const tokens = lines.map((l) => l.split(" ").filter(Boolean));
  const count = tokens.reduce((n, l) => n + l.length, 0);
  if (!karaoke || !words || words.length !== count) {
    return (
      <div className={`caption${original ? " original" : ""}`}>
        {lines.map((l, i) => (
          <div key={i}>
            <span className="line">
              {label(i)}
              {l}
            </span>
          </div>
        ))}
      </div>
    );
  }
  let k = 0;
  return (
    <div className="caption">
      {tokens.map((line, i) => (
        <div key={i}>
          <span className="line">
            {label(i)}
            {line.map((tok, j) => {
              const w = words[k++];
              const future = w[1] != null && w[1] > time;
              return (
                <span key={j} className={`w${future ? " future" : ""}`}>
                  {tok}
                  {j < line.length - 1 ? " " : ""}
                </span>
              );
            })}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function Player({
  jobId, mediaName, mediaRef, time, duration, onTime, onPlaying, cues, originalCues, originalLabel, wordTimes, speakerColors,
  chapters, onSeek,
}: Props) {
  const [src, setSrc] = useState(api.mediaUrl(jobId));
  const [fallback, setFallback] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [source, setSource] = useState<CaptionSource>("corrected");
  const [karaoke, setKaraoke] = useState(true);
  const [rate, setRate] = useState(1);
  const [mediaDur, setMediaDur] = useState(duration);
  const raf = useRef(0);
  const video = isVideo(mediaName) && !fallback;

  useEffect(() => {
    setSrc(api.mediaUrl(jobId));
    setFallback(false);
  }, [jobId]);

  // smooth playhead while playing
  useEffect(() => {
    if (!playing) return;
    const loop = () => {
      if (mediaRef.current) onTime(mediaRef.current.currentTime);
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, mediaRef, onTime]);

  useEffect(() => {
    if (mediaRef.current) mediaRef.current.playbackRate = rate;
  }, [rate, mediaRef]);

  const toggle = () => {
    const m = mediaRef.current;
    if (!m) return;
    if (m.paused) m.play().catch(() => {});
    else m.pause();
  };

  const seekTo = (t: number) => {
    if (onSeek) onSeek(t);
    else {
      if (mediaRef.current) mediaRef.current.currentTime = t;
      onTime(t);
    }
  };
  const chIdx = chapters ? chapterAt(chapters, time) : -1;
  const nextCh = chapters ? nextChapterStart(chapters, time) : null;

  const list = source === "original" ? originalCues : cues;
  const idx = source === "off" ? -1 : findCueAt(list, time);
  const total = mediaDur || duration;

  return (
    <div>
      <div className="screen" onClick={toggle}>
        <video
          ref={mediaRef}
          src={src}
          preload="metadata"
          playsInline
          style={video ? undefined : { position: "absolute", width: 1, height: 1, opacity: 0 }}
          onPlay={() => {
            setPlaying(true);
            onPlaying(true);
          }}
          onPause={() => {
            setPlaying(false);
            onPlaying(false);
          }}
          onTimeUpdate={(e) => !playing && onTime(e.currentTarget.currentTime)}
          onSeeked={(e) => onTime(e.currentTarget.currentTime)}
          onLoadedMetadata={(e) => Number.isFinite(e.currentTarget.duration) && setMediaDur(e.currentTarget.duration)}
          onError={() => {
            if (!fallback) {
              setFallback(true);
              setSrc(api.audioUrl(jobId));
            }
          }}
        />
        {!video && (
          <div className="audio-art">
            <IconWave />
            {fallback && isVideo(mediaName) ? "This browser cannot play the video; playing its audio" : "Audio only"}
          </div>
        )}
        {source !== "off" && <span className="caption-tag">{source === "original" ? originalLabel : "Corrected"}</span>}
        {idx >= 0 && (
          <Caption cue={list[idx]} time={time} karaoke={karaoke && wordTimes && source === "corrected"} original={source === "original"} colors={speakerColors} />
        )}
      </div>
      <div className="controls">
        <button className="btn primary icon sm" onClick={toggle} aria-label={playing ? "Pause" : "Play"} title="Play / pause (space)">
          {playing ? <IconPause /> : <IconPlay />}
        </button>
        <span className="clock num">
          {formatClock(time)} / {formatClock(total)}
        </span>
        <input
          className="scrub"
          type="range"
          min={0}
          max={total || 0}
          step={0.05}
          value={Math.min(time, total || 0)}
          onChange={(e) => {
            const t = parseFloat(e.target.value);
            if (mediaRef.current) mediaRef.current.currentTime = t;
            onTime(t);
          }}
          aria-label="Seek"
        />
        <div className="seg" role="group" aria-label="Captions">
          {(["corrected", "original", "off"] as CaptionSource[]).map((s) => (
            <button key={s} className={source === s ? "on" : ""} onClick={() => setSource(s)}>
              {s === "corrected" ? "Corrected" : s === "original" ? originalLabel : "Off"}
            </button>
          ))}
        </div>
        <select className="input sm" style={{ width: 74 }} value={rate} onChange={(e) => setRate(parseFloat(e.target.value))} aria-label="Speed">
          {[0.5, 0.75, 1, 1.25, 1.5, 2].map((r) => (
            <option key={r} value={r}>
              {r}×
            </option>
          ))}
        </select>
        {wordTimes && (
          <label className="toggle" style={{ padding: 0 }} title="Dim words that have not been spoken yet (shows the word timing)">
            <input type="checkbox" checked={karaoke} onChange={(e) => setKaraoke(e.target.checked)} />
            <span className="sw" />
            <span className="small">Word timing</span>
          </label>
        )}
      </div>
      {chapters && (
        <div className="ch-now">
          <button className="btn ghost icon sm" onClick={() => seekTo(previousChapterStart(chapters, time))} aria-label="Previous chapter" title="Previous chapter ([)">
            <IconSkipBack />
          </button>
          <button className="btn ghost icon sm" onClick={() => nextCh != null && seekTo(nextCh)} disabled={nextCh == null} aria-label="Next chapter" title="Next chapter (])">
            <IconSkipForward />
          </button>
          <span className="ch-label" aria-live="polite">
            {chIdx >= 0 && (
              <>
                <span className="ch-num">{chapters[chIdx].index}</span>
                <b>{chapters[chIdx].title}</b> · chapter {chIdx + 1} of {chapters.length}
              </>
            )}
          </span>
        </div>
      )}
    </div>
  );
}

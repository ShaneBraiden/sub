import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { Chapter, CueData, Waveform } from "../api/types";
import { formatClock } from "../lib/format";

interface Props {
  jobId: string;
  duration: number;
  before: CueData[];
  after: CueData[];
  beforeLabel: string;
  time: number;
  onSeek: (t: number) => void;
  chapters?: Chapter[] | null;
}

const H = 132;
const WAVE = [18, 66];
const LANE_B = [74, 96];
const LANE_A = [104, 126];
const ZOOM_S = 30;

function css(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

export default function Timeline({ jobId, duration, before, after, beforeLabel, time, onSeek, chapters }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [wave, setWave] = useState<Waveform | null>(null);
  const [width, setWidth] = useState(600);
  const [zoom, setZoom] = useState(true);
  const [hover, setHover] = useState<number | null>(null);
  const [scheme, setScheme] = useState(0);

  useEffect(() => {
    let alive = true;
    api.waveform(jobId).then((w) => alive && setWave(w)).catch(() => alive && setWave(null));
    return () => {
      alive = false;
    };
  }, [jobId]);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(200, Math.floor(e.contentRect.width))));
    ro.observe(el);
    // redraw with the new palette when the theme changes
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const bump = () => setScheme((s) => s + 1);
    mq.addEventListener("change", bump);
    const mo = new MutationObserver(bump);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      ro.disconnect();
      mq.removeEventListener("change", bump);
      mo.disconnect();
    };
  }, []);

  const total = Math.max(duration, wave?.duration ?? 0, 0.1);
  const span = zoom ? Math.min(ZOOM_S, total) : total;
  let t0 = zoom ? time - span / 3 : 0;
  t0 = Math.max(0, Math.min(t0, total - span));
  const x = (t: number) => ((t - t0) / span) * width;
  const tAt = (px: number) => t0 + (px / width) * span;

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    if (el.width !== Math.round(width * dpr) || el.height !== Math.round(H * dpr)) {
      el.width = Math.round(width * dpr);
      el.height = Math.round(H * dpr);
    }
    const ctx = el.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, H);
    const c = {
      bg: css("--surface-2"),
      wave: css("--wave"),
      before: css("--lane-before"),
      after: css("--lane-after"),
      head: css("--playhead"),
      muted: css("--muted"),
      border: css("--border"),
      fg: css("--fg"),
      chapter: css("--chapter"),
      chapterFg: css("--chapter-fg"),
    };
    ctx.fillStyle = c.bg;
    ctx.fillRect(0, 0, width, H);

    // time ticks
    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800].find((s) => span / s <= 8) ?? 3600;
    ctx.font = "10px ui-monospace, Consolas, monospace";
    ctx.textBaseline = "top";
    for (let t = Math.ceil(t0 / step) * step; t <= t0 + span; t += step) {
      const px = Math.round(x(t)) + 0.5;
      ctx.strokeStyle = c.border;
      ctx.beginPath();
      ctx.moveTo(px, 14);
      ctx.lineTo(px, H);
      ctx.stroke();
      ctx.fillStyle = c.muted;
      ctx.fillText(formatClock(t), px + 3, 3);
    }

    // waveform
    if (wave && wave.peaks.length) {
      const n = wave.peaks.length;
      const mid = (WAVE[0] + WAVE[1]) / 2;
      const amp = (WAVE[1] - WAVE[0]) / 2;
      ctx.fillStyle = c.wave;
      for (let px = 0; px < width; px += 2) {
        const a = Math.floor((tAt(px) / wave.duration) * n);
        const b = Math.max(a + 1, Math.floor((tAt(px + 2) / wave.duration) * n));
        let peak = 0;
        for (let k = Math.max(0, a); k < Math.min(n, b); k++) peak = Math.max(peak, wave.peaks[k]);
        const h = Math.max(1, peak * amp);
        ctx.fillRect(px, mid - h, 1.4, h * 2);
      }
    }

    // cue lanes
    const lane = (cues: CueData[], y: number[], color: string, label: string, highlight: boolean) => {
      ctx.globalAlpha = 1;
      for (const cue of cues) {
        if (cue.end < t0 || cue.start > t0 + span) continue;
        const a = Math.max(0, x(cue.start));
        const b = Math.min(width, x(cue.end));
        const on = highlight && time >= cue.start && time < cue.end;
        ctx.globalAlpha = on ? 1 : 0.55;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(a, y[0], Math.max(1.5, b - a - 1), y[1] - y[0], 3);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = c.fg;
      ctx.font = "600 10px system-ui, sans-serif";
      ctx.textBaseline = "middle";
      const tw = ctx.measureText(label).width;
      ctx.globalAlpha = 0.75;
      ctx.fillStyle = c.bg;
      ctx.fillRect(2, (y[0] + y[1]) / 2 - 7, tw + 8, 14);
      ctx.globalAlpha = 1;
      ctx.fillStyle = c.fg;
      ctx.fillText(label, 6, (y[0] + y[1]) / 2);
    };
    lane(before, LANE_B, c.before, beforeLabel, false);
    lane(after, LANE_A, c.after, "Corrected", true);

    // chapter markers: a dashed line at each chapter start, with its number and title where there is room
    if (chapters && chapters.length > 1) {
      ctx.font = "600 10px system-ui, sans-serif";
      ctx.textBaseline = "middle";
      let free = -Infinity;
      for (const ch of chapters) {
        if (ch.end < t0 || ch.start > t0 + span) continue;
        const px = Math.round(x(Math.max(ch.start, t0))) + 0.5;
        if (ch.start > 0 && ch.start >= t0) {
          ctx.strokeStyle = c.chapter;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(px, 14);
          ctx.lineTo(px, H);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.lineWidth = 1;
        }
        if (px < free + 6) continue;
        const room = Math.min(x(Math.min(ch.end, t0 + span)) - px - 4, 220);
        let label = `${ch.index} ${ch.title}`;
        while (label.length > 3 && ctx.measureText(label).width + 8 > room) label = `${label.slice(0, -2)}…`;
        if (ctx.measureText(label).width + 8 > room) continue;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = c.chapter;
        ctx.beginPath();
        ctx.roundRect(px + 1, 16, tw + 8, 14, 3);
        ctx.fill();
        ctx.fillStyle = c.chapterFg;
        ctx.fillText(label, px + 5, 23);
        free = px + tw + 9;
      }
    }

    // hover + playhead
    if (hover != null) {
      ctx.strokeStyle = c.muted;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hover + 0.5, 14);
      ctx.lineTo(hover + 0.5, H);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    const ph = Math.round(x(time)) + 0.5;
    ctx.strokeStyle = c.head;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(ph, 0);
    ctx.lineTo(ph, H);
    ctx.stroke();
    ctx.lineWidth = 1;
  });

  return (
    <div className="timeline">
      <div className="row between" style={{ marginBottom: 8 }}>
        <div className="legend">
          <span>
            <i style={{ background: "var(--lane-before)" }} />
            {beforeLabel}
          </span>
          <span>
            <i style={{ background: "var(--lane-after)" }} />
            Corrected
          </span>
          {chapters && chapters.length > 1 && (
            <span>
              <i style={{ background: "var(--chapter)" }} />
              Chapters
            </span>
          )}
          {hover != null && <span className="mono">{formatClock(tAt(hover))}</span>}
        </div>
        <div className="seg" role="group" aria-label="Timeline zoom">
          <button className={zoom ? "on" : ""} onClick={() => setZoom(true)}>
            {Math.round(Math.min(ZOOM_S, total))} s
          </button>
          <button className={!zoom ? "on" : ""} onClick={() => setZoom(false)}>
            Whole file
          </button>
        </div>
      </div>
      <canvas
        ref={canvas}
        data-scheme={scheme}
        style={{ height: H }}
        role="img"
        aria-label={`Waveform with the original and corrected subtitle cues${chapters && chapters.length > 1 ? " and chapter markers" : ""}. Click to jump to a time.`}
        onMouseMove={(e) => setHover(e.clientX - e.currentTarget.getBoundingClientRect().left)}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => onSeek(Math.max(0, tAt(e.clientX - e.currentTarget.getBoundingClientRect().left)))}
      />
    </div>
  );
}

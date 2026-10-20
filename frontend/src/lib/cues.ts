import type { CueData, DiffRow, Result } from "../api/types";
import { stripLabels } from "./speakers";

/** Index of the cue on screen at time t (cues sorted by start), or -1. */
export function findCueAt(cues: { start: number; end: number }[], t: number): number {
  let lo = 0;
  let hi = cues.length - 1;
  let best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cues[mid].start <= t) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return best >= 0 && t < cues[best].end ? best : -1;
}

/** Index of the last cue that started at or before t (the one to keep in view while playing). */
export function lastCueBefore(cues: { start: number }[], t: number): number {
  let lo = 0;
  let hi = cues.length - 1;
  let best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cues[mid].start <= t) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return best;
}

const VIDEO_EXTS = new Set(["mp4", "m4v", "mkv", "mov", "avi", "webm", "mpg", "mpeg", "ts", "wmv"]);

export function isVideo(name: string): boolean {
  return VIDEO_EXTS.has(name.split(".").pop()?.toLowerCase() ?? "");
}

export function extension(name: string): string {
  const ext = name.split(".").pop();
  return ext && ext !== name ? `.${ext.toLowerCase()}` : "";
}

/** The cues the user sees: their saved edits when present (same order as the automatic cues). */
export function currentCues(result: Result): CueData[] {
  return result.edited_cues ?? result.cues;
}

export function cueTextOf(row: DiffRow): string {
  return row.new.map((t) => t.t).join(" ");
}

/** Characters per second with the backend's convention (spaces excluded unless configured; speaker
 * labels, given as the per-line prefixes, are not counted). */
export function cps(text: string, start: number, end: number, countSpaces = false, labels?: string[]): number {
  const flat = stripLabels(text, labels).replace(/\n/g, " ");
  const chars = countSpaces ? flat.length : flat.replace(/ /g, "").length;
  return chars / Math.max(end - start, 1e-3);
}

export type RowFilter = "all" | "text" | "timing" | "cps" | "edited" | "overlap";

export function rowMatches(
  row: DiffRow,
  filter: RowFilter,
  query: string,
  edited: CueData | undefined,
  original: CueData | undefined,
): boolean {
  const changedText = row.new.some((t) => t.k !== "same") || row.orig.some((t) => t.k === "del");
  const isEdited = !!edited && !!original && (edited.text !== original.text || edited.start !== original.start || edited.end !== original.end);
  if (filter === "text" && !changedText) return false;
  if (filter === "timing" && !row.moved) return false;
  if (filter === "cps" && !row.cps_bad) return false;
  if (filter === "edited" && !isEdited) return false;
  if (filter === "overlap" && !row.overlap) return false;
  const q = query.trim().toLowerCase();
  if (!q) return true;
  const hay = [...row.orig.map((t) => t.t), ...row.new.map((t) => t.t), edited?.text ?? ""].join(" ").toLowerCase();
  return hay.includes(q);
}

/** WebVTT text for a <track> or download preview. */
export function toVtt(cues: { start: number; end: number; text: string }[]): string {
  const ts = (t: number) => {
    const ms = Math.round(Math.max(0, t) * 1000);
    const h = Math.floor(ms / 3600000);
    const m = Math.floor((ms % 3600000) / 60000);
    const s = Math.floor((ms % 60000) / 1000);
    const pad = (n: number, w = 2) => String(n).padStart(w, "0");
    return `${pad(h)}:${pad(m)}:${pad(s)}.${pad(ms % 1000, 3)}`;
  };
  return ["WEBVTT", "", ...cues.map((c) => `${ts(c.start)} --> ${ts(c.end)}\n${c.text}\n`)].join("\n");
}

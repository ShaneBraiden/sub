import type { Chapter } from "../api/types";

/** Index of the chapter playing at time t (the last one starting at or before t), or -1. */
export function chapterAt(chapters: Chapter[] | null | undefined, t: number): number {
  if (!chapters?.length) return -1;
  let lo = 0;
  let hi = chapters.length - 1;
  if (t < chapters[0].start) return -1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (chapters[mid].start <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** Where "previous chapter" goes from time t: the current chapter's start, or the one before it
 * when t is within the first 3 s of the current chapter (like a media player's previous button). */
export function previousChapterStart(chapters: Chapter[], t: number): number {
  const i = chapterAt(chapters, t);
  if (i < 0) return 0;
  if (t - chapters[i].start > 3 || i === 0) return chapters[i].start;
  return chapters[i - 1].start;
}

export function nextChapterStart(chapters: Chapter[], t: number): number | null {
  const i = chapterAt(chapters, t);
  return i + 1 < chapters.length ? chapters[i + 1].start : null;
}

/** The chapter mode the job was run with ("talk" when the configuration does not say). */
export function chapterMode(config: Record<string, unknown>): "talk" | "meeting" {
  const c = config.chapters as { mode?: string } | undefined;
  return c?.mode === "meeting" ? "meeting" : "talk";
}

/** The shortest chapter length the job was run with (seconds), from its configuration. */
export function minChapter(config: Record<string, unknown>): number | null {
  const c = config.chapters as { min_duration?: number } | undefined;
  return typeof c?.min_duration === "number" ? c.min_duration : null;
}

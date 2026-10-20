import type { Result } from "../api/types";

/** Categorical colour slots for speakers (--spk-1 ... --spk-8 in styles.css, validated for both themes). */
export const SPEAKER_SLOTS = 8;

/**
 * Speaker -> colour variable, by order of first appearance (the backend's order), so a speaker keeps its
 * colour whatever is filtered. Past eight speakers the rest share a neutral grey; the name is always shown.
 */
export function speakerColors(result: Pick<Result, "speakers">): Map<string, string> {
  const out = new Map<string, string>();
  (result.speakers?.speakers ?? []).forEach((s, i) => {
    out.set(s.speaker, i < SPEAKER_SLOTS ? `var(--spk-${i + 1})` : "var(--muted)");
  });
  return out;
}

/** Distinct speakers of a cue, in line order. */
export function cueSpeakers(speakers: (string | null)[] | undefined): string[] {
  return [...new Set((speakers ?? []).filter((s): s is string => !!s))];
}

/** Text without the speaker-label prefixes the backend added (labels[i] belongs to line i). */
export function stripLabels(text: string, labels: string[] | undefined): string {
  if (!labels?.length) return text;
  return text
    .split("\n")
    .map((line, i) => (labels[i] && line.startsWith(labels[i]) ? line.slice(labels[i].length) : line))
    .join("\n");
}

/** "SPEAKER_01" -> "Speaker 2" (pyannote counts from 00). Other names are shown as they are. */
export function speakerName(s: string): string {
  const m = s.match(/^SPEAKER_(\d+)$/);
  return m ? `Speaker ${parseInt(m[1], 10) + 1}` : s;
}

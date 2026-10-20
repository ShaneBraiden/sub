import type { JobOptions, JobStatus } from "../api/types";

export const STATUS_LABEL: Record<JobStatus, string> = {
  queued: "Queued",
  running: "Running",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

/** Edit-log stage names -> what people call them. */
export const EDIT_STAGE_LABEL: Record<string, string> = {
  fusion: "Audio comparison",
  punct: "Punctuation",
  truecase: "Capitalisation",
  symspell: "Spelling",
  spell: "Spelling",
  languagetool: "Grammar rules",
  gec: "Grammar model",
};

export const EDIT_OP_LABEL: Record<string, string> = {
  sub: "Replaced",
  ins: "Inserted",
  del: "Deleted",
  format: "Formatting",
  split: "Split",
  merge: "Merged",
  blocked: "Blocked by audio",
  guard: "Rejected (too many changes)",
};

export const STAGE_TIME_LABEL: Record<string, string> = {
  preprocess: "Audio conversion",
  asr: "Speech recognition",
  sync: "Subtitle sync",
  fusion: "Audio comparison",
  punct: "Punctuation",
  truecase: "Capitalisation",
  spell: "Spelling",
  languagetool: "Grammar rules",
  gec: "Grammar model",
  timing: "Word timing",
  diarize: "Speaker identification",
  segment: "Cue segmentation",
  chapters: "Chapter markers",
  export: "Export",
};

/** Pipeline stages that run for a job (Mode B has no sync/fusion; disabled steps are skipped; speaker
 * identification only when asked for). Stages the job has already reached are always kept. */
export function plannedStages(
  all: { key: string; label: string }[],
  mode: "A" | "B",
  options: JobOptions,
  reached: string[] = [],
): { key: string; label: string }[] {
  const off = new Set<string>();
  if (options.diarize !== true) off.add("diarize");
  if (options.chapters === false) off.add("chapters");
  if (mode === "B") {
    off.add("sync");
    off.add("fusion");
  }
  if (options.sync === false) off.add("sync");
  if (options.fusion === false) off.add("fusion");
  if (options.punctuation === false) off.add("punct");
  if (options.truecase === false) off.add("truecase");
  if (options.spelling === false) off.add("spell");
  if (options.grammar_rules === false) off.add("languagetool");
  if (options.grammar_model === false) off.add("gec");
  for (const k of reached) off.delete(k);
  return all.filter((s) => !off.has(s.key));
}

export const ASR_MODEL_LABEL: Record<string, string> = {
  auto: "Automatic",
  "large-v3-turbo": "large-v3-turbo (most accurate)",
  "small.en": "small.en (faster)",
  "base.en": "base.en (fast)",
  "tiny.en": "tiny.en (fastest, least accurate)",
};

export const SPEAKER_LABEL_STYLE: Record<string, string> = {
  every: "On every cue (SPEAKER_00: …)",
  change: "Only when the speaker changes",
  none: "No names (dashes in two-speaker cues)",
};

export const SEGMENTATION_LABEL: Record<string, string> = {
  dp: "Optimal (dynamic programming)",
  greedy: "Simple (greedy)",
  keep: "Keep the original cues",
};

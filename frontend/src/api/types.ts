// Shapes of the app.backend API (see app/backend/app.py).

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";
export type Mode = "A" | "B";

export interface JobOptions {
  max_cps?: number;
  max_chars?: number;
  max_lines?: number;
  min_duration?: number;
  max_duration?: number;
  segmentation?: "dp" | "greedy" | "keep";
  sync?: boolean;
  fusion?: boolean;
  punctuation?: boolean;
  truecase?: boolean;
  spelling?: boolean;
  grammar_rules?: boolean;
  grammar_model?: boolean;
  timing?: "align" | "whisper";
  asr_model?: string;
  diarize?: boolean;
  num_speakers?: number;
  speaker_labels?: "every" | "change" | "none";
  chapters?: boolean;
  /** talk: talks, lectures, audiobooks; meeting: meetings and discussions. */
  chapter_mode?: "talk" | "meeting";
  num_chapters?: number;
  min_chapter?: number;
  /** Extractive summaries (our own). */
  summary?: boolean;
  /** Also summaries written by the local LLM (Ollama). */
  llm_summary?: boolean;
  /** Length of the overview in words. */
  summary_words?: number;
  sentiment?: boolean;
  terms?: string;
}

export interface Counts {
  ins: number;
  del: number;
  sub: number;
  fmt: number;
  moved_cues: number;
  cps_over: number;
}

export interface JobSummary {
  audio_duration: number | null;
  rtf: number | null;
  cues: number;
  counts: Counts;
  sync_note: string | null;
  chapters?: number;
  /** One-sentence summary (the LLM's when the job made one). */
  headline?: string | null;
  /** Mean sentiment score, -1 (negative) .. +1 (positive). */
  sentiment?: number | null;
}

export interface StageEntry {
  key: string;
  label: string;
  started: number;
  seconds: number | null;
}

export interface Job {
  id: string;
  status: JobStatus;
  stage: string | null;
  stage_key: string | null;
  progress: number;
  created: number;
  started: number | null;
  finished: number | null;
  elapsed: number | null;
  media_name: string;
  subs_name: string | null;
  mode: Mode;
  options: JobOptions;
  error: string | null;
  summary: JobSummary | null;
  edited: number | null;
  stages: StageEntry[];
  cancelling: boolean;
  queue_position: number | null;
  version: number;
}

export interface OptionField {
  key: keyof JobOptions;
  type: string;
  choices?: string[];
  min?: number;
  max?: number;
  max_length?: number;
  description?: string;
}

export interface OptionsInfo {
  defaults: JobOptions;
  fields: OptionField[];
  stages: { key: string; label: string }[];
  media_exts: string[];
  subs_exts: string[];
  max_upload_mb: number;
  /** Can speaker diarization run on this server (pyannote.audio installed, Hugging Face token found)? */
  diarization?: { installed: boolean; token: boolean };
  /** Can the LLM summaries run (Ollama running and the model pulled)? */
  llm?: LlmStatus;
}

export interface LlmStatus {
  available: boolean;
  installed: boolean;
  model: string | null;
  error: string | null;
}

export interface SystemInfo {
  version: string;
  device: { device: "cuda" | "cpu"; name: string | null; memory_free_mb?: number; memory_total_mb?: number };
  models_loaded: string[];
  running: string | null;
  queued: number;
}

export type TokenKind = "same" | "fmt" | "ins" | "del" | "sub";

export interface Token {
  k: TokenKind;
  t: string;
  tip?: string;
}

export interface DiffRow {
  index: number;
  start: number;
  end: number;
  old_start: number | null;
  old_end: number | null;
  moved: boolean;
  cps: number;
  cps_bad: boolean;
  orig: Token[];
  new: Token[];
  /** Diarized jobs: speaker of each line, and whether the cue has overlapping speech. */
  speakers?: (string | null)[];
  overlap?: boolean;
}

export type TimedWord = [string, number | null, number | null];

export interface CueData {
  index: number;
  start: number;
  end: number;
  text: string;
  words?: TimedWord[];
  /** Diarized jobs: speaker of each line, and the label prefix shown on it ("" = none). */
  speakers?: (string | null)[];
  labels?: string[];
}

export interface SpeakerStats {
  speaker: string;
  talk_time: number;
  share: number;
  words: number;
  wpm: number | null;
  turns: number;
  mean_turn: number;
  longest_turn: number;
  overlap_time: number;
  interruptions: number;
  cues: number;
  first: number | null;
  last: number | null;
}

export interface SpeakerTurn {
  speaker: string | null;
  start: number | null;
  end: number | null;
  words: number;
  overlap: boolean;
  text: string;
}

export interface SpeakerReport {
  info: Record<string, unknown>;
  totals: {
    speakers: number;
    audio_duration: number;
    speech_time: number;
    overlap_time: number;
    overlap_share: number;
    turns: number;
    speaker_changes: number;
    mean_turn: number;
    overlap_words: number;
    overlap_cues: number;
    dialogue_cues: number;
  };
  speakers: SpeakerStats[];
  turns: SpeakerTurn[];
  overlaps: { start: number; end: number; speakers: string[] }[];
}

/** A chapter marker from topic segmentation (the second output). */
export interface Chapter {
  index: number;
  start: number;
  end: number;
  title: string;
  keywords: string[];
  /** The sentence that best represents the chapter. */
  summary: string;
  first_word: number;
  last_word: number;
  /** Strength of the topic shift at the chapter's start (0 for the first chapter). */
  score: number;
}

/** A sentence quoted by an extractive summary. */
export interface SummarySentence {
  index: number;
  start: number;
  end: number;
  text: string;
  speaker?: string;
}

export interface SummaryText {
  text: string;
  /** Extractive: the quoted sentences (indices into Summary.sentences), in time order. */
  sentences?: number[];
}

export interface SummaryChapter {
  index: number;
  start: number;
  end: number;
  title: string;
  text: string;
  sentences?: number[];
}

export interface SummaryVersion {
  headline: SummaryText;
  overview: SummaryText;
  chapters: SummaryChapter[];
}

/** Summaries at three granularities (the third output). */
export interface Summary {
  settings: { method: string; headline_words: number; overview_words: number; chapter_words: number };
  sentences: SummarySentence[];
  extractive: SummaryVersion & { method: string };
  abstractive: (SummaryVersion & { model: string; stats?: Record<string, number> }) | null;
  /** Why there is no LLM summary although one was asked for. */
  llm_error: string | null;
}

export type SentimentLabel = "neg" | "neu" | "pos";

export interface SentimentUnit {
  start: number;
  end: number;
  text: string;
  speaker: string | null;
  label: SentimentLabel;
  /** P(positive) - P(negative). */
  score: number;
  probs: [number, number, number];
}

export interface SentimentShare {
  negative: number;
  neutral: number;
  positive: number;
}

export interface SentimentMoment {
  kind: "positive" | "negative";
  time: number;
  value: number;
  start: number;
  end: number;
  text: string;
  speaker: string | null;
  score: number;
}

/** Sentiment over time (the fourth output). */
export interface Sentiment {
  bandwidth: number;
  duration: number;
  totals: { units: number; mean: number; share: SentimentShare };
  times: number[];
  curve: (number | null)[];
  speakers: { speaker: string; units: number; words: number; mean: number; share: SentimentShare }[];
  speaker_curves: Record<string, (number | null)[]>;
  chapters: { index: number; title: string; start: number; end: number; units: number; mean: number | null }[];
  moments: SentimentMoment[];
  units: SentimentUnit[];
}

export interface Edit {
  stage: string;
  op: string;
  old?: string | null;
  new?: string | null;
  decision?: string;
  rule?: string;
  prob?: number;
  score?: number;
  lm?: number;
  time?: number;
  index?: number;
  label?: string;
  rejected_sentences?: number;
  [key: string]: unknown;
}

export interface FormatReport {
  fmt_lines_ok?: number;
  fmt_nlines_ok?: number;
  fmt_cps17_ok?: number;
  fmt_cps20_ok?: number;
  fmt_dur_ok?: number;
  fmt_gap_ok?: number;
  fmt_all_ok?: number;
  n_cues?: number;
}

export interface SyncInfo {
  a: number;
  b: number;
  pieces: { until: number | null; a: number; b: number }[];
  n_anchors: number;
  inlier_ratio: number;
  residual_mad: number;
  method: string;
  note: string | null;
}

export interface Limits {
  max_cps: number;
  lenient_cps: number;
  max_chars: number;
  max_lines: number;
  min_dur: number;
  max_dur: number;
  min_gap: number;
  cps_count_spaces: boolean;
}

export interface Result {
  mode: Mode;
  audio_duration: number;
  processing_time: number;
  rtf: number;
  stage_times: Record<string, number>;
  sync: SyncInfo | null;
  timing: Record<string, unknown>;
  asr: Record<string, unknown>;
  counts: Counts;
  edit_counts: Record<string, number>;
  format_before: FormatReport;
  format_after: FormatReport;
  limits: Limits;
  rows: DiffRow[];
  original_cues: CueData[];
  cues: CueData[];
  edits: Edit[];
  speakers?: SpeakerReport | null;
  /** Chapter markers (null when the job ran without them). */
  chapters?: Chapter[] | null;
  /** Summaries (null when the job ran without them). */
  summary?: Summary | null;
  /** Sentiment over time (null when the job ran without it). */
  sentiment?: Sentiment | null;
  config: Record<string, unknown>;
  files: Record<string, string>;
  edited_cues: CueData[] | null;
}

export interface Waveform {
  duration: number;
  peaks: number[];
}

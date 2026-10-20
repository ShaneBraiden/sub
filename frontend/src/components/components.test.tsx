import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Chapter, Job, Result } from "../api/types";
import ChaptersPanel from "./ChaptersPanel";
import DiffTable from "./DiffTable";
import ProgressPanel from "./ProgressPanel";
import SpeakersPanel from "./SpeakersPanel";
import { ToastProvider } from "./Toasts";

afterEach(cleanup);

const stages = ["preprocess", "asr", "sync", "fusion", "punct", "timing", "segment", "export"].map((key) => ({ key, label: `L-${key}` }));

const job = (over: Partial<Job>): Job => ({
  id: "j1", status: "running", stage: "L-punct", stage_key: "punct", progress: 0.45, created: 0, started: Date.now() / 1000 - 5,
  finished: null, elapsed: 5, media_name: "a.wav", subs_name: null, mode: "B", options: {}, error: null, summary: null,
  edited: null, stages: [], cancelling: false, queue_position: null, version: 1, ...over,
});

describe("ProgressPanel", () => {
  it("marks finished, current, reused and pending stages", () => {
    const now = Date.now() / 1000;
    const j = job({
      stages: [
        { key: "preprocess", label: "L-preprocess", started: now - 5, seconds: 0.4 },
        { key: "asr", label: "L-asr", started: now - 4, seconds: 3.1 },
        { key: "timing", label: "L-timing", started: now - 1, seconds: null },
      ],
      stage_key: "timing",
      stage: "L-timing",
    });
    const onCancel = vi.fn();
    const { container } = render(<ProgressPanel job={j} stages={stages} onCancel={onCancel} />);
    const items = [...container.querySelectorAll(".stages li")].map((li) => `${li.className}:${li.textContent?.split("·")[0].replace(/[\d.]+ s$/, "").trim()}`);
    expect(items).toEqual([
      "done:L-preprocess", "done:L-asr", "skipped:L-punct", "current:L-timing", "pending:L-segment", "pending:L-export",
    ]);
    fireEvent.click(screen.getByText("Cancel"));
    expect(onCancel).toHaveBeenCalled();
  });

  it("shows the queue position", () => {
    render(<ProgressPanel job={job({ status: "queued", queue_position: 2, stage_key: null })} stages={stages} onCancel={() => {}} />);
    expect(screen.getByText(/2 jobs ahead/)).toBeTruthy();
    expect(screen.getByText("Remove from queue")).toBeTruthy();
  });
});

const result: Result = {
  mode: "A", audio_duration: 5, processing_time: 1, rtf: 0.2, stage_times: {}, sync: null, timing: {}, asr: {},
  counts: { ins: 0, del: 0, sub: 1, fmt: 0, moved_cues: 1, cps_over: 0 }, edit_counts: {}, format_before: {}, format_after: {},
  limits: { max_cps: 17, lenient_cps: 20, max_chars: 42, max_lines: 2, min_dur: 0.833, max_dur: 7, min_gap: 0.083, cps_count_spaces: false },
  rows: [
    { index: 1, start: 0.6, end: 2.1, old_start: 0.5, old_end: 2, moved: true, cps: 12, cps_bad: false,
      orig: [{ k: "same", t: "Hello" }, { k: "sub", t: "their," }], new: [{ k: "same", t: "Hello" }, { k: "sub", t: "there,", tip: "ASR: there, (p=0.95)" }] },
    { index: 2, start: 2.2, end: 4, old_start: 2.2, old_end: 4, moved: false, cps: 10, cps_bad: false,
      orig: [{ k: "same", t: "Bye." }], new: [{ k: "same", t: "Bye." }] },
  ],
  original_cues: [], cues: [
    { index: 1, start: 0.6, end: 2.1, text: "Hello there," },
    { index: 2, start: 2.2, end: 4, text: "Bye." },
  ],
  edits: [], config: {}, files: {}, edited_cues: null,
};

describe("DiffTable", () => {
  const renderTable = (onSeek = vi.fn()) =>
    render(
      <ToastProvider>
        <DiffTable jobId="j1" result={result} time={0} playing={false} onSeek={onSeek} onSaved={() => {}} />
      </ToastProvider>,
    );

  it("renders highlighted tokens and filters", () => {
    const { container } = renderTable();
    expect(container.querySelectorAll(".cue:not(.cue-head)").length).toBe(2);
    expect(container.querySelector(".tk.sub[data-tip]")?.textContent).toBe("there,");
    fireEvent.click(screen.getByText("Re-timed"));
    expect(container.querySelectorAll(".cue:not(.cue-head)").length).toBe(1);
    fireEvent.change(screen.getByLabelText("Search cues"), { target: { value: "zzz" } });
    expect(screen.getByText("No cues match.")).toBeTruthy();
  });

  it("seeks when a cue is clicked and edits in edit mode", () => {
    const onSeek = vi.fn();
    const { container } = renderTable(onSeek);
    fireEvent.click(container.querySelectorAll(".cue:not(.cue-head)")[1]);
    expect(onSeek).toHaveBeenCalledWith(2.2);
    fireEvent.click(screen.getByText("Edit cues"));
    const box = screen.getByLabelText("Text of cue 2") as HTMLTextAreaElement;
    fireEvent.change(box, { target: { value: "Goodbye." } });
    expect(screen.getByText("1 cue changed")).toBeTruthy();
    expect((screen.getByText("Save edits") as HTMLButtonElement).disabled).toBe(false);
    fireEvent.change(box, { target: { value: "  " } });
    expect((screen.getByText("Save edits") as HTMLButtonElement).disabled).toBe(true);
  });
});

const diarized: Result = {
  ...result,
  rows: [
    { ...result.rows[0], speakers: ["SPEAKER_00"], overlap: false },
    { ...result.rows[1], speakers: ["SPEAKER_01"], overlap: true },
  ],
  cues: [
    { index: 1, start: 0.6, end: 2.1, text: "SPEAKER_00: Hello there,", speakers: ["SPEAKER_00"], labels: ["SPEAKER_00: "] },
    { index: 2, start: 2.2, end: 4, text: "SPEAKER_01: Bye.", speakers: ["SPEAKER_01"], labels: ["SPEAKER_01: "] },
  ],
  speakers: {
    info: { model: "fake" },
    totals: { speakers: 2, audio_duration: 5, speech_time: 3.4, overlap_time: 0.2, overlap_share: 0.06, turns: 2, speaker_changes: 1, mean_turn: 1.6, overlap_words: 1, overlap_cues: 1, dialogue_cues: 0 },
    speakers: [
      { speaker: "SPEAKER_00", talk_time: 1.6, share: 0.47, words: 2, wpm: 75, turns: 1, mean_turn: 1.5, longest_turn: 1.5, overlap_time: 0.2, interruptions: 0, cues: 1, first: 0.5, last: 2.1 },
      { speaker: "SPEAKER_01", talk_time: 2, share: 0.59, words: 1, wpm: 30, turns: 1, mean_turn: 1.8, longest_turn: 1.8, overlap_time: 0.2, interruptions: 1, cues: 1, first: 2.0, last: 4 },
    ],
    turns: [
      { speaker: "SPEAKER_00", start: 0.6, end: 2.1, words: 2, overlap: false, text: "Hello there," },
      { speaker: "SPEAKER_01", start: 2.2, end: 4, words: 1, overlap: true, text: "Bye." },
    ],
    overlaps: [{ start: 2.0, end: 2.2, speakers: ["SPEAKER_00", "SPEAKER_01"] }],
  },
};

describe("speakers", () => {
  it("shows speaker chips, the overlap filter and a speaker filter in the cue list", () => {
    const { container } = render(
      <ToastProvider>
        <DiffTable jobId="j1" result={diarized} time={0} playing={false} onSeek={() => {}} onSaved={() => {}} />
      </ToastProvider>,
    );
    expect([...container.querySelectorAll(".c-spk .spk")].map((e) => e.textContent)).toEqual(["Speaker 1", "Speaker 2"]);
    expect(container.querySelector(".c-spk .chip.warn")?.textContent).toBe("overlap");
    fireEvent.click(screen.getByText("Overlap"));
    expect(container.querySelectorAll(".cue:not(.cue-head)").length).toBe(1);
    fireEvent.click(screen.getByText("All"));
    fireEvent.change(screen.getByLabelText("Show one speaker"), { target: { value: "SPEAKER_00" } });
    expect(container.querySelectorAll(".cue:not(.cue-head)").length).toBe(1);
    // reading speed ignores the label: "Hellothere," = 11 chars / 1.5 s
    expect(container.querySelector(".c-cps")?.textContent).toBe("7.3");
  });

  it("renders the speaker panel and seeks from a turn", () => {
    const onSeek = vi.fn();
    const { container } = render(<SpeakersPanel result={diarized} time={1} onSeek={onSeek} />);
    expect(container.querySelectorAll(".spk-lane").length).toBe(3);         // two speakers + overlap lane
    expect(container.querySelectorAll(".spk-turn").length).toBe(2);
    expect(container.querySelectorAll(".spk-table tbody tr").length).toBe(2);
    expect(screen.getByText("Interruptions")).toBeTruthy();
    fireEvent.click(screen.getByText("Bye."));
    expect(onSeek).toHaveBeenCalledWith(2.2);
  });

  it("explains a job without speakers", () => {
    render(<SpeakersPanel result={result} time={0} onSeek={() => {}} />);
    expect(screen.getByText(/without speaker identification/)).toBeTruthy();
  });
});

const chapter = (index: number, start: number, end: number, title: string): Chapter => ({
  index, start, end, title, keywords: [title.toLowerCase(), "extra"], summary: `About ${title.toLowerCase()}.`, first_word: 0, last_word: 1, score: index > 1 ? 0.2 : 0,
});

describe("chapters", () => {
  const withChapters: Result = {
    ...result, audio_duration: 300, config: { chapters: { min_duration: 60 } },
    chapters: [chapter(1, 0, 100, "Climate"), chapter(2, 100, 220, "Vaccines"), chapter(3, 220, 300, "Energy")],
  };

  it("lists the chapters, marks the current one and seeks on click", () => {
    const onSeek = vi.fn();
    const { container } = render(<ChaptersPanel result={withChapters} time={150} onSeek={onSeek} />);
    expect(container.querySelectorAll(".ch-list li").length).toBe(3);
    expect(container.querySelectorAll(".ch-seg").length).toBe(3);
    expect(container.querySelector(".ch-list li.on .ch-title")?.textContent).toBe("2Vaccines");
    expect(container.querySelector(".ch-seg.on")?.getAttribute("aria-label")).toBe("Chapter 2: Vaccines, at 1:40");
    expect(screen.getByText("“About energy.”")).toBeTruthy();
    fireEvent.click(screen.getByText("Energy"));
    expect(onSeek).toHaveBeenCalledWith(220);
  });

  it("explains a recording too short to split, and a job without chapters", () => {
    const one = { ...withChapters, chapters: [chapter(1, 0, 40, "Climate")] };
    const { container } = render(<ChaptersPanel result={one} time={0} onSeek={() => {}} />);
    expect(screen.getByText(/Too short to split into chapters/).textContent).toContain("at least 1 min");
    expect(container.querySelector(".ch-strip")).toBeNull();
    cleanup();
    render(<ChaptersPanel result={result} time={0} onSeek={() => {}} />);
    expect(screen.getByText(/without chapter markers/)).toBeTruthy();
  });
});

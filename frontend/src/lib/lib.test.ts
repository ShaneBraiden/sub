import { describe, expect, it } from "vitest";
import type { DiffRow } from "../api/types";
import { changedOptions } from "../components/OptionsForm";
import { chapterAt, minChapter, nextChapterStart, previousChapterStart } from "./chapters";
import { cps, extension, findCueAt, isVideo, lastCueBefore, rowMatches, toVtt } from "./cues";
import { formatClock, formatDuration, formatPercent, formatRelative, formatShift, formatTime } from "./format";
import { plannedStages } from "./labels";
import { cueSpeakers, speakerColors, speakerName, stripLabels } from "./speakers";

describe("format", () => {
  it("formats subtitle times", () => {
    expect(formatTime(75.25)).toBe("01:15.250");
    expect(formatTime(3675.5)).toBe("1:01:15.500");
    expect(formatTime(null)).toBe("–");
    expect(formatTime(-1)).toBe("00:00.000");
  });
  it("formats clocks and durations", () => {
    expect(formatClock(75.9)).toBe("1:15");
    expect(formatClock(3675)).toBe("1:01:15");
    expect(formatDuration(8.42)).toBe("8.4 s");
    expect(formatDuration(252)).toBe("4 min 12 s");
    expect(formatDuration(7260)).toBe("2 h 1 min");
  });
  it("formats shifts and percentages", () => {
    expect(formatShift(0.25)).toBe("+250 ms");
    expect(formatShift(-2.5)).toBe("−2.50 s");
    expect(formatPercent(0.9712, 1)).toBe("97.1%");
    expect(formatPercent(undefined)).toBe("–");
  });
  it("formats relative times", () => {
    const now = 1_700_000_000;
    expect(formatRelative(now - 10, now)).toBe("just now");
    expect(formatRelative(now - 600, now)).toBe("10 min ago");
  });
});

describe("cues", () => {
  const cues = [
    { start: 1, end: 2 },
    { start: 2.5, end: 4 },
    { start: 6, end: 7 },
  ];
  it("finds the cue on screen", () => {
    expect(findCueAt(cues, 0.5)).toBe(-1);
    expect(findCueAt(cues, 1)).toBe(0);
    expect(findCueAt(cues, 2.2)).toBe(-1);
    expect(findCueAt(cues, 3.9)).toBe(1);
    expect(findCueAt(cues, 6.5)).toBe(2);
    expect(findCueAt([], 1)).toBe(-1);
  });
  it("finds the last started cue", () => {
    expect(lastCueBefore(cues, 0)).toBe(-1);
    expect(lastCueBefore(cues, 5)).toBe(1);
    expect(lastCueBefore(cues, 100)).toBe(2);
  });
  it("detects media kinds", () => {
    expect(isVideo("talk.MP4")).toBe(true);
    expect(isVideo("talk.mp3")).toBe(false);
    expect(extension("a.b.SRT")).toBe(".srt");
    expect(extension("noext")).toBe("");
  });
  it("computes reading speed without spaces", () => {
    expect(cps("ab cd", 0, 2)).toBe(2);
    expect(cps("ab cd", 0, 2, true)).toBe(2.5);
    expect(cps("ab\ncd", 0, 1)).toBe(4);
  });
  it("writes WebVTT", () => {
    expect(toVtt([{ start: 61.5, end: 63, text: "Hi\nthere" }])).toBe("WEBVTT\n\n00:01:01.500 --> 00:01:03.000\nHi\nthere\n");
  });
  it("filters diff rows", () => {
    const row: DiffRow = {
      index: 1, start: 0, end: 1, old_start: 0, old_end: 1, moved: false, cps: 25, cps_bad: true,
      orig: [{ k: "sub", t: "their" }], new: [{ k: "sub", t: "there" }],
    };
    const cue = { index: 1, start: 0, end: 1, text: "there" };
    expect(rowMatches(row, "text", "", undefined, cue)).toBe(true);
    expect(rowMatches(row, "timing", "", undefined, cue)).toBe(false);
    expect(rowMatches(row, "cps", "THE", undefined, cue)).toBe(true);
    expect(rowMatches(row, "all", "nothing", undefined, cue)).toBe(false);
    expect(rowMatches(row, "edited", "", { ...cue, text: "There." }, cue)).toBe(true);
    expect(rowMatches(row, "edited", "", cue, cue)).toBe(false);
  });
});

describe("options", () => {
  it("sends only changed options the server knows", () => {
    const defaults = { max_cps: 17, grammar_model: true, segmentation: "dp" as const };
    expect(changedOptions({ max_cps: 17, grammar_model: false, terms: "x", timing: "whisper" }, defaults)).toEqual({ grammar_model: false });
  });
  it("plans the stages a job will run", () => {
    const all = ["preprocess", "asr", "sync", "fusion", "punct", "gec", "timing"].map((key) => ({ key, label: key }));
    expect(plannedStages(all, "B", {}).map((s) => s.key)).toEqual(["preprocess", "asr", "punct", "gec", "timing"]);
    expect(plannedStages(all, "A", { grammar_model: false, sync: false }).map((s) => s.key)).toEqual([
      "preprocess", "asr", "fusion", "punct", "timing",
    ]);
  });
});

describe("speakers", () => {
  it("strips labels for reading speed and names speakers", () => {
    expect(stripLabels("SPEAKER_00: Hi.\nSPEAKER_01: Yes.", ["SPEAKER_00: ", "SPEAKER_01: "])).toBe("Hi.\nYes.");
    expect(stripLabels("Edited: text", ["SPEAKER_00: "])).toBe("Edited: text");
    expect(cps("SPEAKER_00: abc", 0, 1, false, ["SPEAKER_00: "])).toBe(3);
    expect(cps("SPEAKER_00: abc", 0, 1)).toBe(14);
    expect(speakerName("SPEAKER_00")).toBe("Speaker 1");
    expect(speakerName("Alice")).toBe("Alice");
    expect(cueSpeakers(["A", "A", null, "B"])).toEqual(["A", "B"]);
  });
  it("gives each speaker a fixed colour slot, grey past eight", () => {
    const speakers = Array.from({ length: 9 }, (_, i) => ({ speaker: `SPEAKER_0${i}` }));
    const colors = speakerColors({ speakers: { speakers } } as never);
    expect(colors.get("SPEAKER_00")).toBe("var(--spk-1)");
    expect(colors.get("SPEAKER_07")).toBe("var(--spk-8)");
    expect(colors.get("SPEAKER_08")).toBe("var(--muted)");
  });
  it("plans speaker identification only when asked for or already reached", () => {
    const all = ["asr", "timing", "diarize", "segment"].map((key) => ({ key, label: key }));
    expect(plannedStages(all, "B", {}).map((s) => s.key)).toEqual(["asr", "timing", "segment"]);
    expect(plannedStages(all, "B", { diarize: true }).map((s) => s.key)).toEqual(["asr", "timing", "diarize", "segment"]);
    expect(plannedStages(all, "B", {}, ["diarize"]).map((s) => s.key)).toContain("diarize");
  });
});

describe("chapters", () => {
  const chs = [0, 100, 220].map((start, i, a) => ({
    index: i + 1, start, end: a[i + 1] ?? 300, title: "t", keywords: [], summary: "", first_word: 0, last_word: 0, score: 0,
  }));
  it("finds the chapter playing at a time", () => {
    expect(chapterAt(chs, 0)).toBe(0);
    expect(chapterAt(chs, 99.9)).toBe(0);
    expect(chapterAt(chs, 100)).toBe(1);
    expect(chapterAt(chs, 999)).toBe(2);
    expect(chapterAt([], 5)).toBe(-1);
    expect(chapterAt(null, 5)).toBe(-1);
  });
  it("jumps like a player's previous / next buttons", () => {
    expect(previousChapterStart(chs, 150)).toBe(100);   // well into chapter 2: back to its start
    expect(previousChapterStart(chs, 101)).toBe(0);     // just after its start: the chapter before
    expect(previousChapterStart(chs, 1)).toBe(0);
    expect(nextChapterStart(chs, 150)).toBe(220);
    expect(nextChapterStart(chs, 250)).toBeNull();
  });
  it("reads the shortest chapter length from the job configuration", () => {
    expect(minChapter({ chapters: { min_duration: 45 } })).toBe(45);
    expect(minChapter({})).toBeNull();
  });
  it("plans the chapter stage unless it is switched off", () => {
    const all = ["segment", "chapters", "export"].map((key) => ({ key, label: key }));
    expect(plannedStages(all, "B", {}).map((s) => s.key)).toEqual(["segment", "chapters", "export"]);
    expect(plannedStages(all, "B", { chapters: false }).map((s) => s.key)).toEqual(["segment", "export"]);
  });
});

"""Speaker diarization: who spoke when, mapped onto the timed words.

The diarizer is our own (src/subfix/speaker.py: a small speaker-embedding network trained from
scratch on LibriSpeech train-clean-100, plus clustering); no pretrained model is involved. It is
opt-in (cfg.diarize.enabled), runs on the in-memory 16 kHz waveform, and its output is cached as
JSON keyed by audio content + model file + diarization settings, like the ASR cache.

Two views of a diarization are kept:
  turns      regular diarization: turns of different speakers may overlap (overlapping speech)
  exclusive  at most one speaker at any time (derived from `turns` when not given) - used to give
             each word one speaker. Our diarizer finds one speaker at a time, so both are the same;
             the overlap handling below applies to turns that do overlap (e.g. imported RTTM)
Speakers are renamed SPEAKER_00, SPEAKER_01, ... in order of first appearance.

Each word then gets meta["speaker"]: the speaker with the largest overlap among the exclusive
turns. A word in a diarization gap takes the nearest turn within max_gap, else its neighbours'
speaker. Words with at least overlap_min of their duration inside overlapping speech get
meta["overlap"] and meta["overlap_with"]; a run of such words between two different speakers is
split at the most likely handover (sentence end > clause mark > longest pause), because there the
exclusive view cannot know whose words Whisper transcribed. A one-word island whose own turns cover
less than smooth_cover of it joins the speaker of the words on both sides. The segmenter never lets a cue span a speaker
change, except as a two-line dialogue cue with one speaker per line, and labels the lines
(cfg.diarize.label: every | change | none, see line_prefixes).
"""
from __future__ import annotations

import bisect
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import text as T
from .subio import Cue, Word

log = logging.getLogger(__name__)

SPEAKER_NAME = "SPEAKER_{:02d}"
LABEL_MODES = ("every", "change", "none")


@dataclass
class Turn:
    start: float
    end: float
    speaker: str

    @property
    def dur(self) -> float:
        return max(self.end - self.start, 0.0)


@dataclass
class Diarization:
    turns: list[Turn]                 # regular diarization (may overlap)
    exclusive: list[Turn]             # one speaker at a time
    info: dict = field(default_factory=dict)

    @property
    def speakers(self) -> list[str]:
        """Speaker names in order of first appearance."""
        seen: dict[str, None] = {}
        for t in self.turns:
            seen.setdefault(t.speaker, None)
        for t in self.exclusive:
            seen.setdefault(t.speaker, None)
        return list(seen)

    @classmethod
    def build(cls, turns: list[Turn], exclusive: list[Turn] | None = None, info: dict | None = None) -> "Diarization":
        """Sort, drop empty turns, rename speakers by first appearance, derive the exclusive view."""
        turns = sorted((t for t in turns if t.end > t.start), key=lambda t: (t.start, t.end))
        if exclusive is None:
            exclusive = exclusive_from_turns(turns)
        exclusive = _merge_adjacent(sorted((t for t in exclusive if t.end > t.start), key=lambda t: (t.start, t.end)))
        order: list[str] = []
        for t in turns + exclusive:
            if t.speaker not in order:
                order.append(t.speaker)
        names = {s: SPEAKER_NAME.format(k) for k, s in enumerate(order)}

        def rename(ts: list[Turn]) -> list[Turn]:
            return [Turn(float(t.start), float(t.end), names[t.speaker]) for t in ts]
        return cls(rename(turns), rename(exclusive), dict(info or {}))

    def to_json(self) -> dict:
        return {"turns": [[round(t.start, 3), round(t.end, 3), t.speaker] for t in self.turns],
                "exclusive": [[round(t.start, 3), round(t.end, 3), t.speaker] for t in self.exclusive],
                "info": self.info}

    @classmethod
    def from_json(cls, d: dict) -> "Diarization":
        return cls([Turn(*t) for t in d["turns"]], [Turn(*t) for t in d["exclusive"]], d.get("info", {}))


# ------------------------------------------------------------------ interval helpers
def _sweep(turns: list[Turn]):
    """Yield (start, end, active turns) for every elementary segment between turn boundaries."""
    turns = sorted(turns, key=lambda t: (t.start, t.end))
    bounds = sorted({x for t in turns for x in (t.start, t.end)})
    active: list[Turn] = []
    i = 0
    for a, b in zip(bounds, bounds[1:]):
        while i < len(turns) and turns[i].start <= a:
            active.append(turns[i])
            i += 1
        active = [t for t in active if t.end > a]
        if active and b > a:
            yield a, b, active


def _merge_adjacent(turns: list[Turn], tol: float = 1e-6) -> list[Turn]:
    out: list[Turn] = []
    for t in turns:
        if out and out[-1].speaker == t.speaker and t.start - out[-1].end <= tol:
            out[-1] = Turn(out[-1].start, max(out[-1].end, t.end), t.speaker)
        else:
            out.append(Turn(t.start, t.end, t.speaker))
    return out


def exclusive_from_turns(turns: list[Turn]) -> list[Turn]:
    """One speaker at a time: each overlap region goes to the speaker whose turn there is longest."""
    out = [Turn(a, b, max(active, key=lambda t: (t.dur, -t.start)).speaker) for a, b, active in _sweep(turns)]
    return _merge_adjacent(out)


def overlap_regions(turns: list[Turn]) -> list[tuple[float, float, tuple[str, ...]]]:
    """Stretches where two or more speakers talk at once: (start, end, speakers)."""
    out: list[tuple[float, float, tuple[str, ...]]] = []
    for a, b, active in _sweep(turns):
        who = tuple(sorted({t.speaker for t in active}))
        if len(who) < 2:
            continue
        if out and abs(out[-1][1] - a) <= 1e-6:
            pa, _, pw = out[-1]
            out[-1] = (pa, b, tuple(sorted(set(pw) | set(who))))
        else:
            out.append((a, b, who))
    return out


def union_length(intervals) -> float:
    total, end = 0.0, -math.inf
    for a, b in sorted(intervals):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


# ------------------------------------------------------------------ words <- speakers
def assign_speakers(words: list[Word], diar: Diarization, max_gap: float = 1.0, overlap_min: float = 0.5,
                    smooth_cover: float = 0.5) -> list[Word]:
    """Copies of `words` with meta speaker, speaker_cover (share of the word inside its speaker's
    exclusive turns) and, inside overlapping speech, overlap=True + overlap_with."""
    ex = diar.exclusive
    starts, ends = [t.start for t in ex], [t.end for t in ex]
    n = len(words)
    spk: list[str | None] = [None] * n
    cover = [0.0] * n
    for k, w in enumerate(words):
        if w.start is None or w.end is None or not ex:
            continue
        s, e = w.start, max(w.end, w.start)
        dur = max(e - s, 1e-3)
        lo = bisect.bisect_right(ends, s)       # first turn ending after s
        hi = bisect.bisect_left(starts, e)      # turns starting before e
        acc: dict[str, float] = {}
        for t in ex[lo:hi]:
            ov = min(e, t.end) - max(s, t.start)
            if ov > 0:
                acc[t.speaker] = acc.get(t.speaker, 0.0) + ov
        if acc:
            best = max(acc, key=acc.get)
            spk[k], cover[k] = best, min(acc[best] / dur, 1.0)
            continue
        near = []
        if lo > 0:
            near.append((s - ends[lo - 1], ex[lo - 1].speaker))
        if hi < len(ex):
            near.append((starts[hi] - e, ex[hi].speaker))
        if near:
            gap, who = min(near)
            if gap <= max_gap:
                spk[k] = who
    # words far from every turn: the previous word's speaker, else the next one's
    last = None
    for k in range(n):
        if spk[k] is None:
            spk[k] = last
        last = spk[k]
    nxt = None
    for k in range(n - 1, -1, -1):
        if spk[k] is None:
            spk[k] = nxt
        nxt = spk[k]
    # overlapping speech, from the regular turns (which may overlap)
    regions = overlap_regions(diar.turns)
    r_starts, r_ends = [r[0] for r in regions], [r[1] for r in regions]
    inside: list[set[str] | None] = [None] * n       # speakers talking during the word, if overlapped
    for k, w in enumerate(words):
        if not regions or w.start is None or w.end is None:
            continue
        s, e = w.start, max(w.end, w.start)
        ov, who = 0.0, set()
        if e - s <= 1e-6:                       # zero-length word: is its instant inside a region?
            r = bisect.bisect_right(r_starts, s) - 1
            if r >= 0 and r_ends[r] >= s:
                ov, who = 1.0, set(regions[r][2])
        else:
            for a, b, ws in regions[bisect.bisect_right(r_ends, s):bisect.bisect_left(r_starts, e)]:
                part = min(e, b) - max(s, a)
                if part > 0:
                    ov += part
                    who.update(ws)
        if ov > 0 and ov >= overlap_min * max(e - s, 1e-6):
            inside[k] = who
    _split_handovers(words, spk, cover, inside)
    # one-word islands with weak support join their neighbours
    for k in range(1, n - 1):
        if spk[k] != spk[k - 1] and spk[k - 1] == spk[k + 1] and cover[k] < smooth_cover:
            spk[k], cover[k] = spk[k - 1], 0.0
    out = []
    for k, w in enumerate(words):
        meta = dict(w.meta, speaker=spk[k], speaker_cover=round(cover[k], 3))
        if inside[k] is not None:
            meta["overlap"] = True
            meta["overlap_with"] = sorted(inside[k] - {spk[k]})
        out.append(w.copy(meta=meta))
    return out


def _split_handovers(words: list[Word], spk: list, cover: list, inside: list) -> None:
    """Speaker changes inside overlapping speech. When a run of overlapped words lies between words
    of speaker X and speaker Y and both talk during it, the exclusive turns cannot tell whose the
    words are (Whisper transcribes one stream). Split the run at the most likely handover instead:
    after a sentence end, else after a clause mark, else at the longest pause between words."""
    n = len(words)
    k = 0
    while k < n:
        if inside[k] is None:
            k += 1
            continue
        a = k
        while k < n and inside[k] is not None:
            k += 1
        b = k                                   # words[a:b] are overlapped
        if a == 0 or b == n:
            continue
        x, y = spk[a - 1], spk[b]
        if x is None or y is None or x == y or not all({x, y} <= inside[m] for m in range(a, b)):
            continue

        def score(cut: int) -> tuple[int, float]:
            """cut = number of run words that stay with x; the boundary is before words[a + cut]."""
            prev, nxt = words[a + cut - 1], words[a + cut]
            p = T.trailing_punct(prev.text)
            punct = 2 if p and p[-1] in ".?!" else 1 if p and p[-1] in ",;:" else 0
            pause = (nxt.start or 0.0) - (prev.end or 0.0)
            return punct, pause
        cut = max(range(b - a + 1), key=score)
        for m in range(a, b):
            who = x if m < a + cut else y
            if spk[m] != who:
                spk[m], cover[m] = who, 0.0


def speaker_turns(words: list[Word]) -> list[dict]:
    """Runs of consecutive words by the same speaker (the speaker turns of the transcript)."""
    turns: list[dict] = []
    for w in words:
        s = w.meta.get("speaker")
        if turns and turns[-1]["speaker"] == s:
            t = turns[-1]
            t["end"] = w.end if w.end is not None else t["end"]
            t["words"] += 1
            t["_text"].append(w.text)
            t["overlap"] = t["overlap"] or bool(w.meta.get("overlap"))
        else:
            turns.append({"speaker": s, "start": w.start, "end": w.end, "words": 1, "_text": [w.text],
                          "overlap": bool(w.meta.get("overlap"))})
    for t in turns:
        t["text"] = T.detokenize(t.pop("_text"))
    return turns


def _interruptions(turns: list[Turn], tol: float = 0.05) -> Counter:
    """Per speaker: turns that start while another speaker is still talking."""
    count: Counter = Counter()
    active: list[Turn] = []
    for t in sorted(turns, key=lambda t: (t.start, t.end)):
        active = [u for u in active if u.end > t.start + tol]
        if any(u.speaker != t.speaker and u.start < t.start - tol for u in active):
            count[t.speaker] += 1
        active.append(t)
    return count


def speaker_report(words: list[Word], cues: list[Cue], diar: Diarization, duration: float = 0.0) -> dict:
    """Per-speaker statistics, overall turn-taking figures, word-level turns and overlap regions."""
    wturns = speaker_turns(words)
    regions = overlap_regions(diar.turns)
    speech = union_length((t.start, t.end) for t in diar.turns)
    overlap = sum(b - a for a, b, _ in regions)
    interrupts = _interruptions(diar.turns)
    names = diar.speakers + sorted({w.meta.get("speaker") for w in words if w.meta.get("speaker")} - set(diar.speakers))
    rows = []
    for s in names:
        own = [(t.start, t.end) for t in diar.turns if t.speaker == s]
        talk = union_length(own)
        mine = [t for t in wturns if t["speaker"] == s]
        n_words = sum(t["words"] for t in mine)
        durs = [max((t["end"] or 0.0) - (t["start"] or 0.0), 0.0) for t in mine]
        rows.append({
            "speaker": s,
            "talk_time": round(talk, 3),
            "share": round(talk / speech, 4) if speech > 0 else 0.0,
            "words": n_words,
            "wpm": round(60.0 * n_words / talk, 1) if talk > 0 else None,
            "turns": len(mine),
            "mean_turn": round(sum(durs) / len(durs), 3) if durs else 0.0,
            "longest_turn": round(max(durs), 3) if durs else 0.0,
            "overlap_time": round(sum(b - a for a, b, who in regions if s in who), 3),
            "interruptions": int(interrupts.get(s, 0)),
            "cues": sum(1 for c in cues if s in c.speakers),
            "first": round(min(a for a, _ in own), 3) if own else (mine[0]["start"] if mine else None),
            "last": round(max(b for _, b in own), 3) if own else (mine[-1]["end"] if mine else None),
        })
    real = [t for t in wturns if t["speaker"]]
    durs = [max((t["end"] or 0.0) - (t["start"] or 0.0), 0.0) for t in real]
    totals = {
        "speakers": len(names),
        "audio_duration": round(duration, 3),
        "speech_time": round(speech, 3),
        "overlap_time": round(overlap, 3),
        "overlap_share": round(overlap / speech, 4) if speech > 0 else 0.0,
        "turns": len(real),
        "speaker_changes": max(len(real) - 1, 0),
        "mean_turn": round(sum(durs) / len(durs), 3) if durs else 0.0,
        "overlap_words": sum(1 for w in words if w.meta.get("overlap")),
        "overlap_cues": sum(1 for c in cues if any(w.meta.get("overlap") for w in c.words)),
        "dialogue_cues": sum(1 for c in cues if len({s for s in c.speakers if s}) > 1),
    }
    return {
        "info": diar.info,
        "totals": totals,
        "speakers": rows,
        "turns": [{"speaker": t["speaker"], "start": _r(t["start"]), "end": _r(t["end"]), "words": t["words"],
                   "overlap": t["overlap"], "text": t["text"]} for t in wturns],
        "overlaps": [{"start": round(a, 3), "end": round(b, 3), "speakers": list(who)} for a, b, who in regions],
    }


def _r(x):
    return None if x is None else round(float(x), 3)


# ------------------------------------------------------------------ labels
def line_prefixes(line_speakers: list[str | None], prev: str | None, mode: str = "every",
                  fmt: str = "{speaker}: ") -> list[str]:
    """Label prefix for each line of a cue; `prev` = speaker of the previous cue's last line.

    every   label the first line of every cue, and a line whose speaker differs from the line above
    change  label a line only when its speaker differs from the line before it (across cues)
    none    no names; a two-speaker cue gets the usual "- " dash on each line
    """
    if mode not in LABEL_MODES:
        raise ValueError(f"unknown speaker label mode {mode!r} (use {', '.join(LABEL_MODES)})")
    dialogue = any(a and b and a != b for a, b in zip(line_speakers, line_speakers[1:]))
    out = []
    for k, s in enumerate(line_speakers):
        if mode == "none":
            out.append("- " if dialogue else "")
            continue
        before = line_speakers[k - 1] if k > 0 else (None if mode == "every" else prev)
        out.append(fmt.format(speaker=s) if s is not None and s != before else "")
    return out


def parse_labels(lines: list[str], speakers: list[str], prev: str | None = None,
                 fmt: str = "{speaker}: ") -> tuple[list[str | None], list[str]]:
    """Speakers and label prefixes of hand-edited cue lines: a line starting with a known label
    belongs to that speaker, any other line continues the speaker before it."""
    known = sorted(((fmt.format(speaker=s), s) for s in speakers), key=lambda p: -len(p[0]))
    spk: list[str | None] = []
    labs: list[str] = []
    cur = prev
    for ln in lines:
        hit = next(((p, s) for p, s in known if ln.startswith(p)), None)
        if hit:
            cur = hit[1]
        labs.append(hit[0] if hit else "")
        spk.append(cur)
    return spk, labs


def to_rttm(diar: Diarization, uri: str = "audio") -> str:
    """The diarization in NIST RTTM (pyannote.metrics / dscore can score it)."""
    uri = re.sub(r"\s+", "_", uri.strip()) or "audio"
    return "".join(f"SPEAKER {uri} 1 {t.start:.3f} {t.dur:.3f} <NA> <NA> {t.speaker} <NA> <NA>\n"
                   for t in diar.turns)


# ------------------------------------------------------------------ our diarizer
def make_diarizer(cfg, device: str = "cpu"):
    """Our speaker-embedding model + clustering (src/subfix/speaker.py), set up from cfg.diarize."""
    from .lm import resolve_model_path
    from .speaker import OwnDiarizer, SpeakerEmbedder
    d = cfg.diarize
    path = resolve_model_path(d.model)
    if not path.exists():
        raise RuntimeError(f"speaker model {path} not found: train it with python experiments/train_speaker.py "
                           "and tune it with python experiments/eval_diarize.py --stage tune")
    return OwnDiarizer(SpeakerEmbedder(path, device), d.threshold, d.window, d.hop, d.vad_rel, d.min_turn)


def diarize(wav: str, cfg, models=None, audio: np.ndarray | None = None, use_cache: bool = True) -> Diarization:
    """Diarization of a 16 kHz WAV (cached by audio content + model file + diarization settings)."""
    from .lm import resolve_model_path
    from .paths import cache_dir
    from .preprocess import SR, load_audio
    from .utils.cache import file_hash, load_json, save_json, text_hash
    d = cfg.diarize
    mpath = resolve_model_path(d.model)
    key = text_hash(file_hash(wav), file_hash(str(mpath)) if mpath.exists() else d.model, d.num_speakers,
                    d.min_speakers, d.max_speakers, d.threshold, d.window, d.hop, d.vad_rel, d.min_turn)
    path = cache_dir("diarize") / f"{Path(wav).stem}-{key}.json"
    if use_cache and path.exists():
        return Diarization.from_json(load_json(path))
    if models is not None:
        diarizer = models.diarizer(cfg)
    else:
        from .utils.device import resolve_device
        diarizer = make_diarizer(cfg, resolve_device(cfg.device))
    if audio is None:
        audio = load_audio(wav)
    t0 = time.perf_counter()
    res = diarizer(audio, SR, d.num_speakers, d.min_speakers, d.max_speakers)
    res.info["seconds"] = round(time.perf_counter() - t0, 2)
    log.info("diarization: %d speakers, %d turns in %.1f s", len(res.speakers), len(res.turns), res.info["seconds"])
    if use_cache:
        save_json(path, res.to_json())
    return res

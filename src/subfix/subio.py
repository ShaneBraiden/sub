"""Data model + subtitle I/O (.srt / .vtt) — imp.md 8.1."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from . import text as T


@dataclass
class Word:
    text: str
    start: float | None = None   # seconds
    end: float | None = None
    prob: float = 1.0            # ASR confidence (1.0 for subtitle words)
    source: str = "sub"          # "sub" | "asr" | "fused" | stage name
    meta: dict = field(default_factory=dict)

    @property
    def norm(self) -> str:
        return T.norm(self.text)

    def copy(self, **changes) -> "Word":
        w = Word(self.text, self.start, self.end, self.prob, self.source, dict(self.meta))
        for k, v in changes.items():
            setattr(w, k, v)
        return w


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str                    # lines separated by "\n" (speaker labels included, as displayed)
    words: list[Word] = field(default_factory=list)
    speakers: list[str | None] = field(default_factory=list)  # diarization: speaker of each line
    labels: list[str] = field(default_factory=list)           # label prefix shown on each line ("" = none)

    @property
    def lines(self) -> list[str]:
        return [ln for ln in self.text.split("\n")]

    @property
    def flat_text(self) -> str:
        return " ".join(self.text.split())


def spoken_text(cue: Cue) -> str:
    """Cue text without its speaker-label prefixes (what reading speed is counted on)."""
    if not cue.labels:
        return cue.text
    out = []
    for k, line in enumerate(cue.text.split("\n")):
        lab = cue.labels[k] if k < len(cue.labels) else ""
        out.append(line[len(lab):] if lab and line.startswith(lab) else line)
    return "\n".join(out)


# ---------------------------------------------------------------- cleaning / tokenizing
_TAG = re.compile(r"</?[a-zA-Z][^>]*>|\{\\[^}]*\}")
_SPEAKER_DASH = re.compile(r"^\s*[-–—]\s+")
_SPEAKER_LABEL = re.compile(r"^\s*[\[(]?SPEAKER_\d+[\])]?:\s*")   # our own diarization labels
_EVENT_SPAN = re.compile(r"(\[[^\]]*\]|\([^)]*\))")
_SPACE = "\ue000"   # private-use placeholder (str.split() treats NBSP as whitespace)


def clean_text(raw: str) -> str:
    """Strip HTML/ASS styling tags, leading speaker dashes and SPEAKER_nn labels; keep line breaks."""
    lines = []
    for line in raw.replace("\r", "").split("\n"):
        line = _TAG.sub("", line)
        line = _SPEAKER_DASH.sub("", line)
        line = _SPEAKER_LABEL.sub("", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def tokenize(text: str) -> list[str]:
    """Whitespace tokens; bracketed sound events stay atomic; punctuation-only tokens are
    merged into the previous token."""
    protected = _EVENT_SPAN.sub(lambda m: m.group(0).replace(" ", _SPACE), text)
    out: list[str] = []
    for tok in protected.split():
        tok = tok.replace(_SPACE, " ")
        if not T.norm(tok) and not T.is_event(tok):
            if out:
                out[-1] = out[-1] + tok
            continue
        out.append(tok)
    return out


def cue_words(cue: Cue, cue_pos: int | None = None) -> list[Word]:
    """Tokens of a cue with approximate times (proportional to character count)."""
    toks: list[tuple[str, int]] = []
    for ln, line in enumerate(cue.text.split("\n")):
        toks.extend((t, ln) for t in tokenize(line))
    if not toks:
        return []
    weights = [max(len(T.norm(t)), 1) + 1 for t, _ in toks]
    total = float(sum(weights))
    dur = max(cue.end - cue.start, 1e-3)
    words, acc = [], 0.0
    for k, ((tok, ln), wgt) in enumerate(zip(toks, weights)):
        s = cue.start + dur * acc / total
        acc += wgt
        e = cue.start + dur * acc / total
        meta = {"cue": cue.index if cue_pos is None else cue_pos, "pos": k, "line": ln, "approx_time": True}
        if T.is_event(tok):
            meta["event"] = True
        words.append(Word(tok, s, e, 1.0, "sub", meta))
    return words


def cues_to_words(cues: list[Cue]) -> list[Word]:
    words: list[Word] = []
    for pos, c in enumerate(cues):
        ws = cue_words(c, pos)
        c.words = ws
        words.extend(ws)
    for i, w in enumerate(words):
        w.meta["orig_index"] = i
    return words


def is_caseless(words: list[Word]) -> bool:
    """True if the subtitle has no usable casing (all lower- or all upper-case)."""
    letters = "".join(w.text for w in words if not w.meta.get("event"))
    letters = [c for c in letters if c.isalpha()]
    if len(letters) < 20:
        return False
    upper = sum(c.isupper() for c in letters) / len(letters)
    return upper < 0.005 or upper > 0.95


# ---------------------------------------------------------------- SRT
def _fmt_srt(t: float) -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt(t: float) -> str:
    return _fmt_srt(t).replace(",", ".")


_TIME = r"(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}"
_TIMING_LINE = re.compile(rf"^\s*({_TIME})\s*-->\s*({_TIME})")


def _parse_time(s: str) -> float:
    s = s.strip().replace(",", ".")
    parts = s.split(":")
    if len(parts) == 2:
        parts = ["0"] + parts
    h, m, sec = parts
    return int(h) * 3600 + int(m) * 60 + float(sec)


def parse_srt_text(content: str) -> list[Cue]:
    content = content.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    try:
        import srt as srtlib
        subs = list(srtlib.parse(content, ignore_errors=True))
        cues = [Cue(s.index, s.start.total_seconds(), s.end.total_seconds(), clean_text(s.content))
                for s in subs]
    except Exception:
        cues = _parse_blocks(content)
    return _finish(cues)


def _parse_blocks(content: str) -> list[Cue]:
    """Generic SRT/VTT block parser (fallback + VTT)."""
    cues: list[Cue] = []
    blocks = re.split(r"\n\s*\n", content)
    for block in blocks:
        lines = [ln for ln in block.split("\n")]
        for k, line in enumerate(lines):
            m = _TIMING_LINE.match(line)
            if m:
                start, end = _parse_time(m.group(1)), _parse_time(m.group(3))
                body = clean_text("\n".join(lines[k + 1:]))
                cues.append(Cue(len(cues) + 1, start, end, body))
                break
    return cues


def _finish(cues: list[Cue]) -> list[Cue]:
    cues = [c for c in cues if c.text.strip()]
    cues.sort(key=lambda c: (c.start, c.end))
    for i, c in enumerate(cues):
        c.index = i + 1
    return cues


def read_srt(path: str | Path) -> list[Cue]:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return parse_srt_text(f.read())


def parse_vtt_text(content: str) -> list[Cue]:
    content = content.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    # drop header, NOTE/STYLE/REGION blocks
    blocks = re.split(r"\n\s*\n", content)
    kept = [b for b in blocks
            if not b.lstrip().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")) or "-->" in b]
    return _finish(_parse_blocks("\n\n".join(kept)))


def read_vtt(path: str | Path) -> list[Cue]:
    # own parser: webvtt-py's start_in_seconds truncates to whole seconds
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return parse_vtt_text(f.read())


def read_subs(path: str | Path) -> list[Cue]:
    p = str(path).lower()
    if p.endswith(".vtt"):
        return read_vtt(path)
    return read_srt(path)


def compose_srt(cues: list[Cue]) -> str:
    out = []
    for i, c in enumerate(cues):
        out.append(f"{i + 1}\n{_fmt_srt(c.start)} --> {_fmt_srt(c.end)}\n{c.text.strip()}\n")
    return "\n".join(out)


def _vtt_text(c: Cue) -> str:
    """Cue text with WebVTT voice spans (<v SPEAKER_00>...</v>) when the speakers are known."""
    text = c.text.strip()
    known = [s for s in c.speakers if s]
    if not known:
        return text
    lines = text.split("\n")
    if len(set(c.speakers)) == 1 or len(c.speakers) != len(lines):
        return f"<v {known[0]}>{text}</v>"
    return "\n".join(f"<v {s}>{ln}</v>" if s else ln for s, ln in zip(c.speakers, lines))


def compose_vtt(cues: list[Cue]) -> str:
    out = ["WEBVTT\n"]
    for i, c in enumerate(cues):
        out.append(f"{i + 1}\n{_fmt_vtt(c.start)} --> {_fmt_vtt(c.end)}\n{_vtt_text(c)}\n")
    return "\n".join(out)


def write_srt(cues: list[Cue], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(compose_srt(cues))


def write_vtt(cues: list[Cue], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(compose_vtt(cues))


def write_subs(cues: list[Cue], path: str | Path) -> None:
    if str(path).lower().endswith(".vtt"):
        write_vtt(cues, path)
    else:
        write_srt(cues, path)


def srt_timedelta(t: float) -> timedelta:
    return timedelta(seconds=t)

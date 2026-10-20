"""Re-segmentation of timed words into subtitle cues — imp.md 3.5 / 8.7.

DP over word boundaries: each candidate cue words[i:j] must fit in <= max_lines lines of
<= max_chars (a single over-long word may exceed it) and last <= max_dur seconds of speech.
Its cost combines a squared reading-speed (CPS) excess, a linguistic break penalty at the cue
boundary (POS from our perceptron tagger: never split article-noun / adjective-noun / auxiliary-verb, prefer breaks
after punctuation and before conjunctions/prepositions), a bonus for breaking at pauses, the
best two-line split (bottom-heavy preferred) and a per-cue cost against over-splitting.
CPS convention: characters excluding spaces (cfg.segment.cps_count_spaces=false), configurable.
"""
from __future__ import annotations

import math
from collections import Counter

from . import text as T
from .diarize import line_prefixes
from .subio import Cue, Word
from .timing import finalize_cue_times

_DET = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its",
        "our", "their", "some", "any", "no", "every", "each"}
_ADP = {"of", "to", "in", "on", "at", "for", "with", "by", "from", "about", "into", "over", "under",
        "after", "before", "between", "through", "during", "without", "within", "upon", "across"}
_CCONJ = {"and", "or", "but", "nor", "yet", "so"}
_SCONJ = {"because", "although", "though", "if", "when", "while", "since", "unless", "whether",
          "until", "whereas", "where"}
_AUX = {"is", "are", "was", "were", "be", "been", "being", "am", "will", "would", "can", "could",
        "should", "shall", "may", "might", "must", "have", "has", "had", "do", "does", "did"}
_PRON = {"i", "you", "he", "she", "we", "they", "it", "me", "him", "us", "them"}


def heuristic_pos(word: str) -> str:
    n = T.norm(word)
    if n in _DET:
        return "DET"
    if n in _ADP:
        return "ADP"
    if n in _CCONJ:
        return "CCONJ"
    if n in _SCONJ:
        return "SCONJ"
    if n in _AUX:
        return "AUX"
    if n in _PRON:
        return "PRON"
    if n.isdigit():
        return "NUM"
    return "X"


def pos_tags(words: list[Word], tagger=None) -> list[str]:
    """One UD POS tag per word (our perceptron tagger, or a lexical fallback)."""
    if not words:
        return []
    if tagger is None:
        return [heuristic_pos(w.text) for w in words]
    return tagger.tag_tokens([w.text for w in words])


def break_penalty(prev_text: str, next_text: str | None, pl: str = "X", pr: str = "X") -> float:
    """Cost of breaking (line or cue) between two words; 0 = ideal."""
    if next_text is None:
        return 0.0
    p = T.trailing_punct(prev_text)
    if p and p[-1] in ".?!":
        return 0.0
    if p and p[-1] in ",;:":
        return 0.3
    if pl == "DET" or (pl == "PRON" and T.norm(prev_text) in {"my", "your", "his", "her", "its", "our", "their"}):
        return 3.0
    if pl == "PROPN" and pr == "PROPN":
        return 3.0
    if pl == "ADJ" and pr in ("NOUN", "PROPN"):
        return 2.5
    if pl in ("AUX", "PART") and pr in ("VERB", "AUX"):
        return 2.0
    if pl == "NUM" and pr == "NOUN":
        return 2.0
    if pl in ("ADP", "CCONJ", "SCONJ"):
        return 2.0             # do not end a line with a preposition / conjunction
    if pr in ("CCONJ", "SCONJ"):
        return 0.4
    if pr == "ADP":
        return 0.6
    return 1.0


def char_count(s: str, count_spaces: bool = False) -> int:
    s = s.replace("\n", " ")
    return len(s) if count_spaces else len(s.replace(" ", ""))


def best_two_line_split(tokens: list[str], max_chars: int, pos: list[str] | None = None,
                        max_lines: int = 2, prefix: int = 0) -> tuple[list[str], float] | None:
    """Best layout of the tokens in <= 2 lines of <= max_chars (single long words allowed);
    `prefix` characters (a speaker label) are reserved at the start of the first line."""
    n = len(tokens)
    if n == 0:
        return [""], 0.0
    text = " ".join(tokens)
    if len(text) + prefix <= max_chars or n == 1:
        return [text], 0.0
    if max_lines < 2:
        return None
    pos = pos or ["X"] * n
    lens = [len(t) for t in tokens]
    total = sum(lens) + n - 1
    best: tuple[list[str], float] | None = None
    acc = 0
    for k in range(1, n):
        acc += lens[k - 1]
        l1 = acc + (k - 1)
        l2 = total - l1 - 1
        l1 += prefix
        if (l1 > max_chars and k > 1) or (l2 > max_chars and n - k > 1):
            continue
        cost = break_penalty(tokens[k - 1], tokens[k], pos[k - 1], pos[k])
        cost += 0.6 * max(0, l1 - l2) / max_chars      # bottom-heavy (pyramid) preference
        cost += 0.2 * abs(l1 - l2) / max_chars
        if best is None or cost < best[1]:
            best = ([" ".join(tokens[:k]), " ".join(tokens[k:])], cost)
    return best


class _Seg:
    def __init__(self, words: list[Word], cfg, pos: list[str] | None):
        self.w = words
        self.c = cfg.segment
        self.n = len(words)
        self.txt = [w.text for w in words]
        self.pos = pos or [heuristic_pos(t) for t in self.txt]
        self.st = [w.start for w in words]
        self.en = [w.end for w in words]
        # speakers (words carry meta["speaker"] after diarization): a cue never spans a speaker change,
        # except a two-line dialogue cue with one speaker per line
        d = getattr(cfg, "diarize", None)
        self.spk = [w.meta.get("speaker") for w in words]
        self.has_spk = any(s is not None for s in self.spk)
        self.label_mode = d.label if d is not None else "none"
        self.label_fmt = d.label_format if d is not None else "{speaker}: "
        self.dialogue_cost = d.dialogue_cost if d is not None else 0.5
        self.max_changes = int(self.has_spk and d is not None and d.dialogue and self.c.max_lines >= 2)
        self.cp = [0] * self.n              # cp[k] = speaker changes at word positions 1..k
        for k in range(1, self.n):
            self.cp[k] = self.cp[k - 1] + (self.spk[k] != self.spk[k - 1])

    def changes(self, i: int, j: int) -> int:
        """Speaker changes inside words[i:j]."""
        return self.cp[j - 1] - self.cp[i] if self.has_spk else 0

    def speaker_break(self, j: int) -> bool:
        return self.has_spk and 0 < j < self.n and self.spk[j] != self.spk[j - 1]

    def layout(self, i: int, j: int, dialogue: bool = True):
        """(lines, line speakers, line label prefixes, layout cost) of the cue words[i:j], or None."""
        c = self.c
        if not self.has_spk:
            split = best_two_line_split(self.txt[i:j], c.max_chars, self.pos[i:j], c.max_lines)
            return None if split is None else (split[0], [], [], split[1])
        prev = self.spk[i - 1] if i > 0 else None
        nchg = self.changes(i, j)
        if nchg == 0:
            pre = line_prefixes([self.spk[i]], prev, self.label_mode, self.label_fmt)[0]
            split = best_two_line_split(self.txt[i:j], c.max_chars, self.pos[i:j], c.max_lines, prefix=len(pre))
            if split is None:
                return None
            lines, cost = split
            return lines, [self.spk[i]] * len(lines), [pre] + [""] * (len(lines) - 1), cost
        if nchg > self.max_changes or not dialogue:
            return None
        k = next(m for m in range(i + 1, j) if self.spk[m] != self.spk[m - 1])
        speakers = [self.spk[i], self.spk[k]]
        pres = line_prefixes(speakers, prev, self.label_mode, self.label_fmt)
        lines = [" ".join(self.txt[i:k]), " ".join(self.txt[k:j])]
        for pre, line, count in zip(pres, lines, (k - i, j - k)):
            if len(pre) + len(line) > c.max_chars and count > 1:
                return None
        return lines, speakers, pres, self.dialogue_cost

    def display_dur(self, i: int, j: int) -> float:
        c = self.c
        nxt = self.st[j] if j < self.n else math.inf
        end = max(min(self.en[j - 1] + c.linger, nxt - c.min_gap), self.en[j - 1])
        return max(end - self.st[i], min(c.min_dur, max(nxt - c.min_gap - self.st[i], 1e-3)))

    def cost(self, i: int, j: int) -> float:
        c = self.c
        speech = self.en[j - 1] - self.st[i]
        if speech > c.max_dur and j - i > 1:
            return math.inf
        lay = self.layout(i, j)
        if lay is None:
            return math.inf
        line_cost = lay[3]
        chars = sum(char_count(t, c.cps_count_spaces) for t in self.txt[i:j])
        if c.cps_count_spaces:
            chars += j - i - 1
        cps = chars / max(self.display_dur(i, j), 1e-3)
        cost = 10.0 * max(0.0, cps - c.max_cps) ** 2
        nxt_txt = self.txt[j] if j < self.n else None
        nxt_pos = self.pos[j] if j < self.n else "X"
        if not self.speaker_break(j):       # a speaker change is the ideal place for a cue break
            cost += 1.5 * break_penalty(self.txt[j - 1], nxt_txt, self.pos[j - 1], nxt_pos)
        if speech < c.min_dur:
            cost += 0.5
        cost += line_cost
        pause = (self.st[j] - self.en[j - 1]) if j < self.n else 1.0
        cost -= c.pause_bonus * min(max(pause, 0.0), 1.0)
        return cost + c.cue_cost

    def too_long(self, i: int, j: int) -> bool:
        c = self.c
        chars = sum(len(t) + 1 for t in self.txt[i:j]) - 1
        return (chars > c.max_lines * c.max_chars + 1 and j - i > 1) or \
            (self.en[j - 1] - self.st[i] > c.max_dur and j - i > 1) or \
            self.changes(i, j) > self.max_changes


def segment_dp(words: list[Word], cfg, pos: list[str] | None = None) -> list[tuple[int, int]]:
    s = _Seg(words, cfg, pos)
    n = s.n
    best = [0.0] + [math.inf] * n
    back = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j - 1, max(-1, j - 1 - cfg.segment.max_words), -1):
            if s.too_long(i, j):
                break
            if best[i] == math.inf:
                continue
            c = best[i] + s.cost(i, j)
            if c < best[j]:
                best[j], back[j] = c, i
        if best[j] == math.inf:   # nothing fits (e.g. one very long word): single-word cue
            best[j], back[j] = best[j - 1] + 100.0, j - 1
    cuts, j = [], n
    while j > 0:
        cuts.append((back[j], j))
        j = back[j]
    return cuts[::-1]


def segment_greedy(words: list[Word], cfg, pos: list[str] | None = None) -> list[tuple[int, int]]:
    s = _Seg(words, cfg, pos)
    c = cfg.segment
    cuts, i, n = [], 0, s.n

    def fits(a: int, b: int) -> bool:
        if s.too_long(a, b) or s.changes(a, b):     # greedy cues never span a speaker change
            return False
        if s.layout(a, b, dialogue=False) is None:
            return False
        chars = sum(char_count(t, c.cps_count_spaces) for t in s.txt[a:b])
        return b - a == 1 or chars / max(s.display_dur(a, b), 1e-3) <= c.lenient_cps or b - a <= 2

    while i < n:
        j = i + 1
        while j < n and fits(i, j + 1):
            j += 1
        if j < n:   # prefer cutting right after punctuation in the last third
            for k in range(j, i + max(1, 2 * (j - i) // 3), -1):
                if T.trailing_punct(s.txt[k - 1]):
                    j = k
                    break
        cuts.append((i, j))
        i = j
    return cuts


def build_cues(words: list[Word], cuts: list[tuple[int, int]], cfg, pos: list[str] | None = None) -> list[Cue]:
    c = cfg.segment
    s = _Seg(words, cfg, pos)
    cues = []
    for k, (i, j) in enumerate(cuts):
        lay = s.layout(i, j)
        if lay is not None:
            lines, speakers, prefixes, _ = lay
        else:                               # forced cut (nothing fitted): one line per speaker run
            runs: list[list[int]] = []
            for m in range(i, j):
                if runs and not s.speaker_break(m):
                    runs[-1].append(m)
                else:
                    runs.append([m])
            lines = [" ".join(s.txt[m] for m in run) for run in runs]
            speakers = [s.spk[run[0]] for run in runs] if s.has_spk else []
            prefixes = (line_prefixes(speakers, s.spk[i - 1] if i > 0 else None, s.label_mode, s.label_fmt)
                        if s.has_spk else [])
        text = "\n".join(p + ln for p, ln in zip(prefixes, lines)) if prefixes else "\n".join(lines)
        cues.append(Cue(k + 1, words[i].start, words[j - 1].end, text, list(words[i:j]), speakers, prefixes))
    return finalize_cue_times(cues, c.min_dur, c.max_dur, c.min_gap, c.lead_in, c.linger)


def keep_original(words: list[Word], orig_cues: list[Cue], cfg, keep_times: bool) -> list[Cue]:
    """Group words back into their original cues (and original line breaks)."""
    groups: dict[int, list[Word]] = {}
    last = 0
    for w in words:
        cue = w.meta.get("cue")
        cue = last if cue is None else cue
        last = cue
        groups.setdefault(cue, []).append(w)
    d = getattr(cfg, "diarize", None)
    has_spk = d is not None and any(w.meta.get("speaker") is not None for w in words)
    prev = None
    cues = []
    for pos_, ws in sorted(groups.items()):
        oc = orig_cues[pos_] if 0 <= pos_ < len(orig_cues) else None
        lines: dict[int, list[Word]] = {}
        last_line = 0
        for w in ws:
            ln = w.meta.get("line", last_line)
            last_line = ln
            lines.setdefault(ln, []).append(w)
        line_words = [v for _, v in sorted(lines.items())]
        texts = [" ".join(w.text for w in v) for v in line_words]
        speakers: list[str | None] = []
        prefixes: list[str] = []
        if has_spk:                          # each original line: the speaker of most of its words
            speakers = [Counter(w.meta.get("speaker") for w in v).most_common(1)[0][0] for v in line_words]
            prefixes = line_prefixes(speakers, prev, d.label, d.label_format)
            prev = next((sp for sp in reversed(speakers) if sp is not None), prev)
            texts = [p + t for p, t in zip(prefixes, texts)]
        text = "\n".join(texts)
        if keep_times and oc is not None:
            cues.append(Cue(len(cues) + 1, oc.start, oc.end, text, ws, speakers, prefixes))
        else:
            cues.append(Cue(len(cues) + 1, ws[0].start, ws[-1].end, text, ws, speakers, prefixes))
    if not keep_times:
        c = cfg.segment
        cues = finalize_cue_times(cues, c.min_dur, c.max_dur, c.min_gap, c.lead_in, c.linger)
    return cues


def segment(words: list[Word], cfg, pos: list[str] | None = None, method: str | None = None) -> list[Cue]:
    method = method or cfg.segment.method
    words = [w for w in words if w.text]
    if not words:
        return []
    cuts = segment_greedy(words, cfg, pos) if method == "greedy" else segment_dp(words, cfg, pos)
    return build_cues(words, cuts, cfg, pos)

"""Chapter markers from topic segmentation — the second output (README "Chapter markers").

The corrected, forced-aligned words are cut into sentences (our restored punctuation; a run of
more than 50 words without a full stop is cut at its longest pause), each sentence becomes a bag
of content-word stems, and the sentence sequence is split into topically coherent chapters:

* ``dp`` (default) - Utiyama & Isahara (2001): the segmentation that maximises the probability
  of the text under one Laplace-smoothed unigram model per segment plus a description-length
  prior per segment, found exactly by dynamic programming over sentence boundaries. No training;
  the per-chapter penalty is a tunable multiplier of the prior.
* ``tiling`` - TextTiling (Hearst 1997): cosine similarity of word blocks either side of every
  sentence boundary, depth scores, boundaries where the depth exceeds mean - cutoff * std.

Content words are those our POS tagger tags NOUN/PROPN/VERB/ADJ (configurable), minus fillers,
closed-class words and words our KN language model finds too common to signal a topic (unigram
log-prob above ``generic_logp``); a light Porter step-1 stemmer merges inflections. Every chapter
lasts at least ``min_duration`` seconds.

Times come from the word timings: a chapter starts at the cue its first word opens, or just
before that word. Titles are the chapter's most distinctive noun phrases (frequency x specificity
under the LM unigram distribution x share of the phrase's occurrences inside this chapter); the
description is the chapter sentence closest to the chapter's centroid.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import text as T
from .subio import Cue, Word

MAX_SENTENCE_WORDS = 50
LEAD = 0.25                      # a chapter starts this long before its first word (inside the pause)
_TITLE_POS = {"NOUN", "PROPN", "ADJ"}
_HEAD_POS = {"NOUN", "PROPN"}
_SPOKEN_STOP = {"yeah", "okay", "ok", "gonna", "wanna", "gotta", "kinda", "sorta", "alright"}


@lru_cache(maxsize=1)
def generic_words() -> frozenset[str]:
    """Vague nouns/adjectives (data/lexicons/generic_words.txt) + spoken fillers + closed-class words."""
    return frozenset(T.norm(w) for w in T._read_lexicon("generic_words.txt")) | _SPOKEN_STOP \
        | T.function_words() | T.fillers()[0]


def _bare(token: str) -> str:
    """Normalized word without a possessive 's ("Mother's" -> "mother")."""
    n = T.norm(token)
    return n[:-2] if n.endswith("'s") else n


# ---------------------------------------------------------------- stemming (Porter 1980, step 1)
_VOWELS = set("aeiou")


def _cons(w: str, i: int) -> bool:
    if w[i] in _VOWELS:
        return False
    if w[i] == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _measure(stem: str) -> int:
    m, prev_v = 0, False
    for i in range(len(stem)):
        v = not _cons(stem, i)
        if prev_v and not v:
            m += 1
        prev_v = v
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _cons(stem, i) for i in range(len(stem)))


def _cvc(w: str) -> bool:
    return len(w) >= 3 and _cons(w, -3 % len(w)) and not _cons(w, len(w) - 2) and _cons(w, len(w) - 1) \
        and w[-1] not in "wxy"


@lru_cache(maxsize=100_000)
def light_stem(w: str) -> str:
    """Porter's step 1 (plurals, -ed, -ing, final y): merges inflections without the
    over-stemming of the derivational steps ("organization" stays apart from "organ")."""
    if len(w) <= 3:
        return w
    if w.endswith("sses") or w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
        w = w[:-1]
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    else:
        for suf in ("ed", "ing"):
            if w.endswith(suf) and _has_vowel(w[:-len(suf)]):
                w = w[:-len(suf)]
                if w.endswith(("at", "bl", "iz")):
                    w += "e"
                elif len(w) > 1 and w[-1] == w[-2] and _cons(w, len(w) - 1) and w[-1] not in "lsz":
                    w = w[:-1]
                elif _measure(w) == 1 and _cvc(w):
                    w += "e"
                break
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"
    return w


# ---------------------------------------------------------------- background word frequencies
class Background:
    """Unigram log-probabilities of our KN LM (its lowest-order distribution), read from the
    model file without loading the higher orders (the npz is read lazily: ~30 ms)."""

    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=True)
        vocab = z["vocab"]
        probs = np.full(len(vocab), float(z["g1"][0]) / (len(vocab) - 1))
        self.unk = float(np.log(probs[0]))
        probs[z["k1"].astype(np.int64)] += z["p1"]
        self.table = dict(zip(vocab.tolist(), np.log(probs).tolist()))

    def logp(self, word: str) -> float:
        return self.table.get(word, self.unk)


# ---------------------------------------------------------------- data model
@dataclass
class Chapter:
    index: int
    start: float
    end: float
    title: str
    keywords: list[str] = field(default_factory=list)
    summary: str = ""
    words: tuple[int, int] = (0, 0)        # word index range [i, j)
    score: float = 0.0                     # strength of the topic shift at its start (0 for the first)

    def to_json(self) -> dict:
        return {"index": self.index, "start": round(self.start, 3), "end": round(self.end, 3),
                "title": self.title, "keywords": self.keywords, "summary": self.summary,
                "first_word": self.words[0], "last_word": self.words[1], "score": round(self.score, 4)}


def _times(words: list[Word]) -> tuple[list[float], list[float]]:
    st, en, last = [], [], 0.0
    for w in words:
        s = w.start if w.start is not None else last
        e = w.end if w.end is not None else s
        st.append(s)
        en.append(max(e, s))
        last = en[-1]
    return st, en


def sentence_spans(words: list[Word], st: list[float], en: list[float],
                   max_words: int = MAX_SENTENCE_WORDS) -> list[tuple[int, int]]:
    spans, i = [], 0
    for k, w in enumerate(words):
        if T.ends_sentence(w.text) or k == len(words) - 1:
            spans.append((i, k + 1))
            i = k + 1

    def split(a: int, b: int) -> list[tuple[int, int]]:
        if b - a <= max_words:
            return [(a, b)]
        cut = max(range(a + 5, b - 4), key=lambda k: st[k] - en[k - 1])
        return split(a, cut) + split(cut, b)
    return [s for a, b in spans for s in split(a, b)]


class Document:
    """Words prepared once (sentences, content terms) for any number of segmentation settings."""

    def __init__(self, words: list[Word], pos: list[str] | None, chapters_cfg, background: Background | None = None,
                 duration: float = 0.0, sents: list[tuple[int, int]] | None = None):
        c = chapters_cfg
        self.words = words
        self.pos = pos or ["NOUN"] * len(words)
        self.bg = background
        self.st, self.en = _times(words)
        self.duration = max(duration, self.en[-1] if words else 0.0)
        self.sents = sents if sents is not None else sentence_spans(words, self.st, self.en)
        content = set(c.content_pos)
        self.ids: dict[str, int] = {}
        self.terms: list[list[int]] = []
        for a, b in self.sents:
            row = []
            for k in range(a, b):
                t = self.term(k, content, c.stem, c.generic_logp)
                if t is not None:
                    row.append(self.ids.setdefault(t, len(self.ids)))
            self.terms.append(row)
        S = len(self.sents)
        # chapter boundary times: T[0] = 0, T[S] = media end, else the first word of sentence k
        self.T = [0.0] + [self.st[self.sents[k][0]] for k in range(1, S)] + [self.duration]
        self._cost: np.ndarray | None = None
        self._depth: dict[tuple[int, int], np.ndarray] = {}

    def term(self, k: int, content: set[str], stem: bool, generic_logp: float) -> str | None:
        if self.pos[k] not in content:
            return None
        n = _bare(self.words[k].text)
        if len(n) < 3 or not n.isalpha() or n in generic_words():
            return None
        if self.bg is not None and generic_logp < 0 and self.bg.logp(n) > generic_logp:
            return None
        return light_stem(n) if stem else n

    # ------------------------------------------------------------ Utiyama & Isahara DP
    def cost(self) -> np.ndarray:
        """cost[a, b] = -log P(words of sentences a..b-1 | Laplace unigram of that segment)."""
        if self._cost is None:
            S, k = len(self.terms), max(len(self.ids), 1)
            n_max = sum(len(t) for t in self.terms) + 2
            F = np.arange(n_max) * np.log(np.arange(n_max) + 1.0)       # F[c] = c log(c+1)
            cost = np.full((S + 1, S + 1), np.inf)
            counts = [0] * k
            for a in range(S):
                touched, acc, n, row = [], 0.0, 0, []
                for b in range(a + 1, S + 1):
                    for t in self.terms[b - 1]:
                        c = counts[t]
                        if c == 0:
                            touched.append(t)
                        acc += F[c + 1] - F[c]
                        counts[t] = c + 1
                    n += len(self.terms[b - 1])
                    row.append(n * math.log(n + k) - acc)
                cost[a, a + 1:] = row
                for t in touched:
                    counts[t] = 0
            self._cost = cost
        return self._cost

    def segment_dp(self, min_duration: float, penalty: float, num: int = 0,
                   bonus: np.ndarray | None = None) -> list[int]:
        """Chapter starts [0, b1, ..., S]. `bonus[k]` (nats, optional) is subtracted when a chapter starts at
        sentence k: evidence for a topic change other than the words, e.g. a long silence before it."""
        S = len(self.sents)
        if S < 2:
            return [0, S]
        cost = self.cost()
        Tm = np.asarray(self.T)
        valid = (Tm[None, :] - Tm[:, None]) >= min_duration
        valid[0, S] = True                                   # one chapter is always allowed
        C = np.where(valid, cost, np.inf)
        bon = np.zeros(S + 1)
        if bonus is not None:
            bon[1:S] = np.asarray(bonus, dtype=float)[1:S]
        if num > 0:                                          # exactly `num` chapters (fewer if they cannot fit)
            for K in range(min(num, S), 0, -1):
                D = np.full((K + 1, S + 1), np.inf)
                back = np.zeros((K + 1, S + 1), dtype=np.int64)
                D[0, 0] = 0.0
                for m in range(1, K + 1):
                    v = (D[m - 1] - bon)[:, None] + C
                    back[m] = np.argmin(v, axis=0)
                    D[m] = v[back[m], np.arange(S + 1)]
                if np.isfinite(D[K, S]):
                    out, b = [S], S
                    for m in range(K, 0, -1):
                        b = int(back[m, b])
                        out.append(b)
                    return out[::-1]
            return [0, S]
        pen = penalty * math.log(max(sum(len(t) for t in self.terms), 2))
        best = np.full(S + 1, np.inf)
        back = np.zeros(S + 1, dtype=np.int64)
        best[0] = 0.0
        for b in range(1, S + 1):
            v = best[:b] - bon[:b] + C[:b, b] + pen
            a = int(np.argmin(v))
            best[b], back[b] = v[a], a
        out, b = [S], S
        while b > 0:
            b = int(back[b])
            out.append(b)
        return out[::-1]

    def dp_scores(self, bounds: list[int]) -> list[float]:
        """Nats per content word saved by each inner boundary (vs merging its two neighbours)."""
        cost, out = self.cost(), [0.0]
        for a, b, c in zip(bounds, bounds[1:], bounds[2:]):
            n = max(sum(len(t) for t in self.terms[a:c]), 1)
            out.append(float(cost[a, c] - cost[a, b] - cost[b, c]) / n)
        return out

    # ------------------------------------------------------------ TextTiling
    def similarity(self, window_words: int) -> np.ndarray:
        """Cosine similarity of the term counts in the blocks before and after every sentence gap
        g = 1..S-1 (index g-1). A block grows sentence by sentence away from the gap until it holds
        `window_words` words (or reaches the end); counts come from prefix sums."""
        S, V = len(self.sents), max(len(self.ids), 1)
        W = np.concatenate([[0], np.cumsum([b - a for a, b in self.sents])])     # words before sentence k
        g = np.arange(1, S)
        left = np.maximum(np.searchsorted(W, W[g] - window_words, side="right") - 1, 0)
        right = np.minimum(np.searchsorted(W, W[g] + window_words, side="left"), S)
        C = np.zeros((S + 1, V), dtype=np.int32)                                  # C[k] = counts in sentences < k
        rows = np.repeat(np.arange(1, S + 1), [len(t) for t in self.terms])
        np.add.at(C, (rows, np.fromiter((t for ts in self.terms for t in ts), dtype=np.int64, count=len(rows))), 1)
        np.cumsum(C, axis=0, out=C)
        sims = np.zeros(S - 1)
        for a in range(0, S - 1, 256):                                           # bounded memory on long media
            sl = slice(a, a + 256)
            lb = (C[g[sl]] - C[left[sl]]).astype(np.float64)
            rb = (C[right[sl]] - C[g[sl]]).astype(np.float64)
            norm = np.sqrt((lb * lb).sum(1) * (rb * rb).sum(1))
            sims[sl] = np.where(norm > 0, (lb * rb).sum(1) / np.maximum(norm, 1e-12), 0.0)
        return sims

    def depth(self, window_words: int, smoothing: int) -> np.ndarray:
        """Depth score of every sentence gap (Hearst): how far the smoothed similarity climbs on each
        side before it falls again. Cached per setting."""
        key = (window_words, smoothing)
        if key not in self._depth:
            s = self.similarity(window_words)
            if smoothing > 0 and len(s):
                ker = np.ones(2 * smoothing + 1)
                s = np.convolve(np.pad(s, smoothing, mode="edge"), ker, "valid") / len(ker)
            left_peak, right_peak = s.copy(), s.copy()
            for k in range(1, len(s)):                  # climb left while the curve does not fall
                if s[k - 1] >= s[k]:
                    left_peak[k] = left_peak[k - 1]
            for k in range(len(s) - 2, -1, -1):
                if s[k + 1] >= s[k]:
                    right_peak[k] = right_peak[k + 1]
            self._depth[key] = (left_peak - s) + (right_peak - s)
        return self._depth[key]

    def segment_tiling(self, min_duration: float, window_words: int, smoothing: int, cutoff: float,
                       num: int = 0) -> tuple[list[int], np.ndarray]:
        S = len(self.sents)
        if S < 2:
            return [0, S], np.zeros(0)
        d = self.depth(window_words, smoothing)
        thr = d.mean() - cutoff * d.std()
        chosen = [0, S]
        peak = np.ones(len(d), dtype=bool)          # candidates: local maxima of the depth curve
        peak[1:] &= d[1:] >= d[:-1]
        peak[:-1] &= d[:-1] >= d[1:]
        for g in np.argsort(-d, kind="stable"):
            if not peak[g] or d[g] <= 0:
                continue
            b = int(g) + 1
            if num > 0 and len(chosen) - 1 >= num:
                break
            if num <= 0 and d[g] <= max(thr, 1e-9):
                break
            k = bisect_left(chosen, b)
            if self.T[b] - self.T[chosen[k - 1]] >= min_duration and self.T[chosen[k]] - self.T[b] >= min_duration:
                chosen.insert(k, b)
        return chosen, d

    def boundaries(self, c) -> tuple[list[int], list[float]]:
        """Sentence indices [0, b1, ..., S] of the chapters, and the score of each chapter start."""
        if c.method == "tiling":
            b, d = self.segment_tiling(c.min_duration, c.window_words, c.smoothing, c.cutoff, c.num_chapters)
            return b, [0.0] + [float(d[x - 1]) for x in b[1:-1]]
        if c.method != "dp":
            raise ValueError(f"unknown chapters.method {c.method!r} (use dp or tiling)")
        b = self.segment_dp(c.min_duration, c.penalty, c.num_chapters)
        return b, self.dp_scores(b)

    # ------------------------------------------------------------ titles and descriptions
    def _surface(self, k: int) -> str:
        core = T.core(self.words[k].text)
        if core.lower().endswith("'s"):
            core = core[:-2]
        if self.pos[k] == "PROPN" or T.case_pattern(core) == "upper":
            return core
        return core.lower()

    def phrases(self, i: int, j: int) -> list[tuple[tuple[str, ...], str]]:
        """Noun phrases (ADJ|NOUN|PROPN)* ending in NOUN|PROPN, 1-3 words, inside words[i:j]."""
        out, run = [], []

        def flush():
            for a in range(len(run)):
                for b in range(a + 1, min(a + 3, len(run)) + 1):
                    ks = run[a:b]
                    if self.pos[ks[-1]] in _HEAD_POS:
                        key = tuple(light_stem(_bare(self.words[k].text)) for k in ks)
                        out.append((key, " ".join(self._surface(k) for k in ks)))
        for k in range(i, j):
            n = _bare(self.words[k].text)
            ok = (self.pos[k] in _TITLE_POS and len(n) >= 2 and n.isalnum()
                  and n not in generic_words())
            if ok:
                run.append(k)
            if not ok or T.trailing_punct(self.words[k].text):
                flush()
                run = []
        flush()
        return out

    def spec(self, word: str) -> float:
        """How specific a word is: 0 at unigram log-prob -7 (common) .. 1 at -16 (rare)."""
        if self.bg is None:
            return 1.0
        return min(max((-self.bg.logp(word) - 7.0) / 9.0, 0.05), 1.0)

    def describe(self, bounds: list[int], n_title: int, n_keywords: int = 8) -> list[tuple[str, list[str], str]]:
        per = [self.phrases(self.sents[a][0], self.sents[b - 1][1]) for a, b in zip(bounds, bounds[1:])]
        doc_tf = Counter(key for ph in per for key, _ in ph)
        out = []
        for ci, (ph, (a, b)) in enumerate(zip(per, zip(bounds, bounds[1:]))):
            tf = Counter(key for key, _ in ph)
            forms: dict[tuple, Counter] = {}
            for key, surf in ph:
                forms.setdefault(key, Counter())[surf] += 1
            min_tf = 2 if any(v >= 2 for v in tf.values()) else 1
            scored = []
            for key, n in tf.items():
                if n < min_tf:
                    continue
                surf = forms[key].most_common(1)[0][0]
                sp = sum(self.spec(T.norm(w)) for w in surf.split()) / len(key)
                scored.append((n * sp * (n / doc_tf[key]) * (1 + 0.3 * (len(key) - 1)), surf, key))
            scored.sort(key=lambda x: -x[0])
            picked, used = [], set()
            for _, surf, key in scored:
                if used & set(key):
                    continue
                picked.append(surf)
                used |= set(key)
                if len(picked) >= n_keywords:
                    break
            head = picked[:n_title]
            if not head:
                title = f"Chapter {ci + 1}"
            elif len(head) == 1:
                title = head[0]
            else:
                title = ", ".join(head[:-1]) + " and " + head[-1]
            title = title[:1].upper() + title[1:]
            out.append((title, picked, self.central_sentence(a, b)))
        return out

    def central_sentence(self, a: int, b: int) -> str:
        """The sentence closest to the chapter centroid (cosine), favouring complete sentences
        near the start of the chapter, where a topic is usually introduced."""
        cen = Counter(t for k in range(a, b) for t in self.terms[k])
        best, best_k = -1.0, a
        for k in range(a, b):
            i, j = self.sents[k]
            if not 6 <= j - i <= 35:
                continue
            v = Counter(self.terms[k])
            dot = sum(x * cen[t] for t, x in v.items())
            nv = math.sqrt(sum(x * x for x in v.values()))
            if not nv:
                continue
            score = dot / nv * (1.0 - 0.3 * (k - a) / max(b - a, 1))
            if not T.core(self.words[i].text)[:1].isupper() or not T.ends_sentence(self.words[j - 1].text):
                score *= 0.7
            if score > best:
                best, best_k = score, k
        i, j = self.sents[best_k]
        text = T.detokenize([w.text for w in self.words[i:j] if not T.is_filler(w.text)])
        return text if len(text) <= 220 else text[:217].rsplit(" ", 1)[0] + "…"


# ---------------------------------------------------------------- entry point
def make_chapters(words: list[Word], cfg, pos: list[str] | None = None, background: Background | None = None,
                  cues: list[Cue] | None = None, duration: float = 0.0) -> list[Chapter]:
    c = cfg.chapters
    keep = [k for k, w in enumerate(words) if w.text]
    words = [words[k] for k in keep]
    if pos is not None:
        pos = [pos[k] for k in keep]
    if not words:
        return []
    doc = Document(words, pos, c, background, duration)
    bounds, scores = doc.boundaries(c)
    info = doc.describe(bounds, c.title_terms)
    cue_start = {id(cue.words[0]): cue.start for cue in (cues or []) if cue.words}
    starts = []
    for b in bounds[:-1]:
        if b == 0:
            starts.append(0.0)
            continue
        i = doc.sents[b][0]
        cs = cue_start.get(id(words[i]))
        prev_end = doc.en[i - 1]
        starts.append(cs if cs is not None and cs >= prev_end - 1e-6 else max(prev_end, doc.st[i] - LEAD))
    ends = starts[1:] + [doc.duration]
    return [Chapter(k + 1, s, e, title, kw, summ, (doc.sents[a][0], doc.sents[b - 1][1]), sc)
            for k, (s, e, (title, kw, summ), a, b, sc) in
            enumerate(zip(starts, ends, info, bounds, bounds[1:], scores))]


# ---------------------------------------------------------------- output formats
def _clock(t: float) -> str:
    s = int(t)                   # rounded down: a jump never lands after the chapter has started
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def to_vtt(chapters: list[Chapter]) -> str:
    from .subio import compose_vtt
    return compose_vtt([Cue(ch.index, ch.start, ch.end, ch.title) for ch in chapters])


def to_ffmetadata(chapters: list[Chapter]) -> str:
    def esc(s: str) -> str:
        return "".join("\\" + ch if ch in "=;#\\\n" else ch for ch in s)
    out = [";FFMETADATA1"]
    for ch in chapters:
        out += ["", "[CHAPTER]", "TIMEBASE=1/1000", f"START={int(round(ch.start * 1000))}",
                f"END={int(round(ch.end * 1000))}", f"title={esc(ch.title)}"]
    return "\n".join(out) + "\n"


def to_youtube(chapters: list[Chapter]) -> str:
    """Timestamp list for a video description (YouTube wants >= 3 chapters of >= 10 s, first at 0:00)."""
    return "".join(f"{_clock(ch.start)} {ch.title}\n" for ch in chapters)


def write_chapter_files(chapters: list[Chapter], out_dir: str | Path, stem: str) -> dict[str, str]:
    from .utils.cache import save_json
    out = Path(out_dir)
    paths = {"chapters_vtt": out / f"{stem}.chapters.vtt", "chapters_txt": out / f"{stem}.chapters.txt",
             "ffmetadata": out / f"{stem}.ffmetadata", "chapters_json": out / f"{stem}.chapters.json"}
    paths["chapters_vtt"].write_text(to_vtt(chapters), encoding="utf-8", newline="\n")
    paths["chapters_txt"].write_text(to_youtube(chapters), encoding="utf-8", newline="\n")
    paths["ffmetadata"].write_text(to_ffmetadata(chapters), encoding="utf-8", newline="\n")
    save_json(paths["chapters_json"], [ch.to_json() for ch in chapters])
    return {k: str(v) for k, v in paths.items()}

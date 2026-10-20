"""Averaged-perceptron POS tagger (Collins 2002; feature set after Honnibal's "A good
part-of-speech tagger in about 200 lines"), trained from scratch on the Brown corpus by
experiments/train_pos.py. Tags are Universal Dependencies POS tags (DET, NOUN, PROPN, ADJ,
VERB, AUX, ADP, CCONJ, SCONJ, PRON, PART, NUM, ADV, INTJ, PUNCT, SYM, X), the set the
segmenter's break penalties are written for.

Brown tags are mapped to UD with a few lexical/contextual rules: forms of "be" and modals are
AUX; "have"/"do" are AUX when a verb follows, VERB otherwise; CS = SCONJ, RP = ADP, TO = PART.
"""
from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

from . import text as T

START = ("-START-", "-START2-")
END = ("-END-", "-END2-")

_BROWN = {
    "AT": "DET", "AP": "ADJ", "ABN": "DET", "ABX": "DET", "ABL": "ADV", "DT": "DET", "DTI": "DET",
    "DTS": "DET", "DTX": "DET", "CC": "CCONJ", "CS": "SCONJ", "CD": "NUM", "OD": "ADJ", "EX": "PRON",
    "IN": "ADP", "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ", "JJT": "ADJ", "NN": "NOUN", "NNS": "NOUN",
    "NP": "PROPN", "NPS": "PROPN", "NR": "NOUN", "NRS": "NOUN", "PN": "PRON", "PP": "PRON",
    "PPL": "PRON", "PPLS": "PRON", "PPO": "PRON", "PPS": "PRON", "PPSS": "PRON", "QL": "ADV",
    "QLP": "ADV", "RB": "ADV", "RBR": "ADV", "RBT": "ADV", "RN": "ADV", "RP": "ADP", "TO": "PART",
    "UH": "INTJ", "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB", "VBZ": "VERB",
    "WDT": "PRON", "WP": "PRON", "WPO": "PRON", "WPS": "PRON", "WQL": "ADV", "WRB": "ADV",
    "MD": "AUX", "NIL": "X", "*": "PART",
}
_PUNCT_TAGS = {".", ",", ":", "(", ")", "''", "``", "--", "'", "'"}


def _base(tag: str) -> str:
    """'NN$-TL' -> 'NN', 'PPS+BEZ' -> 'PPS', 'DOD*' -> 'DOD', 'FW-NN' -> 'FW'."""
    if tag in _PUNCT_TAGS:
        return tag
    t = tag.split("+")[0]
    if t.startswith("FW-"):
        return "FW"
    t = t.split("-")[0] or t
    if t != "*":
        t = t.rstrip("*")
    return t.rstrip("$") or t


def brown_to_ud(tagged: list[tuple[str, str]]) -> list[tuple[str, str]]:
    bases = [_base(t) for _, t in tagged]
    out = []
    for k, (word, _) in enumerate(tagged):
        b = bases[k]
        if b in _PUNCT_TAGS:
            ud = "PUNCT"
        elif b == "FW":
            ud = "X"
        elif b.startswith("BE"):
            ud = "AUX"
        elif b.startswith("HV") or b.startswith("DO"):
            follow = [x for x in bases[k + 1:k + 4] if x not in ("*", "RB", "PPS", "PPSS", "QL")]
            nxt = follow[0] if follow else ""
            ud = "AUX" if nxt.startswith(("VB", "BEN", "HVN", "BE", "HV", "DO")) else "VERB"
        else:
            ud = _BROWN.get(b, "X")
        out.append((word, ud))
    return out


def split_token(token: str) -> tuple[str, list[str], list[str]]:
    """Subtitle token -> (core, leading punctuation tokens, trailing punctuation tokens), matching
    the Brown tokenization (punctuation is a separate token, contractions stay whole)."""
    pre, core, suf = T.split_affixes(token)
    return core, [c for c in pre if not c.isspace()], [c for c in suf if not c.isspace()]


class PerceptronTagger:
    def __init__(self, weights: dict[str, dict[str, float]] | None = None,
                 tagdict: dict[str, str] | None = None, classes: list[str] | None = None):
        self.weights = weights or {}
        self.tagdict = tagdict or {}
        self.classes = classes or []

    # ------------------------------------------------------------------ features
    @staticmethod
    def _norm(word: str) -> str:
        if "-" in word and word[0] != "-":
            return "!HYPHEN"
        if word.isdigit() and len(word) == 4:
            return "!YEAR"
        if word[:1].isdigit():
            return "!DIGITS"
        return word.lower()

    @staticmethod
    def _shape(word: str) -> str:
        if word.isupper() and len(word) > 1:
            return "UPPER"
        if word[:1].isupper():
            return "TITLE"
        if any(c.isdigit() for c in word):
            return "DIGIT"
        if not any(c.isalnum() for c in word):
            return "PUNCT"
        return "lower"

    def _features(self, i: int, word: str, context: list[str], raw: list[str], prev: str, prev2: str) -> list[str]:
        w = context[i]
        r = raw[i - 2]
        return [
            "bias",
            "i suffix " + w[-3:],
            "i suffix2 " + w[-2:],
            "i pref1 " + w[:1],
            "i shape " + self._shape(r),
            "i shape+first " + self._shape(r) + str(i == 2),
            "i-1 tag " + prev,
            "i-2 tag " + prev2,
            "i tag+i-2 tag " + prev + " " + prev2,
            "i word " + w,
            "i-1 tag+i word " + prev + " " + w,
            "i-1 word " + context[i - 1],
            "i-1 suffix " + context[i - 1][-3:],
            "i-2 word " + context[i - 2],
            "i+1 word " + context[i + 1],
            "i+1 suffix " + context[i + 1][-3:],
            "i+2 word " + context[i + 2],
        ]

    def _predict(self, feats: list[str]) -> str:
        scores: dict[str, float] = defaultdict(float)
        for f in feats:
            wts = self.weights.get(f)
            if wts:
                for c, v in wts.items():
                    scores[c] += v
        return max(self.classes, key=lambda c: (scores[c], c))

    # ------------------------------------------------------------------ tagging
    def tag(self, words: list[str]) -> list[str]:
        """One UD tag per (already tokenized) word."""
        if not words:
            return []
        context = list(START) + [self._norm(w) for w in words] + list(END)
        raw = list(words)
        prev, prev2 = START
        out = []
        for i, word in enumerate(words):
            tag = self.tagdict.get(word)
            if not tag:
                tag = self._predict(self._features(i + 2, word, context, raw, prev, prev2))
            out.append(tag)
            prev2, prev = prev, tag
        return out

    def tag_tokens(self, tokens: list[str]) -> list[str]:
        """One tag per subtitle token ('Hello,' -> tag of 'Hello'); punctuation is split off and
        tagged as context, exactly as in training."""
        flat, owner = [], []
        for k, tok in enumerate(tokens):
            core, pre, suf = split_token(tok)
            flat.extend(pre)
            owner.extend([-1] * len(pre))
            flat.append(core or tok)
            owner.append(k)
            flat.extend(suf)
            owner.extend([-1] * len(suf))
        tags = self.tag(flat)
        return [t for t, o in zip(tags, owner) if o >= 0]

    # ------------------------------------------------------------------ training
    def train(self, sentences: list[list[tuple[str, str]]], n_iter: int = 5, seed: int = 0, log=print) -> None:
        self._make_tagdict(sentences)
        self.classes = sorted({t for s in sentences for _, t in s})
        totals: dict[tuple[str, str], float] = defaultdict(float)
        stamps: dict[tuple[str, str], int] = defaultdict(int)
        n_updates = 0
        rng = random.Random(seed)
        sents = list(sentences)
        for it in range(n_iter):
            correct = total = 0
            for sent in sents:
                words = [w for w, _ in sent]
                context = list(START) + [self._norm(w) for w in words] + list(END)
                prev, prev2 = START
                for i, (word, gold) in enumerate(sent):
                    guess = self.tagdict.get(word)
                    if not guess:
                        feats = self._features(i + 2, word, context, words, prev, prev2)
                        guess = self._predict(feats)
                        n_updates += 1
                        if guess != gold:
                            for f in feats:
                                wts = self.weights.setdefault(f, {})
                                for c, delta in ((gold, 1.0), (guess, -1.0)):
                                    key = (f, c)
                                    totals[key] += (n_updates - stamps[key]) * wts.get(c, 0.0)
                                    stamps[key] = n_updates
                                    wts[c] = wts.get(c, 0.0) + delta
                    prev2, prev = prev, guess
                    correct += guess == gold
                    total += 1
            rng.shuffle(sents)
            log(f"[pos] iter {it + 1}/{n_iter}: train accuracy {correct / total:.4f}")
        for f, wts in self.weights.items():           # average
            for c, w in list(wts.items()):
                key = (f, c)
                tot = totals[key] + (n_updates - stamps[key]) * w
                avg = round(tot / n_updates, 3)
                if avg:
                    wts[c] = avg
                else:
                    del wts[c]
        self.weights = {f: w for f, w in self.weights.items() if w}

    def _make_tagdict(self, sentences, freq_thresh: int = 20, ambiguity_thresh: float = 0.97) -> None:
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for sent in sentences:
            for word, tag in sent:
                counts[word][tag] += 1
        for word, tags in counts.items():
            tag, mode = max(tags.items(), key=lambda kv: kv[1])
            n = sum(tags.values())
            if n >= freq_thresh and mode / n >= ambiguity_thresh:
                self.tagdict[word] = tag

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"classes": self.classes, "tagdict": self.tagdict, "weights": self.weights}, f)

    @classmethod
    def load(cls, path: str | Path) -> "PerceptronTagger":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["weights"], d["tagdict"], d["classes"])

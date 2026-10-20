"""Sentence scorer used to arbitrate between subtitle and ASR words: our own modified
Kneser-Ney 4-gram LM (subfix.ngram, trained by experiments/train_lm.py on LibriSpeech-PC train
text)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from . import text as T
from .paths import ROOT

log = logging.getLogger(__name__)

DEFAULT_LM = "models/lm/kn4.npz"


def lm_words(token: str) -> list[str]:
    """LM vocabulary form of one subtitle token: lowercase, punctuation stripped, numbers spelled out."""
    n = T.norm(token)
    if not n:
        return []
    return T.number_words(n) or [n]


def sentence_pieces(tokens: list[str], starts_sentence: bool = False) -> list[tuple[list[str], bool, bool]]:
    """Split a token window at sentence-final punctuation into (words, starts, ends) pieces.
    A window taken from the middle of a text does not start a sentence unless told so."""
    pieces, cur, starts = [], [], starts_sentence
    for tok in tokens:
        cur.extend(lm_words(tok))
        if T.ends_sentence(tok):
            pieces.append((cur, starts, True))
            cur, starts = [], True
    if cur:
        pieces.append((cur, starts, False))
    return pieces


def resolve_model_path(name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else ROOT / p


class LMScorer:
    """Total log-probability (nats) of short texts, memoized."""

    def __init__(self, model_path: str = DEFAULT_LM, device: str = "auto"):
        from .ngram import KneserNeyLM
        path = resolve_model_path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: train it with python experiments/train_lm.py")
        self.model = KneserNeyLM.load(path)
        self.cache: dict[str, float] = {}

    def score(self, texts: Iterable[str]) -> list[float]:
        texts = list(texts)
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            pieces, owner = [], []
            for n, t in enumerate(todo):
                for p in sentence_pieces(t.split()):
                    pieces.append(p)
                    owner.append(n)
            scores = self.model.score_sequences(pieces) if pieces else []
            totals = [0.0] * len(todo)
            for n, s in zip(owner, scores):
                totals[n] += s
            for t, s in zip(todo, totals):
                self.cache[t] = float(s)
        return [self.cache[t] for t in texts]

    def logp(self, word: str, ctx: list[str]) -> float:
        """log P(ctx + word) - log P(ctx): conditional log-prob of `word` after `ctx`."""
        a, b = self.score([" ".join(ctx + [word]), " ".join(ctx)])
        return a - b

    def close(self) -> None:
        self.model = None
        self.cache.clear()


class StubLM:
    """Deterministic LM for tests: penalises unknown bigrams, rewards known phrases."""

    def __init__(self, good_phrases: Iterable[str] = (), bad_phrases: Iterable[str] = ()):
        self.good = [p.lower() for p in good_phrases]
        self.bad = [p.lower() for p in bad_phrases]

    def score(self, texts: Iterable[str]) -> list[float]:
        out = []
        for t in texts:
            low = " " + " ".join(T.norm(w) for w in t.split()) + " "
            s = -1.0 * len(t.split())
            s += sum(3.0 for p in self.good if f" {p} " in low)
            s -= sum(3.0 for p in self.bad if f" {p} " in low)
            out.append(s)
        return out

    def logp(self, word: str, ctx: list[str]) -> float:
        a, b = self.score([" ".join(ctx + [word]), " ".join(ctx)])
        return a - b

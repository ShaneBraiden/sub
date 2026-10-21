"""ROUGE (Lin 2004) for the summary evaluation (experiments/eval_summaries.py).

Follows Google's rouge-score package (the usual Python implementation), so the numbers are comparable
with published ones: text is lowercased, everything but [a-z0-9] becomes a space, tokens longer than
3 characters are Porter-stemmed (NLTK's stemmer, its default mode), and

* ROUGE-N: clipped n-gram overlap; precision = overlap / candidate n-grams, recall = overlap / reference
  n-grams, F1 = their harmonic mean.
* ROUGE-L: longest common subsequence of the two token sequences.
* ROUGE-Lsum: summary-level LCS over sentences: for every reference sentence the union of its LCS
  with each candidate sentence, each token counted at most as often as it occurs on both sides.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Score:
    precision: float
    recall: float
    fmeasure: float


def fmeasure(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem.porter import PorterStemmer      # imported on first use (nltk takes seconds to import)
    return PorterStemmer()


@lru_cache(maxsize=200_000)
def _stem(token: str) -> str:
    return _stemmer().stem(token)


def tokenize(text: str, stem: bool = True) -> list[str]:
    toks = _NON_ALNUM.sub(" ", text.lower()).split()
    return [_stem(t) if stem and len(t) > 3 else t for t in toks]


def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def rouge_n(cand: list[str], ref: list[str], n: int) -> Score:
    c, r = _ngrams(cand, n), _ngrams(ref, n)
    hits = sum(min(k, c[g]) for g, k in r.items())
    p = hits / max(sum(c.values()), 1)
    rec = hits / max(sum(r.values()), 1)
    return Score(p, rec, fmeasure(p, rec))


def _lcs_table(ref: list[str], cand: list[str]) -> list[list[int]]:
    t = [[0] * (len(cand) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        ri, row, prev = ref[i - 1], t[i], t[i - 1]
        for j in range(1, len(cand) + 1):
            row[j] = prev[j - 1] + 1 if ri == cand[j - 1] else max(prev[j], row[j - 1])
    return t


def rouge_l(cand: list[str], ref: list[str]) -> Score:
    if not cand or not ref:
        return Score(0.0, 0.0, 0.0)
    lcs = _lcs_table(ref, cand)[-1][-1]
    p, r = lcs / len(cand), lcs / len(ref)
    return Score(p, r, fmeasure(p, r))


def _lcs_indices(ref: list[str], cand: list[str]) -> list[int]:
    """Positions in `ref` of one LCS with `cand` (rouge-score's backtracking order)."""
    t = _lcs_table(ref, cand)
    i, j, out = len(ref), len(cand), []
    while i > 0 and j > 0:
        if ref[i - 1] == cand[j - 1]:
            out.append(i - 1)
            i, j = i - 1, j - 1
        elif t[i][j - 1] > t[i - 1][j]:
            j -= 1
        else:
            i -= 1
    return out[::-1]


def rouge_lsum(cand_sents: list[list[str]], ref_sents: list[list[str]]) -> Score:
    m = sum(len(s) for s in ref_sents)
    n = sum(len(s) for s in cand_sents)
    if not m or not n:
        return Score(0.0, 0.0, 0.0)
    left_r = Counter(t for s in ref_sents for t in s)
    left_c = Counter(t for s in cand_sents for t in s)
    hits = 0
    for ref in ref_sents:
        union = sorted(set().union(*(_lcs_indices(ref, c) for c in cand_sents)))
        for k in union:
            tok = ref[k]
            if left_c[tok] > 0 and left_r[tok] > 0:
                hits += 1
                left_c[tok] -= 1
                left_r[tok] -= 1
    p, r = hits / n, hits / m
    return Score(p, r, fmeasure(p, r))


TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")


def rouge(candidate: str | list[str], reference: str | list[str], stem: bool = True,
          types: tuple[str, ...] = TYPES) -> dict[str, Score]:
    """ROUGE-1, -2, -L and -Lsum (or the `types` asked for) of a candidate against one reference. Either side
    may be a string or a list of sentences (sentences only matter for ROUGE-Lsum; a string is split at
    newlines)."""
    cs = candidate.split("\n") if isinstance(candidate, str) else list(candidate)
    rs = reference.split("\n") if isinstance(reference, str) else list(reference)
    c_sents = [t for t in (tokenize(s, stem) for s in cs) if t]
    r_sents = [t for t in (tokenize(s, stem) for s in rs) if t]
    c = [t for s in c_sents for t in s]
    r = [t for s in r_sents for t in s]
    fns = {"rouge1": lambda: rouge_n(c, r, 1), "rouge2": lambda: rouge_n(c, r, 2), "rougeL": lambda: rouge_l(c, r),
           "rougeLsum": lambda: rouge_lsum(c_sents, r_sents)}
    return {t: fns[t]() for t in types}


def truncate_words(text: str | list[str], budget: int) -> list[str]:
    """The first `budget` words (whitespace tokens) of a summary, keeping its sentence breaks: the
    length limit of the DUC evaluations, applied to every system alike."""
    sents = text.split("\n") if isinstance(text, str) else list(text)
    out, left = [], budget
    for s in sents:
        w = s.split()
        if not w or left <= 0:
            continue
        out.append(" ".join(w[:left]))
        left -= len(w[:left])
    return out

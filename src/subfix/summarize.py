"""Summaries at three granularities — the third output (README "Summaries").

The corrected, timed words are cut into sentences (our restored punctuation; a change of speaker or a
pause longer than 1.5 s also ends one, and one longer than `max_sentence_words` is cut at its longest
pause). Fillers ("um", "uh") and stutters ("our our") are dropped from the text. Three granularities,
each with a word budget:

* headline - one sentence saying what the recording is about (`headline_words`)
* overview - a paragraph for the whole recording (`overview_words`)
* chapters - a few sentences for every chapter marker (`chapter_words` each; needs 2+ chapters)

Extractive (always; our own, nothing to download): sentences picked from the recording, shown in time
order with their timestamps.

* ``embed`` (default) - sentence embeddings: LSA (TF-IDF over content-word stems + truncated SVD,
  Deerwester et al. 1990) trained on the Brown corpus and the AMI training meetings
  (experiments/train_summarizer.py). Relevance is a sentence's cosine to the centroid of the text;
  maximal marginal relevance (Carbonell & Goldstein 1998) trades it against the similarity to what is
  already picked.
* ``textrank`` - TextRank (Mihalcea & Tarau 2004): PageRank over a graph of sentences whose edges are
  their shared content words, normalised by sentence length.
* ``tfidf`` - ``embed`` without the SVD (cosine in the full TF-IDF term space), an ablation.

Abstractive (opt-in, ``summary.abstractive``): a local LLM through Ollama (src/subfix/llm.py)
summarises every chapter from its transcript, then writes the overview from the chapter summaries
and the headline from the overview, so a recording of any length fits the model's context.

The settings were chosen on the AMI development meetings (experiments/eval_summaries.py).
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from . import text as T
from .chapters import _bare, generic_words, light_stem
from .subio import Word

log = logging.getLogger(__name__)

PAUSE = 1.5                     # s of silence that ends a sentence
METHODS = ("embed", "textrank", "tfidf")
# frequent spoken words that say nothing about the topic (on top of chapters.generic_words())
_SPOKEN = {"just", "really", "actually", "like", "think", "know", "mean", "gonna", "wanna", "get", "got",
           "getting", "say", "said", "yes", "yep", "nope", "sure", "maybe", "probably", "basically",
           "obviously", "hello", "thanks", "thank", "mhm", "uh-huh", "okay", "alright"}


# ---------------------------------------------------------------- sentences
@dataclass
class Sentence:
    index: int
    start: float
    end: float
    text: str                    # display text: fillers and stutters removed
    speaker: str | None
    n_words: int
    terms: list[str] = field(default_factory=list)      # content-word stems

    def to_json(self) -> dict:
        d = {"index": self.index, "start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}
        if self.speaker:
            d["speaker"] = self.speaker
        return d


@lru_cache(maxsize=1)
def stop_words() -> frozenset[str]:
    return generic_words() | frozenset(_SPOKEN)


def content_terms(tokens: list[str]) -> list[str]:
    """Stems of the content words of a token list (closed-class, filler and vague words dropped)."""
    out = []
    for tok in tokens:
        n = _bare(tok)
        if len(n) >= 3 and n.isalpha() and n not in stop_words():
            out.append(light_stem(n))
    return out


def clean_tokens(tokens: list[str]) -> list[str]:
    """Drop fillers, sound events and immediate repetitions ("our our project" -> "our project")."""
    out: list[str] = []
    for tok in tokens:
        if T.is_filler(tok) or T.is_event(tok) or not T.norm(tok):
            if out and T.trailing_punct(tok) and not T.trailing_punct(out[-1]):
                out[-1] += T.trailing_punct(tok)          # "it, um." keeps the full stop
            continue
        if out and T.norm(tok) == T.norm(out[-1]) and not T.trailing_punct(out[-1]):
            out[-1] = tok
            continue
        out.append(tok)
    if out:
        out[0] = T.capitalize_first(out[0])
    return out


def _speaker(w: Word) -> str | None:
    return w.meta.get("speaker") if w.meta else None


def make_sentences(words: list[Word], max_words: int = 40, pause: float = PAUSE) -> list[Sentence]:
    words = [w for w in words if w.text]
    if not words:
        return []
    st, en, last = [], [], 0.0
    for w in words:
        s = w.start if w.start is not None else last
        st.append(s)
        en.append(max(w.end if w.end is not None else s, s))
        last = en[-1]
    spans, a = [], 0
    for k, w in enumerate(words):
        if (k == len(words) - 1 or T.ends_sentence(w.text) or _speaker(words[k + 1]) != _speaker(w)
                or st[k + 1] - en[k] > pause):
            spans.append((a, k + 1))
            a = k + 1

    def split(a: int, b: int) -> list[tuple[int, int]]:
        if b - a <= max(max_words, 10):
            return [(a, b)]
        cut = max(range(a + 5, b - 4), key=lambda k: st[k] - en[k - 1])
        return split(a, cut) + split(cut, b)

    out: list[Sentence] = []
    for a0, b0 in spans:
        for a, b in split(a0, b0):
            toks = clean_tokens([w.text for w in words[a:b]])
            if not toks:
                continue
            out.append(Sentence(len(out), st[a], en[b - 1], T.detokenize(toks), _speaker(words[a]), len(toks),
                                content_terms(toks)))
    return out


# ---------------------------------------------------------------- sentence vectors
class LSA:
    """TF-IDF over content-word stems (sublinear tf) and its projection on the top right singular vectors
    of a background corpus: `embed` gives unit-length sentence embeddings."""

    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=False)
        self._set(z["terms"].tolist(), z["idf"], z["components"])

    @classmethod
    def from_arrays(cls, terms: list[str], idf, components) -> "LSA":
        lsa = cls.__new__(cls)
        lsa._set(list(terms), idf, components)
        return lsa

    def _set(self, terms: list[str], idf, components) -> None:
        self.terms = terms
        self.index = {t: i for i, t in enumerate(terms)}
        self.idf = np.asarray(idf, dtype=np.float32)
        self.components = np.asarray(components, dtype=np.float32)      # dims x V
        self.dims = self.components.shape[0]

    def tfidf(self, docs: list[list[str]]):
        from scipy.sparse import csr_matrix
        rows, cols, vals = [], [], []
        for r, terms in enumerate(docs):
            counts: dict[int, int] = {}
            for t in terms:
                j = self.index.get(t)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(r)
                cols.append(j)
                vals.append((1.0 + math.log(c)) * self.idf[j])
        X = csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.terms)), dtype=np.float32)
        return _normalize_rows(X)

    def embed(self, docs: list[list[str]]) -> np.ndarray:
        E = np.asarray(self.tfidf(docs) @ self.components.T)
        n = np.linalg.norm(E, axis=1, keepdims=True)
        return E / np.maximum(n, 1e-9)


def _normalize_rows(X):
    from scipy.sparse import diags
    n = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    return diags(1.0 / np.maximum(n, 1e-9)) @ X


def local_tfidf(docs: list[list[str]]):
    """TF-IDF with the IDF of the text itself (sentences as documents): the fallback without an LSA model."""
    from scipy.sparse import csr_matrix
    vocab: dict[str, int] = {}
    df: dict[int, int] = {}
    rows, cols, vals = [], [], []
    for r, terms in enumerate(docs):
        counts: dict[int, int] = {}
        for t in terms:
            j = vocab.setdefault(t, len(vocab))
            counts[j] = counts.get(j, 0) + 1
        for j, c in counts.items():
            df[j] = df.get(j, 0) + 1
            rows.append(r)
            cols.append(j)
            vals.append(1.0 + math.log(c))
    n = max(len(docs), 1)
    idf = np.ones(max(len(vocab), 1), dtype=np.float32)
    for j, d in df.items():
        idf[j] = math.log((1 + n) / (1 + d)) + 1.0
    vals = [v * idf[j] for v, j in zip(vals, cols)]
    return _normalize_rows(csr_matrix((vals, (rows, cols)), shape=(len(docs), max(len(vocab), 1)), dtype=np.float32))


# ---------------------------------------------------------------- scoring
@dataclass
class Scorer:
    """Relevance of every sentence of a set, and the vectors used for redundancy (cosine)."""
    method: str
    lsa: LSA | None = None
    damping: float = 0.85

    def vectors(self, sents: list[Sentence]):
        docs = [s.terms for s in sents]
        if self.method == "embed" and self.lsa is not None:
            return self.lsa.embed(docs)
        if self.lsa is not None:
            return self.lsa.tfidf(docs)
        return local_tfidf(docs)

    def relevance(self, sents: list[Sentence], V) -> np.ndarray:
        if not sents:
            return np.zeros(0)
        if self.method == "textrank":
            return textrank([s.terms for s in sents], self.damping)
        c = np.asarray(V.mean(axis=0)).ravel()           # centroid (a sparse V stays sparse)
        nc = np.linalg.norm(c)
        return np.asarray(V @ (c / nc)).ravel() if nc > 0 else np.zeros(len(sents))


def textrank(docs: list[list[str]], damping: float = 0.85, iters: int = 100, tol: float = 1e-7) -> np.ndarray:
    """PageRank scores of the sentence graph, edge weight = shared terms / (log(1+|Si|) + log(1+|Sj|))."""
    from scipy.sparse import csr_matrix
    S = len(docs)
    if S == 0:
        return np.zeros(0)
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for r, terms in enumerate(docs):
        for t in set(terms):
            rows.append(r)
            cols.append(vocab.setdefault(t, len(vocab)))
    B = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(S, max(len(vocab), 1)))
    O = (B @ B.T).toarray()
    np.fill_diagonal(O, 0.0)
    ln = np.log1p(np.asarray([len(d) for d in docs], dtype=np.float64))
    W = O / np.maximum(ln[:, None] + ln[None, :], 1e-9)
    out = W.sum(axis=1)
    P = np.divide(W, out[:, None], out=np.zeros_like(W), where=out[:, None] > 0)
    score = np.ones(S)
    for _ in range(iters):
        new = (1 - damping) + damping * (P.T @ score)
        if np.abs(new - score).max() < tol:
            score = new
            break
        score = new
    return score / max(score.max(), 1e-12)


def _row(V, i: int) -> np.ndarray:
    r = V[i]
    return np.asarray(r.toarray()).ravel() if hasattr(r, "toarray") else np.asarray(r)


def pick(eligible: list[int], rel: np.ndarray, V, lengths: list[int], budget: int, lam: float = 1.0,
         redundancy: float = 1.0, overfill: bool = False) -> list[int]:
    """Greedy maximal-marginal-relevance selection under a word budget; returns indices in time order.

    score = lam * relevance - (1 - lam) * max cosine to a picked sentence; a sentence more similar than
    `redundancy` to a picked one is skipped. Without `overfill` a sentence that would exceed the budget is
    skipped (shorter ones may still fit, until 80% of the budget is used); with it the last sentence may
    run over (the summary is truncated to the budget for scoring, as in the DUC evaluations)."""
    cand = [i for i in eligible if lengths[i] > 0]
    if not cand or budget <= 0:
        return []
    maxsim = np.zeros(len(rel))
    chosen, total = [], 0
    while cand:
        c = np.asarray(cand)
        score = lam * rel[c] - (1 - lam) * maxsim[c]
        picked = None
        for o in np.argsort(-score, kind="stable"):
            i = cand[o]
            if chosen and maxsim[i] > redundancy:
                continue
            if not overfill and total + lengths[i] > budget:
                if total >= 0.8 * budget:
                    break
                continue
            picked = i
            break
        if picked is None:
            break
        chosen.append(picked)
        total += lengths[picked]
        cand.remove(picked)
        if total >= budget:
            break
        sims = np.asarray(V @ _row(V, picked)).ravel()
        maxsim = np.maximum(maxsim, sims)
    return sorted(chosen)


# ---------------------------------------------------------------- extractive summaries
@dataclass
class Part:
    """A stretch of the recording summarised on its own (a chapter, or the whole recording)."""
    index: int
    start: float
    end: float
    title: str = ""


def parts_of(chapters, duration: float, sents: list[Sentence]) -> list[Part]:
    if chapters and len(chapters) >= 2:
        return [Part(c.index, c.start, c.end, c.title) for c in chapters]
    end = max(duration, sents[-1].end if sents else 0.0)
    return [Part(1, 0.0, end, "")]


def members(part: Part, sents: list[Sentence], last: bool) -> list[int]:
    return [s.index for s in sents if part.start <= s.start < part.end or (last and s.start >= part.start)]


def _join(sents: list[Sentence], idx: list[int]) -> str:
    return " ".join(sents[i].text for i in idx)


def extractive(sents: list[Sentence], parts: list[Part], s, scorer: Scorer) -> dict:
    """Headline, overview and per-part summaries picked from `sents` (s = cfg.summary)."""
    lengths = [x.n_words for x in sents]
    ok = [x.n_words >= s.min_words and len(x.terms) >= 2 for x in sents]
    eligible = [i for i in range(len(sents)) if ok[i]] or [i for i in range(len(sents)) if lengths[i]]
    out = {"method": scorer.method, "headline": {"text": "", "sentences": []},
           "overview": {"text": "", "sentences": []}, "chapters": []}
    if not sents:
        return out
    V = scorer.vectors(sents)
    rel = scorer.relevance(sents, V)
    lam = s.mmr_lambda if scorer.method != "textrank" else 1.0
    over = pick(eligible, rel, V, lengths, s.overview_words, lam, s.redundancy)
    out["overview"] = {"text": _join(sents, over), "sentences": over}
    fits = [i for i in eligible if lengths[i] <= 1.6 * s.headline_words] or eligible
    if fits:
        best = max(fits, key=lambda i: (rel[i], -i))
        out["headline"] = {"text": sents[best].text, "sentences": [best]}
    if len(parts) >= 2:
        for k, p in enumerate(parts):
            idx = members(p, sents, k == len(parts) - 1)
            el = [i for i in idx if ok[i]] or [i for i in idx if lengths[i]]
            sub = [sents[i] for i in idx]
            if not sub:
                out["chapters"].append({"index": p.index, "start": p.start, "end": p.end, "title": p.title,
                                        "text": "", "sentences": []})
                continue
            Vs = V[idx]
            r = scorer.relevance(sub, Vs)
            local = {i: n for n, i in enumerate(idx)}
            chosen = pick([local[i] for i in el], r, Vs, [lengths[i] for i in idx], s.chapter_words, lam,
                          s.redundancy)
            chosen = [idx[n] for n in chosen]
            out["chapters"].append({"index": p.index, "start": p.start, "end": p.end, "title": p.title,
                                    "text": _join(sents, chosen), "sentences": chosen})
    return out


# ---------------------------------------------------------------- abstractive summaries (LLM)
SYSTEM = ("You write short, faithful summaries of transcripts of spoken recordings. Use only what the "
          "transcript says. Write plain prose in the third person and the past tense, for example \"The "
          "speaker explained ...\". No title, no preamble, no lists, no quotation marks around the summary.")
_PREFIX = re.compile(r"^\s*(?:\*\*)?(?:summary|headline|overview|part \d+ summary)(?:\*\*)?\s*:\s*", re.I)


def clock(t: float) -> str:
    s = int(max(t, 0))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def tidy(reply: str) -> str:
    """One paragraph of plain text from an LLM reply (labels, markdown and wrapping quotes removed)."""
    text = _PREFIX.sub("", reply.strip())
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)
    text = text.replace("**", "").replace("__", "")
    text = " ".join(text.split())
    if len(text) > 1 and text[0] in "\"“" and text[-1] in "\"”":
        text = text[1:-1].strip()
    return text


def transcript(sents: list[Sentence], idx: list[int]) -> str:
    """Transcript lines, one per speaker turn ("A: ..."), or plain text without speakers."""
    if not any(sents[i].speaker for i in idx):
        return " ".join(sents[i].text for i in idx)
    lines: list[list] = []
    for i in idx:
        s = sents[i]
        if lines and lines[-1][0] == s.speaker:
            lines[-1][1].append(s.text)
        else:
            lines.append([s.speaker, [s.text]])
    return "\n".join(f"{spk or 'Unknown'}: {' '.join(t)}" for spk, t in lines)


def chunk(sents: list[Sentence], idx: list[int], max_words: int) -> list[list[int]]:
    out, cur, n = [], [], 0
    for i in idx:
        if cur and n + sents[i].n_words > max_words:
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += sents[i].n_words
    if cur:
        out.append(cur)
    return out


def _ask(client, prompt: str, words: int) -> str:
    reply = client.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                        max_tokens=int(words * 2.5) + 64)
    return tidy(reply)


def llm_part(client, sents: list[Sentence], idx: list[int], kind: str, words: int, label: str,
             chunk_words: int) -> str:
    """Summary of one stretch from its transcript (long stretches: summaries of chunks, then combined)."""
    pieces = chunk(sents, idx, chunk_words)
    if len(pieces) == 1:
        head = f"Transcript of {label} of a {kind}" if label else f"Transcript of a {kind}"
        return _ask(client, f"{head}:\n\n{transcript(sents, idx)}\n\nSummarize this in at most {words} words.",
                    words)
    subs = []
    for k, p in enumerate(pieces):
        sec = f"section {k + 1} of {len(pieces)}"
        subs.append((sents[p[0]].start, sents[p[-1]].end,
                     llm_part(client, sents, p, kind, words, f"{label}, {sec}" if label else sec, chunk_words)))
    return llm_combine(client, subs, kind, words, label)


def llm_combine(client, summaries: list[tuple[float, float, str]], kind: str, words: int, label: str = "") -> str:
    """Overview from the summaries of consecutive parts."""
    body = "\n\n".join(f"Part {k + 1} ({clock(a)}-{clock(b)}): {t}" for k, (a, b, t) in enumerate(summaries))
    what = f"{label} of a {kind}" if label else f"the whole {kind}"
    return _ask(client, f"Summaries of the consecutive parts of {what}:\n\n{body}\n\n"
                        f"Write a summary of {what} in at most {words} words.", words)


def llm_headline(client, overview: str, kind: str, words: int) -> str:
    return _ask(client, f"Summary of a {kind}:\n\n{overview}\n\nWrite one sentence of at most {words} words "
                        f"that says what the {kind} is about.", words)


def llm_direct(client, sents: list[Sentence], kind: str, words: int) -> str:
    """The whole transcript in one prompt (only for recordings that fit the context)."""
    return _ask(client, f"Transcript of a {kind}:\n\n{transcript(sents, list(range(len(sents))))}\n\n"
                        f"Summarize it in at most {words} words.", words)


def abstractive(sents: list[Sentence], parts: list[Part], s, client, kind: str, chunk_words: int,
                tick: Callable[[], None] | None = None) -> dict:
    """Chapter summaries from their transcripts, the overview from those, the headline from the overview."""
    tick = tick or (lambda: None)
    out = {"model": client.model, "headline": {"text": ""}, "overview": {"text": ""}, "chapters": []}
    if not sents:
        return out
    every = list(range(len(sents)))
    if len(parts) >= 2:
        summaries = []
        for k, p in enumerate(parts):
            idx = members(p, sents, k == len(parts) - 1)
            tick()
            text = llm_part(client, sents, idx, kind, s.chapter_words, f"part {k + 1} of {len(parts)}",
                            chunk_words) if idx else ""
            out["chapters"].append({"index": p.index, "start": p.start, "end": p.end, "title": p.title, "text": text})
            if text:
                summaries.append((p.start, p.end, text))
        tick()
        overview = llm_combine(client, summaries, kind, s.overview_words)
    else:
        tick()
        overview = llm_part(client, sents, every, kind, s.overview_words, "", chunk_words)
    out["overview"] = {"text": overview}
    tick()
    out["headline"] = {"text": llm_headline(client, overview, kind, s.headline_words)}
    return out


# ---------------------------------------------------------------- entry point
@dataclass
class SummaryResult:
    sentences: list[Sentence]
    extractive: dict
    abstractive: dict | None = None
    llm_error: str | None = None
    settings: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        ex = self.extractive
        used = set(ex["headline"]["sentences"]) | set(ex["overview"]["sentences"])
        for c in ex["chapters"]:
            used |= set(c["sentences"])
        return {"settings": self.settings, "sentences": [self.sentences[i].to_json() for i in sorted(used)],
                "extractive": ex, "abstractive": self.abstractive, "llm_error": self.llm_error}


def recording_kind(cfg) -> str:
    kind = cfg.summary.kind
    if kind == "auto":
        return "meeting" if cfg.chapters.mode == "meeting" else "recording"
    return kind


def summarize(words: list[Word], cfg, chapters=None, duration: float = 0.0, lsa: LSA | None = None,
              client=None, tick: Callable[[], None] | None = None) -> SummaryResult:
    """Extractive summaries at three granularities, + the LLM's when `client` is given (an llm.Ollama).
    LLM failures do not fail the run: the extractive summary stays and `llm_error` says why."""
    s = cfg.summary
    if s.method not in METHODS:
        raise ValueError(f"unknown summary.method {s.method!r} (use {', '.join(METHODS)})")
    sents = make_sentences(words, s.max_sentence_words)
    parts = parts_of(chapters, duration, sents)
    method = s.method
    if method == "embed" and lsa is None:
        log.warning("no LSA model (%s): extractive summary with tfidf instead of embed", s.lsa_model)
        method = "tfidf"
    ex = extractive(sents, parts, s, Scorer(method, lsa, s.damping))
    settings = {"method": method, "headline_words": s.headline_words, "overview_words": s.overview_words,
                "chapter_words": s.chapter_words}
    res = SummaryResult(sents, ex, settings=settings)
    if client is not None:
        from .llm import LLMError
        try:
            res.abstractive = abstractive(sents, parts, s, client, recording_kind(cfg), cfg.llm.chunk_words, tick)
            res.abstractive["stats"] = client.stats.to_json()
        except LLMError as exc:
            log.warning("abstractive summary skipped: %s", exc)
            res.llm_error = str(exc)
    return res


# ---------------------------------------------------------------- output files
def to_markdown(res: SummaryResult, title: str = "Summary") -> str:
    """Readable summary: the LLM's text when there is one, and the key sentences with their times."""
    main = res.abstractive or res.extractive
    by_index = {x.index: x for x in res.sentences}
    out = [f"# {title}", ""]
    if main["headline"]["text"]:
        out += [f"**In one sentence:** {main['headline']['text']}", ""]
    out += ["## Overview", "", main["overview"]["text"] or "(no speech found)", ""]
    if main["chapters"]:
        out += ["## Chapters", ""]
        for c in main["chapters"]:
            head = f"**{c['index']}. {c['title']}**" if c.get("title") else f"**Part {c['index']}**"
            out += [f"{head} ({clock(c['start'])}-{clock(c['end'])})  ", c["text"] or "-", ""]
    key = res.extractive["overview"]["sentences"]
    if key:
        out += ["## Key sentences" + (" (picked from the recording)" if res.abstractive else ""), ""]
        for i in key:
            x = by_index[i]
            who = f"{x.speaker}: " if x.speaker else ""
            out.append(f"- [{clock(x.start)}] {who}{x.text}")
        out.append("")
    if res.abstractive:
        out.append(f"_Headline, overview and chapter summaries written by {res.abstractive['model']} "
                   "(a local LLM); they can contain mistakes._")
    else:
        out.append(f"_Extractive summary ({res.extractive['method']}): every sentence is quoted from the recording._")
    return "\n".join(out) + "\n"


def write_summary_files(res: SummaryResult, out_dir: str | Path, stem: str) -> dict[str, str]:
    from .utils.cache import save_json
    out = Path(out_dir)
    paths = {"summary_json": out / f"{stem}.summary.json", "summary_md": out / f"{stem}.summary.md"}
    save_json(paths["summary_json"], res.to_json())
    paths["summary_md"].write_text(to_markdown(res, f"Summary: {stem}"), encoding="utf-8", newline="\n")
    return {k: str(v) for k, v in paths.items()}

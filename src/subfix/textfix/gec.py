"""Seq2seq subtitle correction with an over-editing guard — imp.md 8.5.

The corrector is our own small Transformer encoder-decoder with our own BPE vocabulary
(textfix/seq2seq.py, bpe.py), trained from scratch by experiments/train_corrector.py on
synthetic noisy -> clean LibriSpeech-PC sentences plus real whisper-tiny confusions. It runs per
sentence (cues cut sentences), and the rewritten tokens are re-aligned to the original words so
timing and the diff survive.
"""
from __future__ import annotations

import logging

from .. import text as T
from ..subio import Word
from ..utils.device import resolve_device
from .common import retokenize

log = logging.getLogger(__name__)

DEFAULT_MODEL = "models/corrector"


class GEC:
    def __init__(self, model_dir: str = DEFAULT_MODEL, device: str = "auto", num_beams: int = 4,
                 max_length: int = 128, batch_size: int = 16):
        import torch
        from ..bpe import BPE
        from ..lm import resolve_model_path
        from .seq2seq import build_seq2seq
        self.torch = torch
        self.device = resolve_device(device)
        d = resolve_model_path(model_dir)
        if not (d / "model.pt").exists():
            raise FileNotFoundError(f"{d / 'model.pt'} not found: train it with python experiments/train_corrector.py")
        ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        self.bpe = BPE.load(d / "bpe.json")
        self.model = build_seq2seq(len(self.bpe), **ck["net"])
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.num_beams = num_beams
        self.max_length = min(max_length, self.model.max_len)
        self.batch_size = batch_size
        self.name = str(model_dir)

    def __call__(self, sents: list[str]) -> list[str]:
        """Corrected sentences; sentences the vocabulary cannot represent exactly (rare characters)
        or that are too long for the model are returned unchanged."""
        torch = self.torch
        from ..bpe import EOS, PAD
        out = list(sents)
        todo = []
        for k, s in enumerate(sents):
            ids = self.bpe.encode(s)
            if s.strip() and self.bpe.covers(s) and len(ids) + 1 <= self.max_length:
                todo.append((k, ids + [EOS]))
        todo.sort(key=lambda x: len(x[1]))
        for b in range(0, len(todo), self.batch_size):
            batch = todo[b:b + self.batch_size]
            L = max(len(ids) for _, ids in batch)
            src = torch.full((len(batch), L), PAD, dtype=torch.long)
            for r, (_, ids) in enumerate(batch):
                src[r, :len(ids)] = torch.tensor(ids)
            gen = self.model.generate(src.to(self.device), self.num_beams, min(self.max_length, L + 16))
            for (k, _), ids in zip(batch, gen):
                out[k] = self.bpe.decode(ids)
        return out

    def close(self) -> None:
        self.model = None


def edit_ratio(src: str, hyp: str) -> float:
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.normalized_distance(src.split(), hyp.split())


def guarded_gec(src: str, hyp: str, max_ratio: float = 0.25) -> str:
    """Reject GEC outputs that change too much (prevents paraphrasing/hallucination)."""
    if not hyp.strip():
        return src
    return hyp if edit_ratio(src, hyp) <= max_ratio else src


def sentence_spans(words: list[Word], max_words: int = 40) -> list[tuple[int, int]]:
    """Sentence index ranges; events and over-long runs also break sentences."""
    spans, start = [], 0
    for k, w in enumerate(words):
        if w.meta.get("event"):
            if k > start:
                spans.append((start, k))
            spans.append((k, k + 1))
            start = k + 1
            continue
        if T.ends_sentence(w.text) or (k + 1 - start) >= max_words:
            spans.append((start, k + 1))
            start = k + 1
    if start < len(words):
        spans.append((start, len(words)))
    return spans


def apply_gec(words: list[Word], gec: GEC, max_ratio: float = 0.25, max_sentence_words: int = 40,
              protect=None) -> tuple[list[Word], list[dict]]:
    """GEC per sentence; `protect` (audio-consistency guard) blocks edits to words the ASR confirmed."""
    spans = sentence_spans(words, max_sentence_words)
    todo = [(a, b) for a, b in spans if not (b - a == 1 and words[a].meta.get("event"))]
    if protect is not None:   # every word confirmed by the audio -> no edit could survive the guard
        todo = [(a, b) for a, b in todo if not all(protect(w) for w in words[a:b])]
    srcs = [" ".join(w.text for w in words[a:b]) for a, b in todo]
    hyps = gec(srcs) if srcs else []
    fixed = {}
    rejected = 0
    for (a, b), src, hyp in zip(todo, srcs, hyps):
        final = guarded_gec(src, hyp, max_ratio)
        if final != src:
            fixed[a] = final
        elif hyp != src:
            rejected += 1
    out: list[Word] = []
    edits: list[dict] = []
    for a, b in spans:
        if a in fixed:
            new_words, ed = retokenize(words[a:b], fixed[a].split(), "gec", offset=len(out), protect=protect)
            out.extend(new_words)
            edits.extend(ed)
        else:
            out.extend(words[a:b])
    if rejected:
        edits.append({"stage": "gec", "op": "guard", "old": None, "new": None, "index": -1,
                      "rejected_sentences": rejected})
    return out, edits

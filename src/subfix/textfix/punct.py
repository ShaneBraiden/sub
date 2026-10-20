"""Punctuation restoration as sequence labelling — our own BiLSTM tagger (imp.md 8.5).

Each word gets one label: "0" (none) "." "," "?". The network (trained from scratch by
experiments/train_punct.py on LibriSpeech-PC train text) reads, per word,
  * a word embedding (50k most frequent lowercase forms + <unk>),
  * a character-CNN embedding of the lowercase spelling (handles rare words and typos),
  * a casing embedding (lower / title / upper / mixed / none), so capitalised words that start
    a sentence are a cue when casing is present; half of the training text is lowercased, so the
    model also works on caseless subtitles,
and a 2-layer BiLSTM over the word sequence. Long transcripts are tagged in overlapping
windows; each word keeps the prediction from the window where it is farthest from an edge.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from .. import text as T
from ..subio import Word
from ..utils.device import resolve_device

log = logging.getLogger(__name__)

LABELS = ["0", ".", ",", "?"]
LABEL_TO_MARK = {"0": "", ".": ".", ",": ",", "?": "?", ":": ":", "-": ","}
CASES = ["none", "lower", "title", "upper", "mixed"]
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789'"
MAX_CHARS = 20
DEFAULT_MODEL = "models/punct"
_STRIP = re.compile(r"(?<!\d)[.,;:!?](?!\d)")


def label_of(token: str) -> str:
    """Training label of a punctuated token: its last sentence/clause mark."""
    marks = [ch for ch in T.trailing_punct(token) if ch in ".,?!;:"]
    if not marks:
        return "0"
    m = marks[-1]
    return {"!": ".", ";": ",", ":": ","}.get(m, m)


def char_ids(form: str) -> list[int]:
    """Character ids of a lowercase form (0 = pad, 1 = unknown character)."""
    return [CHARS.index(c) + 2 if c in CHARS else 1 for c in form[:MAX_CHARS]]


def build_net(n_words: int, word_dim: int = 192, char_dim: int = 32, char_filters: int = 64,
              hidden: int = 256, layers: int = 2, dropout: float = 0.3):
    import torch
    from torch import nn

    class PunctNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.word = nn.Embedding(n_words, word_dim, padding_idx=0)
            self.case = nn.Embedding(len(CASES), 8)
            self.char = nn.Embedding(len(CHARS) + 2, char_dim, padding_idx=0)
            self.conv = nn.Conv1d(char_dim, char_filters, kernel_size=3, padding=1)
            self.drop = nn.Dropout(dropout)
            self.lstm = nn.LSTM(word_dim + 8 + char_filters, hidden, num_layers=layers, batch_first=True,
                                bidirectional=True, dropout=dropout)
            self.out = nn.Linear(2 * hidden, len(LABELS))

        def forward(self, words, cases, chars):
            # words, cases: (B, L); chars: (B, L, C)
            B, L, C = chars.shape
            ce = self.char(chars.view(B * L, C)).transpose(1, 2)            # (B*L, char_dim, C)
            ce = torch.relu(self.conv(ce)).masked_fill((chars.view(B * L, 1, C) == 0), 0.0)
            ce = ce.max(dim=2).values.view(B, L, -1)
            x = torch.cat([self.word(words), self.case(cases), ce], dim=-1)
            h, _ = self.lstm(self.drop(x))
            return self.out(self.drop(h))

    return PunctNet()


class Featurizer:
    """Word -> (word id, case id, char ids); shared by training and inference."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab                       # index 0 = <pad>, 1 = <unk>
        self.ids = {w: i for i, w in enumerate(vocab)}

    def __call__(self, words: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(words)
        wid = np.zeros(n, dtype=np.int64)
        case = np.zeros(n, dtype=np.int64)
        chars = np.zeros((n, MAX_CHARS), dtype=np.int64)
        for k, w in enumerate(words):
            form = T.norm(w)
            wid[k] = self.ids.get(form, 1)
            case[k] = CASES.index(T.case_pattern(T.core(w)))
            ci = char_ids(form) or [1]
            chars[k, :len(ci)] = ci
        return wid, case, chars


class PunctuationRestorer:
    def __init__(self, model_dir: str = DEFAULT_MODEL, device: str = "auto", batch_size: int = 16):
        import torch
        from ..lm import resolve_model_path
        self.torch = torch
        self.device = resolve_device(device)
        path = resolve_model_path(model_dir) / "model.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: train it with python experiments/train_punct.py")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.feat = Featurizer(ck["vocab"])
        self.model = build_net(len(ck["vocab"]), **ck.get("net", {}))
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.batch_size = batch_size

    @staticmethod
    def clean(token: str) -> str:
        c = _STRIP.sub("", token)
        return c if c.strip() else token

    def _probs(self, chunks: list[list[str]]) -> list[np.ndarray]:
        torch = self.torch
        out: list[np.ndarray] = []
        for k in range(0, len(chunks), self.batch_size):
            batch = chunks[k:k + self.batch_size]
            L = max(len(c) for c in batch)
            W = np.zeros((len(batch), L), dtype=np.int64)
            Cs = np.zeros((len(batch), L), dtype=np.int64)
            Ch = np.zeros((len(batch), L, MAX_CHARS), dtype=np.int64)
            for b, c in enumerate(batch):
                w, cs, ch = self.feat(c)
                W[b, :len(c)], Cs[b, :len(c)], Ch[b, :len(c)] = w, cs, ch
            with torch.inference_mode():
                logits = self.model(*(torch.from_numpy(a).to(self.device) for a in (W, Cs, Ch)))
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            out.extend(probs[b, :len(c)] for b, c in enumerate(batch))
        return out

    def predict(self, words: list[str], chunk_words: int = 230, overlap: int = 30) -> list[tuple[str, float]]:
        clean = [self.clean(w) for w in words]
        n = len(clean)
        if not n:
            return []
        step = max(1, chunk_words - 2 * overlap) if n > chunk_words else chunk_words
        spans = []
        for s in range(0, n, step):
            spans.append((s, min(n, s + chunk_words)))
            if s + chunk_words >= n:
                break
        probs = self._probs([clean[s:e] for s, e in spans])
        best = np.full(n, -1.0)
        out: list[tuple[str, float]] = [("0", 0.0)] * n
        for (s, e), p in zip(spans, probs):
            for k in range(s, e):
                margin = min(k - s if s > 0 else n, e - 1 - k if e < n else n)
                if margin > best[k]:
                    best[k] = margin
                    lid = int(p[k - s].argmax())
                    out[k] = (LABELS[lid], float(p[k - s, lid]))
        return out

    def close(self) -> None:
        self.model = None


def add_mark(token: str, mark: str) -> str:
    """Insert a punctuation mark before closing quotes/brackets: 'word"' + '.' -> 'word."'."""
    pre, c, suf = T.split_affixes(token)
    closing = "".join(ch for ch in suf if ch not in T.PUNCT_MARKS)
    return pre + c + mark + closing


def restore(words: list[Word], restorer: PunctuationRestorer, strategy: str = "fill",
            min_conf: float = 0.0, chunk_words: int = 230, overlap: int = 30) -> tuple[list[Word], list[dict]]:
    """strategy='fill' keeps existing marks and only adds missing ones; 'replace' re-predicts all."""
    idx = [k for k, w in enumerate(words) if not w.meta.get("event")]
    preds = restorer.predict([words[k].text for k in idx], chunk_words, overlap)
    out = list(words)
    edits = []
    for k, (lab, p) in zip(idx, preds):
        w = words[k]
        mark = LABEL_TO_MARK.get(lab, "") if p >= min_conf else ""
        existing = T.trailing_punct(w.text)
        if strategy == "fill":
            if existing or not mark:
                continue
            new = add_mark(w.text, mark)
        else:
            base = T.strip_trailing_punct(w.text)
            if mark == "." and "!" in existing:
                mark = "!"
            new = add_mark(base, mark) if mark else base
        if new != w.text:
            nw = w.copy(text=new)
            nw.meta.setdefault("history", []).append({"stage": "punct", "from": w.text})
            out[k] = nw
            edits.append({"stage": "punct", "op": "format", "old": w.text, "new": new, "index": k,
                          "label": lab, "prob": round(p, 3)})
    # the transcript ends a sentence
    for k in range(len(out) - 1, -1, -1):
        if out[k].meta.get("event"):
            continue
        if not T.ends_sentence(out[k].text):
            base = T.strip_trailing_punct(out[k].text)
            nw = out[k].copy(text=add_mark(base, "."))
            edits.append({"stage": "punct", "op": "format", "old": out[k].text, "new": nw.text, "index": k})
            out[k] = nw
        break
    return out, edits

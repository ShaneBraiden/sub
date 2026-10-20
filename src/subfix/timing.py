"""Fine timing: word times for the *corrected* text + cue timing rules — imp.md 3.4 / 8.6.

"What was said" and "when it was said" are separate problems: after text correction the final
words are force-aligned to the audio with a dedicated CTC aligner instead of reusing Whisper's
DTW timestamps.

Backends (cfg.timing.backend):
  ctc     our own CTC acoustic model (subfix.acoustic, trained from scratch on LibriSpeech
          train-clean-100) + our NumPy CTC Viterbi — the default
  whisper Whisper DTW word timestamps (interpolated for words that did not come from ASR)
  keep    keep the provisional times as they are

Audio is aligned in chunks of <= chunk_max_s seconds, cut at pauses, with chunk_margin_s of
extra audio on both sides absorbed by <star>/wildcard tokens, so memory stays bounded.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from . import text as T
from .preprocess import SR, load_audio
from .subio import Cue, Word
from .utils.device import resolve_device

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ provisional times
def provisional_times(words: list[Word], audio_dur: float) -> np.ndarray:
    """(N, 2) start/end per word: known times kept, gaps interpolated by character length,
    then made monotonic."""
    n = len(words)
    st = np.array([w.start if w.start is not None else np.nan for w in words], dtype=float)
    en = np.array([w.end if w.end is not None else np.nan for w in words], dtype=float)
    known = ~np.isnan(st) & ~np.isnan(en)
    if not known.any():
        weights = np.array([max(len(T.norm(w.text)), 1) + 1 for w in words], float)
        edges = np.concatenate([[0.0], np.cumsum(weights)]) / weights.sum() * max(audio_dur, 1e-3)
        return np.stack([edges[:-1], edges[1:]], axis=1)
    k = 0
    while k < n:
        if known[k]:
            k += 1
            continue
        a = k
        while k < n and not known[k]:
            k += 1
        b = k  # words[a:b] unknown
        left = en[a - 1] if a > 0 else 0.0
        right = st[b] if b < n else max(audio_dur, left + 0.3 * (b - a))
        if right < left:
            right = left
        weights = np.array([max(len(T.norm(words[x].text)), 1) + 1 for x in range(a, b)], float)
        edges = left + (right - left) * np.concatenate([[0.0], np.cumsum(weights)]) / weights.sum()
        st[a:b], en[a:b] = edges[:-1], edges[1:]
    st = np.maximum.accumulate(np.clip(st, 0, None))
    en = np.maximum(en, st + 0.01)
    en = np.maximum.accumulate(en)
    return np.stack([st, en], axis=1)


def plan_chunks(times: np.ndarray, audio_dur: float, max_s: float = 60.0) -> list[tuple[int, int]]:
    """Word index ranges whose span is <= max_s, cut at the largest pause near the end."""
    n = len(times)
    chunks, a = [], 0
    while a < n:
        b = a
        while b < n and times[b, 1] - times[a, 0] <= max_s:
            b += 1
        if b == a:
            b = a + 1
        if b < n:
            lo = a + max(1, int(0.6 * (b - a)))
            gaps = times[lo:b, 0] - times[lo - 1:b - 1, 1] if b > lo else np.array([])
            if len(gaps):
                b = lo + int(np.argmax(gaps))
        chunks.append((a, b))
        a = b
    return chunks


# ------------------------------------------------------------------ CTC Viterbi (NumPy)
def ctc_viterbi(logp: np.ndarray, targets: list[int], blank: int) -> np.ndarray | None:
    """Best CTC path; returns the extended-state index per frame (odd = target token)."""
    T_, L = logp.shape[0], len(targets)
    if L == 0:
        return None
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = targets
    min_frames = L + sum(1 for k in range(1, L) if targets[k] == targets[k - 1])
    if T_ < min_frames:
        return None
    skip_ok = np.zeros(S, dtype=bool)
    skip_ok[3::2] = ext[3::2] != ext[1:-2:2]
    neg = -1e30
    alpha = np.full(S, neg)
    alpha[0] = logp[0, blank]
    alpha[1] = logp[0, ext[1]]
    bp = np.zeros((T_, S), dtype=np.int8)
    em = logp[:, ext]                       # (T, S)
    for t in range(1, T_):
        c1 = np.concatenate([[neg], alpha[:-1]])
        c2 = np.concatenate([[neg, neg], alpha[:-2]])
        c2 = np.where(skip_ok, c2, neg)
        stack = np.stack([alpha, c1, c2])
        choice = stack.argmax(axis=0)
        alpha = stack[choice, np.arange(S)] + em[t]
        bp[t] = choice
    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    path = np.empty(T_, dtype=np.int64)
    for t in range(T_ - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return path


def spans_from_path(path: np.ndarray, token_word: list[int], n_words: int) -> list[tuple[int, int] | None]:
    """Frame span [first, last+1) for each word from a CTC state path."""
    first = [None] * n_words
    last = [None] * n_words
    for t, s in enumerate(path):
        if s % 2 == 1:
            w = token_word[(s - 1) // 2]
            if w >= 0:
                if first[w] is None:
                    first[w] = t
                last[w] = t
    return [(f, l + 1) if f is not None else None for f, l in zip(first, last)]


DEFAULT_ALIGNER = "models/aligner"


class CTCAligner:
    """Our own CTC acoustic model (subfix.acoustic) + NumPy Viterbi with wildcard edges.

    A synthetic <star> column (best non-blank log-prob minus `star_penalty`) is placed before and
    after the transcript, so the extra audio margin around a chunk is absorbed instead of being
    forced onto the first/last word. Word times are shifted by the model's `onset_shift` (the
    constant CTC emission lag measured on dev when the model was selected)."""

    def __init__(self, model_dir: str = DEFAULT_ALIGNER, device: str = "auto", star_penalty: float = 0.5,
                 model=None, onset_shift: float = 0.0):
        import torch
        from .acoustic import BLANK, FRAME_S, build_acoustic
        self.torch = torch
        self.device = resolve_device(device)
        self.shift = onset_shift
        if model is None:
            from .lm import resolve_model_path
            path = resolve_model_path(model_dir) / "model.pt"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found: train it with python experiments/train_aligner.py")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = build_acoustic(**ck.get("net", {}))
            model.load_state_dict(ck["model"])
            model.to(self.device)
            self.shift = float(ck.get("onset_shift", 0.0))
        self.model = model.eval()
        self.blank = BLANK
        self.sep = 1
        self.stride = FRAME_S
        self.star_penalty = star_penalty

    def emissions(self, audio: np.ndarray) -> np.ndarray:
        torch = self.torch
        dev = next(self.model.parameters()).device
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(dev)
        with torch.inference_mode():
            lp, n = self.model(wav, torch.tensor([wav.shape[1]], device=dev))
        return lp[0, :int(n[0])].float().cpu().numpy()

    def _targets(self, tokens: list[str]) -> tuple[list[int], list[int]]:
        from .acoustic import TOKENS
        ids, owner = [], []
        for w, tok in enumerate(tokens):
            word_ids = [TOKENS.index(ch) for ch in tok.lower() if ch in TOKENS[2:]]
            if not word_ids:
                continue
            if ids:
                ids.append(self.sep)
                owner.append(-1)
            ids.extend(word_ids)
            owner.extend([w] * len(word_ids))
        return ids, owner

    def align(self, audio: np.ndarray, tokens: list[str]) -> list[tuple[float, float] | None]:
        lp = self.emissions(audio)
        V = lp.shape[1]
        nonblank = np.delete(lp, self.blank, axis=1)
        star_col = nonblank.max(axis=1, keepdims=True) - self.star_penalty
        lp = np.concatenate([lp, star_col], axis=1)
        ids, owner = self._targets(tokens)
        ids = [V] + ids + [V]
        owner = [-1] + owner + [-1]
        path = ctc_viterbi(lp, ids, self.blank)
        if path is None:
            return [None] * len(tokens)
        spans = spans_from_path(path, owner, len(tokens))
        return [(max(0.0, f * self.stride + self.shift), max(0.0, l * self.stride + self.shift))
                if sp is not None else None for sp in spans for f, l in [sp or (0, 0)]]

    def close(self) -> None:
        self.model = None


def make_aligner(backend: str, cfg, device: str = "auto"):
    if backend == "ctc":
        return CTCAligner(cfg.timing.model, device, cfg.timing.star_penalty)
    raise ValueError(f"no acoustic aligner for backend '{backend}'")


# ------------------------------------------------------------------ public API
def word_times(words: list[Word], wav_path: str, cfg, aligner=None, backend: str | None = None,
               audio: np.ndarray | None = None) -> tuple[list[Word], dict]:
    """Return copies of `words` with start/end from the chosen timing backend."""
    backend = backend or cfg.timing.backend
    if audio is None:
        audio = load_audio(wav_path)
    dur = len(audio) / SR
    prov = provisional_times(words, dur)
    out = [w.copy(start=float(s), end=float(e)) for w, (s, e) in zip(words, prov)]
    stats = {"backend": backend, "chunks": 0, "failed_chunks": 0, "aligned_words": 0}
    if backend in ("whisper", "keep") or not words:
        return out, stats
    if aligner is None:
        aligner = make_aligner(backend, cfg)
    margin = cfg.timing.chunk_margin_s
    for a, b in plan_chunks(prov, dur, cfg.timing.chunk_max_s):
        stats["chunks"] += 1
        idx = [k for k in range(a, b) if not words[k].meta.get("event") and T.spoken_form(words[k].text)]
        if not idx:
            continue
        t0 = max(0.0, prov[a, 0] - margin)
        t1 = min(dur, prov[b - 1, 1] + margin)
        seg = audio[int(t0 * SR):int(math.ceil(t1 * SR))]
        tokens = [T.spoken_form(words[k].text) for k in idx]
        try:
            spans = aligner.align(seg, tokens)
        except Exception as exc:
            log.warning("alignment failed for chunk %d-%d (%s); keeping provisional times", a, b, exc)
            spans = [None] * len(idx)
        if all(s is None for s in spans):
            stats["failed_chunks"] += 1
            continue
        for k, sp in zip(idx, spans):
            if sp is None:
                continue
            s, e = t0 + sp[0], t0 + max(sp[1], sp[0] + 0.02)
            out[k].start, out[k].end = s, e
            out[k].meta["timing"] = backend
            stats["aligned_words"] += 1
    # unaligned words (events, failures) -> re-interpolate between aligned neighbours
    for k, w in enumerate(out):
        if w.meta.get("timing") != backend:
            w.start = w.end = None
    fixed = provisional_times(out, dur)
    for w, (s, e) in zip(out, fixed):
        w.start, w.end = float(s), float(e)
    return out, stats


def finalize_cue_times(cues: list[Cue], min_dur: float = 0.833, max_dur: float = 7.0,
                       min_gap: float = 0.083, lead_in: float = 0.0, linger: float = 0.0) -> list[Cue]:
    """Cue start/end from word times with duration, gap and monotonicity guarantees:
    start_k < end_k <= start_{k+1} - min_gap for every k (overlaps are impossible)."""
    eps = 0.02
    n = len(cues)
    for c in cues:
        if c.words:
            ws = [w for w in c.words if w.start is not None]
            if ws:
                c.start = max(0.0, ws[0].start - lead_in)
                c.end = max(w.end for w in ws)
        c.end = max(c.end, c.start + eps)
    for k in range(1, n):   # monotonic starts
        cues[k].start = max(cues[k].start, cues[k - 1].start + eps)
    for k, c in enumerate(cues):
        nxt = cues[k + 1].start if k + 1 < n else math.inf
        end = c.end + linger
        end = max(end, c.start + min_dur)
        end = min(end, c.start + max_dur, nxt - min_gap)
        if end <= c.start:
            end = c.start + eps
            if k + 1 < n and cues[k + 1].start < end + min_gap:
                cues[k + 1].start = end + min_gap
                for m in range(k + 2, n):
                    if cues[m].start < cues[m - 1].start + eps:
                        cues[m].start = cues[m - 1].start + eps
        c.end = end
    for k in range(n):
        if cues[k].end <= cues[k].start:
            cues[k].end = cues[k].start + eps
    return cues

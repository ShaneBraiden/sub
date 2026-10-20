"""Interpolated modified Kneser-Ney n-gram language model (Chen & Goodman 1998), from scratch.

Counting and scoring are vectorized with NumPy: an n-gram is packed into one uint64 key
(`bits` bits per word id), counted with np.unique and looked up with np.searchsorted, so a
4-gram model over ~9M training words trains in about a minute and scores thousands of short
texts per second.

    P_k(w | h) = max(a_k(hw) - D_k(a_k(hw)), 0) / s_k(h) + gamma_k(h) * P_{k-1}(w | h')
    gamma_k(h) = (D_k1 * N1(h.) + D_k2 * N2(h.) + D_k3+ * N3+(h.)) / s_k(h)

a_N = raw counts at the highest order; a_k (k < N) = continuation counts N1+(. g), except for
n-grams that start with <s>, which have no left context and keep their raw counts. P_0 is
uniform over the vocabulary. Unseen histories back off with weight 1.

Open vocabulary: training words seen fewer than `min_count` times become <unk>; the probability
of an OOV word is P(<unk> | h) times a character-bigram spelling model trained on those rare
words, so a typo like "recieve" scores far below "receive" in the same context.
"""
from __future__ import annotations

import math
from array import array
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789'"
_CEND = len(_CHARS)          # end-of-word symbol for the spelling model
_CSTART = len(_CHARS) + 1    # start-of-word context


def _discounts(counts: np.ndarray) -> np.ndarray:
    """Modified KN discounts D1, D2, D3+ from the count-of-counts n1..n4."""
    n = [int(np.sum(counts == k)) for k in (1, 2, 3, 4)]
    if min(n) == 0:
        return np.array([0.5, 1.0, 1.5])
    y = n[0] / (n[0] + 2 * n[1])
    d = np.array([1 - 2 * y * n[1] / n[0], 2 - 3 * y * n[2] / n[1], 3 - 4 * y * n[3] / n[2]])
    return np.clip(d, 0.05, None)


class KneserNeyLM:
    def __init__(self, vocab: list[str], order: int, bits: int, tables: dict[str, np.ndarray],
                 spell_logp: np.ndarray):
        self.vocab = vocab
        self.ids = {w: i for i, w in enumerate(vocab)}
        self.order = order
        self.bits = bits
        self.t = tables
        self.spell_logp = spell_logp          # (len(_CHARS)+2, len(_CHARS)+1) log P(c | prev)
        self.unk = self.ids[UNK]
        self.bos = self.ids[BOS]
        self.eos = self.ids[EOS]

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, sentences: Callable[[], Iterable[Sequence[str]]], order: int = 4, min_count: int = 2,
              max_vocab: int = 65_000, log=print) -> "KneserNeyLM":
        """`sentences()` yields word sequences; it is called twice (vocabulary, then counting)
        so the corpus never has to be held in memory as Python strings."""
        freq: Counter = Counter()
        n_sent = 0
        for s in sentences():
            freq.update(s)
            n_sent += bool(s)
        keep = [w for w, c in freq.most_common(max_vocab) if c >= min_count and w not in (BOS, EOS, UNK)]
        vocab = [BOS, EOS, UNK] + keep
        ids = {w: i for i, w in enumerate(vocab)}
        bits = max(1, math.ceil(math.log2(len(vocab))))
        if bits * order > 64:
            raise ValueError(f"vocabulary of {len(vocab)} words does not fit a {order}-gram in 64 bits")
        rare = [w for w in freq if w not in ids]
        log(f"[lm] {n_sent} sentences, {sum(freq.values())} words, vocab {len(vocab)} "
            f"({len(rare)} rare types -> <unk>), {bits} bits/word")

        unk = ids[UNK]
        stream, seg = array("I"), array("i")
        n = 0
        for s in sentences():
            if not s:
                continue
            row = [ids[BOS]] + [ids.get(w, unk) for w in s] + [ids[EOS]]
            stream.extend(row)
            seg.extend([n] * len(row))
            n += 1
        stream = np.frombuffer(stream, dtype=np.uint32).astype(np.uint64)
        seg = np.frombuffer(seg, dtype=np.int32)
        B = np.uint64(bits)

        raw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for k in range(1, order + 1):
            m = len(stream) - k + 1
            ok = seg[:m] == seg[k - 1:k - 1 + m]
            key = np.zeros(int(ok.sum()), dtype=np.uint64)
            for j in range(k):
                key = (key << B) | stream[j:j + m][ok]
            raw[k] = np.unique(key, return_counts=True)

        tables: dict[str, np.ndarray] = {}
        bos = np.uint64(ids[BOS])
        for k in range(1, order + 1):
            keys, cnt = raw[k]
            if k == order:
                a = cnt.astype(np.int64)
            else:
                hk, _ = raw[k + 1]
                suffix = hk & np.uint64((1 << (bits * k)) - 1)
                suf, ncont = np.unique(suffix, return_counts=True)
                a = np.zeros(len(keys), dtype=np.int64)
                pos = np.searchsorted(keys, suf)
                a[pos] = ncont
                first = keys >> np.uint64(bits * (k - 1))
                a = np.where(first == bos, cnt, a)
            if k == 1:
                a[ids[BOS]] = 0                      # <s> is never predicted
            nz = a > 0
            keys, a = keys[nz], a[nz]
            D = _discounts(a)
            disc = np.where(a == 1, D[0], np.where(a == 2, D[1], D[2]))
            hist = keys >> B if k > 1 else np.zeros(len(keys), dtype=np.uint64)
            hkeys, start = np.unique(hist, return_index=True)
            s = np.add.reduceat(a, start).astype(np.float64)
            n1 = np.add.reduceat((a == 1).astype(np.int64), start)
            n2 = np.add.reduceat((a == 2).astype(np.int64), start)
            n3 = np.add.reduceat((a >= 3).astype(np.int64), start)
            gamma = (D[0] * n1 + D[1] * n2 + D[2] * n3) / s
            owner = np.searchsorted(hkeys, hist)
            tables[f"k{k}"] = keys
            tables[f"p{k}"] = (np.maximum(a - disc, 0.0) / s[owner]).astype(np.float32)
            tables[f"h{k}"] = hkeys
            tables[f"g{k}"] = gamma.astype(np.float32)
            log(f"[lm] order {k}: {len(keys)} n-grams, {len(hkeys)} histories, D = {np.round(D, 3).tolist()}")

        # character-bigram spelling model of rare (<unk>) words, add-0.5 smoothing
        C = np.full((len(_CHARS) + 2, len(_CHARS) + 1), 0.5)
        cidx = {c: i for i, c in enumerate(_CHARS)}
        for w in rare:
            prev = _CSTART
            for ch in w:
                c = cidx.get(ch)
                if c is None:
                    continue
                C[prev, c] += freq[w]
                prev = c
            C[prev, _CEND] += freq[w]
        spell = np.log(C / C.sum(axis=1, keepdims=True))
        return cls(vocab, order, bits, tables, spell.astype(np.float32))

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, vocab=np.array(self.vocab, dtype=object), order=self.order, bits=self.bits,
                 spell=self.spell_logp, **self.t)

    @classmethod
    def load(cls, path: str | Path) -> "KneserNeyLM":
        z = np.load(path, allow_pickle=True)
        order = int(z["order"])
        tables = {f"{p}{k}": z[f"{p}{k}"] for k in range(1, order + 1) for p in "kphg"}
        return cls(list(z["vocab"]), order, int(z["bits"]), tables, z["spell"])

    # ------------------------------------------------------------------ scoring
    def word_id(self, w: str) -> int:
        return self.ids.get(w, self.unk)

    def spelling_logp(self, w: str) -> float:
        """log P(spelling | <unk>) under the character-bigram model."""
        cidx = {c: i for i, c in enumerate(_CHARS)}
        lp, prev = 0.0, _CSTART
        for ch in w:
            c = cidx.get(ch)
            if c is None:
                continue
            lp += float(self.spell_logp[prev, c])
            prev = c
        return lp + float(self.spell_logp[prev, _CEND])

    def logprobs(self, hist: np.ndarray, words: np.ndarray) -> np.ndarray:
        """Natural-log P(w | h) for a batch. `hist` is (n, order-1) word ids, most recent last,
        with -1 for positions before the start of the available context."""
        n = len(words)
        B = np.uint64(self.bits)
        w = words.astype(np.uint64)
        p = np.full(n, 1.0 / (len(self.vocab) - 1))          # P_0: uniform (excluding <s>)
        for k in range(1, self.order + 1):
            keys, probs = self.t[f"k{k}"], self.t[f"p{k}"]
            hkeys, gam = self.t[f"h{k}"], self.t[f"g{k}"]
            if k == 1:
                avail = np.ones(n, dtype=bool)
                hkey = np.zeros(n, dtype=np.uint64)
            else:
                h = hist[:, self.order - k:]                 # the k-1 most recent words
                avail = np.all(h >= 0, axis=1)
                hkey = np.zeros(n, dtype=np.uint64)
                for j in range(k - 1):
                    hkey = (hkey << B) | np.where(avail, h[:, j], 0).astype(np.uint64)
            hi = np.searchsorted(hkeys, hkey)
            hi_c = np.minimum(hi, len(hkeys) - 1)
            seen = avail & (hkeys[hi_c] == hkey)
            g = np.where(seen, gam[hi_c], 1.0)
            key = (hkey << B) | w
            ki = np.minimum(np.searchsorted(keys, key), len(keys) - 1)
            hit = seen & (keys[ki] == key)
            top = np.where(hit, probs[ki], 0.0)
            p = np.where(avail, top + g * p, p)
        return np.log(np.maximum(p, 1e-30))

    def score_sequences(self, seqs: list[tuple[list[str], bool, bool]]) -> list[float]:
        """Total log-prob of each (words, starts_sentence, ends_sentence) piece; OOV words add
        their spelling log-prob."""
        rows_h, rows_w, owner, extra = [], [], [], np.zeros(len(seqs))
        H = self.order - 1
        for n, (words, starts, ends) in enumerate(seqs):
            ctx = [self.bos] if starts else []
            ids = [self.word_id(x) for x in words] + ([self.eos] if ends else [])
            for x, i in zip(words + [EOS], ids):
                h = ctx[-H:] if H else []
                rows_h.append([-1] * (H - len(h)) + h)
                rows_w.append(i)
                owner.append(n)
                if i == self.unk:
                    extra[n] += self.spelling_logp(x)
                ctx.append(i)
        if not rows_w:
            return [0.0] * len(seqs)
        lp = self.logprobs(np.asarray(rows_h, dtype=np.int64).reshape(len(rows_w), H), np.asarray(rows_w))
        tot = np.bincount(np.asarray(owner), weights=lp, minlength=len(seqs)) + extra
        return tot.tolist()

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        seqs = [(list(s), True, True) for s in sentences if s]
        lp = sum(self.score_sequences(seqs))
        n = sum(len(s) + 1 for s, _, _ in seqs)
        return math.exp(-lp / n)

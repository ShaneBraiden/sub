"""Byte-pair-encoding subword tokenizer (Sennrich et al. 2016), trained from scratch.

Whitespace tokens keep their case and attached punctuation; the last symbol of every token
carries an end-of-word marker "</w>", so decoding is exact: join the pieces and turn "</w>"
into a space. Training uses incremental pair counts with a lazy max-heap, so 8k merges over
~150k word types take about a minute in pure Python.
"""
from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

EOW = "</w>"
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]
PAD, BOS, EOS, UNK = range(4)


def _symbols(word: str) -> list[str]:
    return list(word[:-1]) + [word[-1] + EOW]


class BPE:
    def __init__(self, merges: list[tuple[str, str]], alphabet: list[str]):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {m: r for r, m in enumerate(self.merges)}
        self.alphabet = list(alphabet)
        self.vocab = SPECIALS + self.alphabet + ["".join(m) for m in self.merges]
        self.vocab = list(dict.fromkeys(self.vocab))
        self.ids = {t: i for i, t in enumerate(self.vocab)}
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, words: Iterable[str], vocab_size: int = 8000, min_char_count: int = 20, log=print) -> "BPE":
        freq = Counter(w for w in words if w)
        chars = Counter()
        for w, f in freq.items():
            for s in _symbols(w):
                chars[s] += f
        alphabet = sorted(c for c, n in chars.items() if n >= min_char_count)
        allowed = set(alphabet)
        types = [w for w in freq if all(s in allowed for s in _symbols(w))]
        seqs = [_symbols(w) for w in types]
        fr = [freq[w] for w in types]
        pairs: Counter = Counter()
        where: dict[tuple[str, str], set[int]] = defaultdict(set)
        for i, s in enumerate(seqs):
            for p in zip(s, s[1:]):
                pairs[p] += fr[i]
                where[p].add(i)
        heap = [(-c, p) for p, c in pairs.items()]
        heapq.heapify(heap)
        n_merges = vocab_size - len(SPECIALS) - len(alphabet)
        merges: list[tuple[str, str]] = []
        while len(merges) < n_merges and heap:
            negc, p = heapq.heappop(heap)
            if pairs.get(p, 0) != -negc or negc == 0:
                continue                      # stale heap entry
            a, b = p
            merged = a + b
            merges.append(p)
            changed: set[tuple[str, str]] = set()
            for i in list(where.pop(p, ())):
                s, f = seqs[i], fr[i]
                new, k = [], 0
                while k < len(s):
                    if k + 1 < len(s) and s[k] == a and s[k + 1] == b:
                        new.append(merged)
                        k += 2
                    else:
                        new.append(s[k])
                        k += 1
                old_pairs = list(zip(s, s[1:]))
                new_pairs = list(zip(new, new[1:]))
                for q in old_pairs:
                    pairs[q] -= f
                    changed.add(q)
                for q in new_pairs:
                    pairs[q] += f
                    changed.add(q)
                for q in set(old_pairs) - set(new_pairs):
                    if q in where:
                        where[q].discard(i)
                for q in new_pairs:
                    where[q].add(i)
                seqs[i] = new
            pairs.pop(p, None)
            for q in changed:
                c = pairs.get(q, 0)
                if c > 0 and q != p:
                    heapq.heappush(heap, (-c, q))
            if len(merges) % 1000 == 0:
                log(f"[bpe] {len(merges)} merges, last {merged!r} ({-negc})")
        return cls(merges, alphabet)

    # ------------------------------------------------------------------ encoding
    def encode_word(self, word: str) -> list[int]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        syms = [s if s in self.ids else "<unk>" for s in _symbols(word)]
        while len(syms) > 1:
            best, best_rank = None, None
            for k, p in enumerate(zip(syms, syms[1:])):
                r = self.ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = k, r
            if best is None:
                break
            syms = syms[:best] + [syms[best] + syms[best + 1]] + syms[best + 2:]
        ids = [self.ids.get(s, UNK) for s in syms]
        if len(self._cache) < 500_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        return [i for w in text.split() for i in self.encode_word(w)]

    def decode(self, ids: Iterable[int]) -> str:
        toks = [self.vocab[i] for i in ids if i >= len(SPECIALS)]
        return " ".join("".join(toks).replace(EOW, " ").split())

    def covers(self, text: str) -> bool:
        """True if every symbol is in the alphabet (so decode(encode(text)) == text)."""
        return all(s in self.ids for w in text.split() for s in _symbols(w))

    def __len__(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"alphabet": self.alphabet, "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPE":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in d["merges"]], d["alphabet"])

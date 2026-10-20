"""Synthetic corruption generator — imp.md 5.3.

Applies controllable, logged corruptions to clean reference subtitles:

  spelling   char insert/delete/swap/keyboard-adjacent substitution      3% of words
  homophone  swap within a homophone group (their/there/they're ...)     2%
  asr        real whisper-tiny substitution errors on the same audio     3% (RoBART-style)
  entity     proper nouns -> metaphone neighbour from the vocabulary     30% of entities
  drop       drop function words                                          2%
  dup        duplicate a word                                             1%
  punct/case strip all marks or drop 50% of them; lowercase              per file
  offset     U(-5 s, +5 s)                                                per file
  drift      x 25/23.976 or 23.976/25                                     30% of files
  jitter     N(0, 300 ms) per boundary                                    per cue
  split      offset jump at a random cue (simulated ad cut)              20% of files
  merge      merge 2-3 cues into one long line (60+ chars)               ~20% of cues

Every injected text error is logged with its reference token index so per-error-type recall
can be computed (metrics in experiments/run_eval.py).

Usage: python tools/corrupt.py [--ids ...] [--rate-scale 1.0] [--variant default] [--seed 0]
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subfix import text as T  # noqa: E402
from subfix.align import align, metaphone  # noqa: E402
from subfix.paths import DATA_DIR  # noqa: E402
from subfix.subio import Cue, read_srt, tokenize, write_srt  # noqa: E402
from subfix.utils.cache import save_json  # noqa: E402

DEFAULT_RATES = {"entity": 0.30, "asr": 0.03, "homophone": 0.02, "spelling": 0.03, "drop": 0.02, "dup": 0.01}
TEXT_ORDER = ["entity", "asr", "homophone", "spelling", "drop", "dup"]
TOTAL_WORD_RATE = sum(v for k, v in DEFAULT_RATES.items() if k != "entity")   # 0.11


@dataclass
class TimingParams:
    max_offset: float = 5.0
    drift_prob: float = 0.3
    jitter_sd: float = 0.3
    split_prob: float = 0.2
    merge_start_prob: float = 0.09
    punct_modes: tuple = ("strip_all", "drop_half")
    lowercase_prob: float = 0.5


_ROWS = ["1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm"]


@lru_cache(maxsize=1)
def keyboard_neighbors() -> dict[str, str]:
    pos = {ch: (r, c) for r, row in enumerate(_ROWS) for c, ch in enumerate(row)}
    out = {}
    for ch, (r, c) in pos.items():
        out[ch] = "".join(o for o, (r2, c2) in pos.items() if o != ch and abs(r - r2) <= 1 and abs(c - c2) <= 1
                          and o.isalpha())
    return out


def typo(word_core: str, rng: random.Random) -> str:
    letters = [i for i, ch in enumerate(word_core) if ch.isalpha()]
    if len(letters) < 3:
        return word_core
    w = list(word_core)
    i = rng.choice(letters[1:]) if len(letters) > 1 else letters[0]
    kind = rng.choice(["insert", "delete", "swap", "substitute"])
    low = w[i].lower()
    neigh = keyboard_neighbors().get(low) or "e"
    if kind == "insert":
        w.insert(i, rng.choice(neigh))
    elif kind == "delete":
        del w[i]
    elif kind == "swap" and i + 1 < len(w) and w[i + 1].isalpha() and w[i] != w[i + 1]:
        w[i], w[i + 1] = w[i + 1], w[i]
    else:
        new = rng.choice(neigh)
        w[i] = new.upper() if w[i].isupper() else new
    out = "".join(w)
    return out if out.lower() != word_core.lower() else word_core[:-1] + ("x" if word_core[-1] != "x" else "z")


@lru_cache(maxsize=1)
def _vocab_by_metaphone() -> dict[str, list[str]]:
    from importlib.resources import files
    path = files("symspellpy") / "frequency_dictionary_en_82_765.txt"
    idx: dict[str, list[str]] = {}
    with open(str(path), encoding="utf-8") as f:
        for line in f:
            w = line.split(" ")[0]
            if w.isalpha() and len(w) >= 3:
                idx.setdefault(metaphone(w), []).append(w)
    return idx


def entity_neighbor(core: str, rng: random.Random) -> str | None:
    low = core.lower()
    cands = [w for w in _vocab_by_metaphone().get(metaphone(low), []) if w != low and abs(len(w) - len(low)) <= 2]
    if not cands:
        return None
    return T.apply_case("title", rng.choice(sorted(cands)[:50]))


def asr_confusions(ref_toks: list[str], tiny_words: list[str]) -> dict[int, str]:
    """ref position -> whisper-tiny token that replaced it (1:1 substitutions only)."""
    out = {}
    for op, i, j in align(ref_toks, tiny_words):
        if op == "sub" and T.canon(tiny_words[j]) and T.canon(ref_toks[i]) != T.canon(tiny_words[j]):
            out[i] = tiny_words[j]
    return out


def _transplant(orig: str, new_core: str) -> str:
    pre, c, suf = T.split_affixes(orig)
    _, nc, _ = T.split_affixes(new_core)
    pat = T.case_pattern(c)
    return pre + (T.apply_case(pat, nc) if pat in ("title", "upper") else nc.lower()) + suf


@dataclass
class Corrupted:
    cues: list[Cue]
    log: dict = field(default_factory=dict)


def corrupt_clip(ref_cues: list[Cue], clip_id: str, has_punct: bool, tiny_words: list[str] | None,
                 seed: int = 0, rate_scale: float = 1.0, rates: dict | None = None,
                 tp: TimingParams | None = None) -> Corrupted:
    rates = dict(DEFAULT_RATES, **(rates or {}))
    tp = tp or TimingParams()
    h = int(hashlib.sha1(f"{clip_id}:{seed}:{rate_scale}".encode()).hexdigest()[:8], 16)
    rng = random.Random(h)
    # flatten reference tokens
    toks, cue_of = [], []
    for k, c in enumerate(ref_cues):
        for t in tokenize(c.text.replace("\n", " ")):
            toks.append(t)
            cue_of.append(k)
    n = len(toks)
    sent_start = [True] + [T.ends_sentence(t) for t in toks[:-1]]
    hom = T.homophone_groups()
    func = T.function_words()
    conf = asr_confusions(toks, tiny_words) if tiny_words else {}
    eligible = {
        "entity": [i for i in range(n) if has_punct and not sent_start[i] and T.core(toks[i])[:1].isupper()
                   and T.core(toks[i]).isalpha() and len(T.core(toks[i])) >= 3 and T.norm(toks[i]) != "i"],
        "asr": sorted(conf),
        "homophone": [i for i in range(n) if T.core(toks[i]).lower() in hom],
        "spelling": [i for i in range(n) if sum(ch.isalpha() for ch in T.core(toks[i])) >= 3],
        "drop": [i for i in range(n) if T.norm(toks[i]) in func and not T.trailing_punct(toks[i])],
        "dup": list(range(n)),
    }
    used: set[int] = set()
    assign: dict[int, str] = {}
    for kind in TEXT_ORDER:
        pool = [i for i in eligible[kind] if i not in used]
        if kind == "entity":
            k = round(min(1.0, rates[kind] * rate_scale) * len(eligible[kind]))
        else:
            k = round(rates[kind] * rate_scale * n)
        k = min(k, len(pool))
        for i in rng.sample(pool, k):
            assign[i] = kind
            used.add(i)
    # apply word-level corruption
    new_cue_toks: list[list[str]] = [[] for _ in ref_cues]
    errors = []
    src_len = 0
    for i, tok in enumerate(toks):
        kind = assign.get(i)
        c = cue_of[i]
        pre, core, suf = T.split_affixes(tok)
        if kind is None:
            new_cue_toks[c].append(tok)
            src_len += 1
            continue
        if kind == "drop":
            errors.append({"type": "drop", "ref_index": i, "src_index": src_len, "orig": tok, "new": None})
            continue
        if kind == "dup":
            new_cue_toks[c].extend([pre + core, tok])
            errors.append({"type": "dup", "ref_index": i, "src_index": src_len, "orig": tok, "new": pre + core})
            src_len += 2
            continue
        if kind == "spelling":
            new = pre + typo(core, rng) + suf
        elif kind == "homophone":
            opts = hom[core.lower()]
            new = _transplant(tok, rng.choice(opts))
        elif kind == "asr":
            new = _transplant(tok, conf[i])
        else:  # entity
            nb = entity_neighbor(core, rng)
            new = pre + (nb if nb else typo(core, rng)) + suf
        if T.canon(new) == T.canon(tok):
            new = pre + typo(core, rng) + suf
        new_cue_toks[c].append(new)
        errors.append({"type": kind, "ref_index": i, "src_index": src_len, "orig": tok, "new": new})
        src_len += 1
    # punctuation / case (per file)
    punct_mode = rng.choice(tp.punct_modes) if has_punct else "none"
    lowercase = has_punct and rng.random() < tp.lowercase_prob
    for c_toks in new_cue_toks:
        for k, t in enumerate(c_toks):
            if punct_mode == "strip_all" or (punct_mode == "drop_half" and T.trailing_punct(t) and rng.random() < 0.5):
                t = T.strip_trailing_punct(t)
            if lowercase:
                t = t.lower()
            c_toks[k] = t
    # cue merges (formatting) + timing
    offset = rng.uniform(-tp.max_offset, tp.max_offset)
    drift = rng.choice([25 / 23.976, 23.976 / 25]) if rng.random() < tp.drift_prob else 1.0
    split_at, jump = None, 0.0
    if rng.random() < tp.split_prob and len(ref_cues) >= 8:
        split_at = rng.randint(len(ref_cues) // 4, 3 * len(ref_cues) // 4)
        jump = rng.uniform(2.0, 8.0) * rng.choice([-1, 1])
    groups, merges = [], []
    k = 0
    while k < len(ref_cues):
        m = 1
        if rng.random() < tp.merge_start_prob and k + 1 < len(ref_cues):
            m = min(rng.choice([2, 3]), len(ref_cues) - k)
            merges.append([k, m])
        groups.append(list(range(k, k + m)))
        k += m
    out_cues = []
    for g in groups:
        text_toks = [t for c in g for t in new_cue_toks[c]]
        if not text_toks:
            continue
        lines = [" ".join(text_toks)] if len(g) > 1 else \
            _relayout(ref_cues[g[0]].text, new_cue_toks[g[0]])

        def warp(t: float, cue_idx: int) -> float:
            t2 = drift * t + offset
            if split_at is not None and cue_idx >= split_at:
                t2 += jump
            return t2
        s = warp(ref_cues[g[0]].start, g[0]) + rng.gauss(0, tp.jitter_sd)
        e = warp(ref_cues[g[-1]].end, g[-1]) + rng.gauss(0, tp.jitter_sd)
        s = max(0.0, s)
        e = max(e, s + 0.2)
        out_cues.append(Cue(len(out_cues) + 1, s, e, "\n".join(lines)))
    out_cues.sort(key=lambda c: c.start)
    for i, c in enumerate(out_cues):
        c.index = i + 1
    log = {"id": clip_id, "seed": seed, "rate_scale": rate_scale, "n_ref_tokens": n, "n_src_tokens": src_len,
           "text_errors": errors, "punct_mode": punct_mode, "lowercase": lowercase,
           "timing": {"offset": offset, "drift": drift, "split_at_cue": split_at, "split_jump": jump,
                      "jitter_sd": tp.jitter_sd},
           "merges": merges, "counts": {k: sum(1 for e in errors if e["type"] == k) for k in TEXT_ORDER}}
    return Corrupted(out_cues, log)


def _relayout(orig_text: str, new_toks: list[str]) -> list[str]:
    """Keep the original line split position (by token count) for an unmerged cue."""
    lines = orig_text.split("\n")
    if len(lines) < 2 or not new_toks:
        return [" ".join(new_toks)]
    first = len(tokenize(lines[0]))
    first = min(max(first, 1), len(new_toks))
    l1, l2 = " ".join(new_toks[:first]), " ".join(new_toks[first:])
    return [l1, l2] if l2 else [l1]


def corrupt_file(clip: dict, variant: str = "default", seed: int = 0, rate_scale: float = 1.0,
                 tiny_model: str = "tiny.en") -> dict:
    from subfix.asr import transcribe
    ref_srt = DATA_DIR / "refs" / f"{clip['id']}.srt"
    cues = read_srt(ref_srt)
    tiny = transcribe(str(ROOT / clip["audio"]), model=tiny_model)
    res = corrupt_clip(cues, clip["id"], clip.get("has_punct", False), [w.text for w in tiny.words],
                       seed, rate_scale)
    out_dir = DATA_DIR / "corrupted" / variant
    write_srt(res.cues, out_dir / f"{clip['id']}.srt")
    save_json(out_dir / f"{clip['id']}.log.json", res.log)
    return res.log


def main() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    from fetch_data import all_clips
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--variant", default="default")
    ap.add_argument("--rate-scale", type=float, default=1.0,
                    help=f"multiplies word error rates (default total {TOTAL_WORD_RATE:.0%})")
    ap.add_argument("--target-rate", type=float, default=None, help="total word error rate, e.g. 0.05")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    scale = args.target_rate / TOTAL_WORD_RATE if args.target_rate else args.rate_scale
    for clip in all_clips():
        if args.ids and clip["id"] not in args.ids:
            continue
        if not (DATA_DIR / "refs" / f"{clip['id']}.srt").exists():
            print(f"[corrupt] skip {clip['id']} (no reference yet)")
            continue
        log = corrupt_file(clip, args.variant, args.seed, scale)
        print(f"[corrupt] {clip['id']} {args.variant}: {log['counts']} punct={log['punct_mode']} "
              f"lower={log['lowercase']} offset={log['timing']['offset']:.2f} drift={log['timing']['drift']:.4f}")


if __name__ == "__main__":
    main()

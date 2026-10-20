"""Text metrics — imp.md 6.1.

* WER / CER (jiwer) after Whisper's EnglishTextNormalizer (lowercase, no punctuation, numbers
  and spellings standardised) so raw-Whisper baselines are not penalised for formatting.
* Word-level edit P / R / F0.5 against the gold edits (corrupted -> reference) plus the
  over-correction rate (% of already-correct words that a system changed).
* ERRANT P / R / F0.5 per error type, computed per source cue projected onto ref/hyp.
* GLEU (NLTK, Google variant) and BLEU (sacrebleu) as secondary scores.
* Punctuation per-class P/R/F1 for . , ? + macro-F1, and casing accuracy, on aligned words.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from functools import lru_cache

from subfix import text as T


# ------------------------------------------------------------------ normalisation
@lru_cache(maxsize=1)
def _normalizer():
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
        return EnglishTextNormalizer()
    except Exception:  # pragma: no cover - fallback
        def basic(s: str) -> str:
            s = T.canon_quotes(s).lower()
            s = re.sub(r"[^\w\s']", " ", s)
            return re.sub(r"\s+", " ", s).strip()
        return basic


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", _normalizer()(s)).strip()


def tokens(s: str) -> list[str]:
    """Surface tokens (punctuation attached), as used by the subtitle pipeline."""
    from subfix.subio import tokenize
    return tokenize(s.replace("\n", " "))


def norm_tokens(s: str) -> list[str]:
    return [n for n in (T.norm(t) for t in tokens(s)) if n]


# ------------------------------------------------------------------ WER / CER
def wer_cer(refs: list[str], hyps: list[str]) -> dict:
    import jiwer
    pairs = [(normalize_text(r), normalize_text(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return {"wer": float("nan"), "cer": float("nan")}
    r, h = [p[0] for p in pairs], [p[1] if p[1] else "<empty>" for p in pairs]
    w = jiwer.process_words(r, h)
    c = jiwer.cer(r, h)
    n_ref = w.hits + w.substitutions + w.deletions
    return {"wer": float(w.wer), "cer": float(c), "sub": w.substitutions, "del": w.deletions,
            "ins": w.insertions, "n_ref_words": n_ref}


# ------------------------------------------------------------------ edit P/R/F0.5
def atomic_edits(src: list[str], tgt: list[str]) -> Counter:
    from rapidfuzz.distance import Levenshtein
    out: Counter = Counter()
    for op in Levenshtein.editops(src, tgt):
        if op.tag == "replace":
            out[("R", op.src_pos, tgt[op.dest_pos])] += 1
        elif op.tag == "delete":
            out[("D", op.src_pos, None)] += 1
        else:
            out[("I", op.src_pos, tgt[op.dest_pos])] += 1
    return out


def prf(tp: int, fp: int, fn: int, beta: float = 0.5) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if (p + r) else 0.0
    return p, r, f


def edit_scores(srcs: list[list[str]], refs: list[list[str]], hyps: list[list[str]], beta: float = 0.5) -> dict:
    tp = fp = fn = 0
    over = ok_total = 0
    for s, r, h in zip(srcs, refs, hyps):
        g, p = atomic_edits(s, r), atomic_edits(s, h)
        inter = sum((g & p).values())
        tp += inter
        fp += sum(p.values()) - inter
        fn += sum(g.values()) - inter
        gold_touched = {e[1] for e in g if e[0] in "RD"}
        hyp_touched = {e[1] for e in p if e[0] in "RD"}
        ok = set(range(len(s))) - gold_touched
        ok_total += len(ok)
        over += len(ok & hyp_touched)
    P, R, F = prf(tp, fp, fn, beta)
    return {"edit_p": P, "edit_r": R, "edit_f05": F, "edit_tp": tp, "edit_fp": fp, "edit_fn": fn,
            "overcorrection": over / ok_total if ok_total else 0.0}


# ------------------------------------------------------------------ ERRANT (chunked)
@lru_cache(maxsize=1)
def _errant():
    import errant
    return errant.load("en")


def position_map(src_norm: list[str], tgt_norm: list[str]) -> list[int]:
    """For every src position p (and p = len(src)) the tgt index where it starts."""
    from rapidfuzz.distance import Levenshtein
    m = [0] * (len(src_norm) + 1)
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(src_norm, tgt_norm):
        for k in range(i1, i2):
            m[k] = j1 + min(k - i1, max(j2 - j1 - 1, 0)) if tag in ("equal", "replace") else j1
    m[len(src_norm)] = len(tgt_norm)
    # make monotone
    for k in range(1, len(m)):
        m[k] = max(m[k], m[k - 1])
    return m


def errant_scores(src_chunks: list[list[str]], ref_surface: list[str], hyp_surface: list[str],
                  beta: float = 0.5) -> dict:
    """src_chunks: surface tokens of each source cue (in order); ref/hyp: full surface token lists."""
    ann = _errant()
    src_flat = [t for ch in src_chunks for t in ch]
    sn = [T.norm(t) or t for t in src_flat]
    rmap = position_map(sn, [T.norm(t) or t for t in ref_surface])
    hmap = position_map(sn, [T.norm(t) or t for t in hyp_surface])
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    a = 0
    for ch in src_chunks:
        b = a + len(ch)
        o = " ".join(src_flat[a:b])
        rc = " ".join(ref_surface[rmap[a]:rmap[b]])
        hc = " ".join(hyp_surface[hmap[a]:hmap[b]])
        a = b
        if not o.strip():
            continue
        orig = ann.parse(o)
        gold = {(e.o_start, e.o_end, e.c_str): e.type for e in ann.annotate(orig, ann.parse(rc))} if rc != o else {}
        hyp = {(e.o_start, e.o_end, e.c_str): e.type for e in ann.annotate(orig, ann.parse(hc))} if hc != o else {}
        for k, typ in gold.items():
            if k in hyp:
                tp[typ] += 1
            else:
                fn[typ] += 1
        for k, typ in hyp.items():
            if k not in gold:
                fp[typ] += 1
    P, R, F = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()), beta)
    per_type = {}
    for typ in set(tp) | set(fp) | set(fn):
        p, r, f = prf(tp[typ], fp[typ], fn[typ], beta)
        per_type[typ] = {"tp": tp[typ], "fp": fp[typ], "fn": fn[typ], "p": p, "r": r, "f05": f}
    return {"errant_p": P, "errant_r": R, "errant_f05": F, "errant_tp": sum(tp.values()),
            "errant_fp": sum(fp.values()), "errant_fn": sum(fn.values()), "errant_types": per_type}


# ------------------------------------------------------------------ GLEU / BLEU
def gleu_bleu(refs: list[str], hyps: list[str]) -> dict:
    out = {}
    try:
        from nltk.translate.gleu_score import corpus_gleu
        out["gleu"] = float(corpus_gleu([[normalize_text(r).split()] for r in refs],
                                        [normalize_text(h).split() for h in hyps]))
    except Exception:
        out["gleu"] = float("nan")
    try:
        import sacrebleu
        out["bleu"] = float(sacrebleu.corpus_bleu([normalize_text(h) for h in hyps],
                                                  [[normalize_text(r) for r in refs]]).score)
    except Exception:
        out["bleu"] = float("nan")
    return out


# ------------------------------------------------------------------ punctuation / casing
PUNCT_CLASSES = [".", ",", "?"]


def punct_class(token: str) -> str:
    p = T.trailing_punct(token)
    if not p:
        return "O"
    ch = p[-1]
    if ch in ".!":
        return "."
    if ch == "?":
        return "?"
    return ","


def punct_pairs(ref_surface: list[str], hyp_surface: list[str]) -> list[tuple[str, str]]:
    from rapidfuzz.distance import Levenshtein
    rn = [T.norm(t) for t in ref_surface]
    hn = [T.norm(t) for t in hyp_surface]
    pairs = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(rn, hn):
        if tag == "equal":
            pairs.extend(zip(range(i1, i2), range(j1, j2)))
    return [(ref_surface[i], hyp_surface[j]) for i, j in pairs]


def punct_scores(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        return {}
    y_true = [punct_class(r) for r, _ in pairs]
    y_pred = [punct_class(h) for _, h in pairs]
    out = {}
    tp: dict = defaultdict(int)
    fp: dict = defaultdict(int)
    fn: dict = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t == p:
            if t != "O":
                tp[t] += 1
        else:
            if p != "O":
                fp[p] += 1
            if t != "O":
                fn[t] += 1
    f1s = []
    for c in PUNCT_CLASSES:
        p, r, f = prf(tp[c], fp[c], fn[c], beta=1.0)
        if tp[c] + fn[c] == 0:      # class absent in the reference
            continue
        out[f"punct_{c}_p"], out[f"punct_{c}_r"], out[f"punct_{c}_f1"] = p, r, f
        f1s.append(f)
    out["punct_macro_f1"] = sum(f1s) / len(f1s) if f1s else float("nan")
    cased = [(T.core(r), T.core(h)) for r, h in pairs]
    out["casing_acc"] = sum(T.case_pattern(a) == T.case_pattern(b) for a, b in cased) / len(cased)
    return out

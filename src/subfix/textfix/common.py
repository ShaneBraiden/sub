"""Shared helpers: apply a text-level rewrite while keeping per-word provenance, and the
audio-consistency guard that stops text-only correctors (SymSpell, LanguageTool, GEC) from
overriding words the ASR heard with high confidence."""
from __future__ import annotations

from typing import Callable

from .. import text as T
from ..align import align
from ..subio import Word


def confirmed(w: Word, min_prob: float = 0.8) -> bool:
    """The current word text agrees with a confident ASR word at this position."""
    asr_text = w.meta.get("asr_text")
    if asr_text is None:
        return False
    p = w.meta.get("asr_prob", w.prob)
    return p is not None and p >= min_prob and T.canon(asr_text) == T.canon(w.text)


def protector(min_prob: float | None) -> Callable[[Word], bool] | None:
    if min_prob is None or min_prob > 1.0:
        return None
    return lambda w: confirmed(w, min_prob)


def retokenize(old: list[Word], new_tokens: list[str], stage: str, offset: int = 0,
               protect: Callable[[Word], bool] | None = None) -> tuple[list[Word], list[dict]]:
    """Map a rewritten token list back onto the old Words.

    Matched / substituted tokens inherit times + metadata from the word they replace, inserted
    tokens get no times (the timing stage interpolates them) and deletions are logged. With
    `protect`, substitutions/deletions of protected words and insertions between protected
    words are blocked (punctuation/casing-only changes are always allowed).
    """
    ops = align([w.text for w in old], new_tokens, gap=0.8, homophone_cost=0.3, near_miss_max=0.5,
                use_anchors=len(old) * len(new_tokens) > 40_000)
    is_prot = [bool(protect(w)) if protect else False for w in old]
    out: list[Word] = []
    edits: list[dict] = []
    next_old = 0
    for op, i, j in ops:
        if i is not None:
            next_old = i + 1
        if op in ("match", "sub"):
            w = old[i]
            tok = new_tokens[j]
            if tok == w.text:
                out.append(w)
                continue
            if op == "sub" and is_prot[i]:
                out.append(w)
                edits.append({"stage": stage, "op": "blocked", "old": w.text, "new": tok, "index": offset + len(out) - 1})
                continue
            nw = w.copy(text=tok)
            nw.meta = dict(w.meta)
            kind = "format" if op == "match" else "sub"
            nw.meta.setdefault("history", []).append({"stage": stage, "from": w.text})
            if kind == "sub":
                nw.source = stage
                nw.meta["stage"] = stage
            out.append(nw)
            edits.append({"stage": stage, "op": kind, "old": w.text, "new": tok, "index": offset + len(out) - 1})
        elif op == "ins":
            prev_p = is_prot[next_old - 1] if next_old - 1 >= 0 else True
            next_p = is_prot[next_old] if next_old < len(old) else True
            if protect and prev_p and next_p:
                edits.append({"stage": stage, "op": "blocked", "old": None, "new": new_tokens[j],
                              "index": offset + len(out)})
                continue
            ref = out[-1] if out else (old[0] if old else None)
            meta = {"stage": stage, "op": "ins"}
            if ref is not None and "cue" in ref.meta:
                meta["cue"] = ref.meta["cue"]
            out.append(Word(new_tokens[j], None, None, 1.0, stage, meta))
            edits.append({"stage": stage, "op": "ins", "old": None, "new": new_tokens[j], "index": offset + len(out) - 1})
        else:  # del
            if is_prot[i]:
                out.append(old[i])
                edits.append({"stage": stage, "op": "blocked", "old": old[i].text, "new": None,
                              "index": offset + len(out) - 1})
                continue
            edits.append({"stage": stage, "op": "del", "old": old[i].text, "new": None, "index": offset + len(out)})
    return out, edits

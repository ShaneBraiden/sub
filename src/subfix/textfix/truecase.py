"""Truecasing: casing transfer + sentence-start / "I" rules."""
from __future__ import annotations

from .. import text as T
from ..subio import Word

_I_FORMS = {"i", "i'm", "i've", "i'll", "i'd"}


def truecase(words: list[Word], caseless_source: bool, spell=None,
             common_threshold: int = 50_000_000) -> tuple[list[Word], list[dict]]:
    """Casing source priority: subtitle (if it has real casing) > aligned ASR casing (`case_hint`)
    > lowercase; then capitalize sentence starts and the pronoun "I". A capitalized common word
    that is no longer sentence-initial (its sentence boundary was removed) is lowercased."""
    out, edits = [], []
    sent_start = True
    for k, w in enumerate(words):
        if w.meta.get("event"):
            out.append(w)
            continue
        pre, c, suf = T.split_affixes(w.text)
        new_c = c
        if caseless_source:
            hint = w.meta.get("case_hint")
            if hint and T.core(hint).lower() == c.lower():
                new_c = T.core(hint)
            else:
                new_c = c.lower()
        low = new_c.lower()
        if low in _I_FORMS:
            new_c = "I" + new_c[1:]
        elif (not sent_start and new_c[:1].isupper() and new_c[1:].islower()
              and w.meta.get("was_sent_start") and spell is not None
              and spell.frequency(low) >= common_threshold):
            new_c = low
        if sent_start and new_c:
            new_c = new_c[0].upper() + new_c[1:]
        new = pre + new_c + suf
        if new != w.text:
            nw = w.copy(text=new)
            nw.meta.setdefault("history", []).append({"stage": "truecase", "from": w.text})
            out.append(nw)
            edits.append({"stage": "truecase", "op": "format", "old": w.text, "new": new, "index": k})
        else:
            out.append(w)
        sent_start = T.ends_sentence(new)
    return out, edits


def mark_sentence_starts(words: list[Word]) -> None:
    """Remember which words started a sentence before punctuation was re-predicted."""
    start = True
    for w in words:
        if start:
            w.meta["was_sent_start"] = True
        start = T.ends_sentence(w.text)

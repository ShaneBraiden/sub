"""Token-level text utilities shared by alignment, fusion, correction and metrics.

Conventions
-----------
* A *token* is a whitespace-separated word that keeps its attached punctuation ("Hello,").
* ``norm(token)``  -> lowercase, punctuation stripped, internal apostrophes kept ("don't").
* ``canon(token)`` -> ``norm`` with numbers spelled out and hyphens removed, so that
  "25" == "twenty-five" and "e-mail" == "email" when matching.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .paths import LEXICON_DIR

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-", "…": "..."})
_NON_WORD = re.compile(r"[^\w']+")
_AFFIX = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.S)
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)*$")
_ORDINAL = re.compile(r"^(\d+)(st|nd|rd|th)$")
_EVENT = re.compile(r"^(\[.*\]|\(.*\)|♪.*|#.*)$")
SENT_END = (".", "?", "!", "...")
PUNCT_MARKS = ".,?!:;"


def canon_quotes(s: str) -> str:
    return s.translate(_QUOTES)


def norm(token: str) -> str:
    t = canon_quotes(token).lower()
    t = _NON_WORD.sub("", t).replace("_", "")
    return t.strip("'")


def split_affixes(token: str) -> tuple[str, str, str]:
    """'"Hello,"' -> ('"', 'Hello', ',"').  Apostrophes inside the core are kept."""
    m = _AFFIX.match(canon_quotes(token))
    if not m:
        return "", token, ""
    pre, core, suf = m.groups()
    # keep a trailing apostrophe of plural possessives ("players'") inside the core
    if suf.startswith("'") and core and core[-1].lower() == "s":
        core, suf = core + "'", suf[1:]
    return pre, core, suf


def core(token: str) -> str:
    return split_affixes(token)[1]


def trailing_punct(token: str) -> str:
    """The punctuation marks at the end of a token (quotes/brackets removed)."""
    suf = split_affixes(token)[2]
    return "".join(ch for ch in suf if ch in PUNCT_MARKS or ch == ".")


def strip_trailing_punct(token: str) -> str:
    pre, c, suf = split_affixes(token)
    keep = "".join(ch for ch in suf if ch not in PUNCT_MARKS)
    return pre + c + keep


def ends_sentence(token: str) -> bool:
    p = trailing_punct(token)
    return any(p.endswith(m) for m in SENT_END)


def is_event(token: str) -> bool:
    """Sound-event annotations such as [MUSIC], (laughs), ♪ ... ♪."""
    return bool(_EVENT.match(token.strip())) and len(token.strip()) > 1


@lru_cache(maxsize=100_000)
def number_words(token: str) -> list[str] | None:
    """Spoken form of a numeric token, or None if it is not a number."""
    t = norm(token)
    if not t:
        return None
    try:
        from num2words import num2words
    except ImportError:  # pragma: no cover - optional dependency
        return None
    m = _ORDINAL.match(t)
    try:
        if m:
            words = num2words(int(m.group(1)), to="ordinal")
        elif _NUMBER.match(t):
            raw = t.replace(",", "")
            if "." in raw:
                words = num2words(float(raw))
            else:
                n = int(raw)
                words = num2words(n, to="year") if 1100 <= n <= 2099 and len(raw) == 4 else num2words(n)
        else:
            return None
    except Exception:
        return None
    return [w for w in re.split(r"[\s\-,]+", words.lower()) if w]


@lru_cache(maxsize=200_000)
def canon(token: str) -> str:
    n = norm(token)
    if not n:
        return ""
    nw = number_words(n)
    if nw:
        return "".join(nw)
    return n.replace("-", "").replace("'", "")


def spoken_form(token: str) -> str:
    """Letters-only form used by acoustic aligners (one string per token, no spaces)."""
    n = norm(token)
    nw = number_words(n) if n else None
    if nw:
        n = "".join(nw)
    n = n.replace("&", "and")
    n = re.sub(r"[^a-z']", "", n)
    return n.strip("'")


def case_pattern(word_core: str) -> str:
    letters = [c for c in word_core if c.isalpha()]
    if not letters:
        return "none"
    if all(c.islower() for c in letters):
        return "lower"
    if all(c.isupper() for c in letters):
        return "upper" if len(letters) > 1 else "title"
    if letters[0].isupper() and all(c.islower() for c in letters[1:]):
        return "title"
    return "mixed"


def apply_case(pattern: str, word_core: str) -> str:
    if pattern == "lower":
        return word_core.lower()
    if pattern == "upper":
        return word_core.upper()
    if pattern == "title":
        return word_core[:1].upper() + word_core[1:].lower() if word_core else word_core
    return word_core


def capitalize_first(token: str) -> str:
    pre, c, suf = split_affixes(token)
    if not c:
        return token
    return pre + c[0].upper() + c[1:] + suf


def detokenize(tokens: list[str]) -> str:
    return " ".join(t for t in tokens if t)


# ---------------------------------------------------------------- lexicons
def _read_lexicon(name: str) -> list[str]:
    path = LEXICON_DIR / name
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


@lru_cache(maxsize=1)
def fillers() -> tuple[set[str], set[tuple[str, ...]]]:
    single, phrases = set(), set()
    for entry in _read_lexicon("fillers.txt"):
        parts = tuple(norm(p) for p in entry.split())
        if len(parts) == 1:
            single.add(parts[0])
        else:
            phrases.add(parts)
    return single, phrases


def is_filler(token: str) -> bool:
    return norm(token) in fillers()[0]


@lru_cache(maxsize=1)
def function_words() -> frozenset[str]:
    return frozenset(norm(w) for w in _read_lexicon("function_words.txt"))


@lru_cache(maxsize=1)
def homophone_groups() -> dict[str, tuple[str, ...]]:
    table: dict[str, tuple[str, ...]] = {}
    for line in _read_lexicon("homophones.txt"):
        group = tuple(dict.fromkeys(w.lower() for w in line.split()))
        for w in group:
            others = tuple(o for o in group if o != w)
            table[w] = tuple(dict.fromkeys(table.get(w, ()) + others))
    return table

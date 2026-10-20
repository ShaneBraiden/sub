"""Spelling (SymSpell) + rule-based grammar (LanguageTool) — baseline B1 and chain stage I2."""
from __future__ import annotations

import logging
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

from .. import text as T
from ..subio import Word
from .common import retokenize

log = logging.getLogger(__name__)

CONTRACTIONS = {
    "i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll", "you'd", "he's", "he'll", "he'd",
    "she's", "she'll", "she'd", "it's", "it'll", "it'd", "we're", "we've", "we'll", "we'd",
    "they're", "they've", "they'll", "they'd", "that's", "that'll", "that'd", "there's", "there'll",
    "here's", "what's", "what're", "who's", "where's", "when's", "why's", "how's", "let's",
    "don't", "doesn't", "didn't", "can't", "couldn't", "won't", "wouldn't", "shouldn't", "isn't",
    "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "mustn't", "mightn't", "needn't",
    "ain't", "y'all", "o'clock", "ma'am", "gonna", "wanna", "gotta", "kinda", "sorta", "okay", "ok",
}


class SpellChecker:
    """SymSpell with the bundled 82,765-word unigram + 243,342 bigram dictionaries."""

    def __init__(self, max_edit_distance: int = 2, min_count: int = 100000, prefix_length: int = 7):
        from importlib.resources import files

        from symspellpy import SymSpell
        self.sym = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=prefix_length)
        base = files("symspellpy")
        self.sym.load_dictionary(str(base / "frequency_dictionary_en_82_765.txt"), term_index=0, count_index=1)
        try:
            self.sym.load_bigram_dictionary(str(base / "frequency_bigramdictionary_en_243_342.txt"),
                                            term_index=0, count_index=2)
        except Exception:  # pragma: no cover
            pass
        self.words = self.sym.words
        self.max_ed = max_edit_distance
        self.min_count = min_count

    def frequency(self, token: str) -> int:
        return int(self.words.get(T.norm(token), 0))

    def is_oov(self, token: str) -> bool:
        n = T.norm(token)
        if not n or any(ch.isdigit() for ch in n):
            return False
        if n in self.words or n in CONTRACTIONS:
            return False
        if "'" in n:
            base = n.split("'")[0]
            return not (base in self.words or base in CONTRACTIONS)
        if "-" in T.core(token):
            parts = [p for p in re.split(r"-", T.core(token).lower()) if p]
            return not all(p in self.words for p in parts)
        return True

    def suggest(self, token: str) -> str | None:
        from symspellpy import Verbosity
        pre, c, suf = T.split_affixes(token)
        low = c.lower()
        if len(low) < 3 or not low.isalpha() or not self.is_oov(c):
            return None
        max_ed = 1 if len(low) <= 4 else self.max_ed
        sugg = self.sym.lookup(low, Verbosity.TOP, max_edit_distance=max_ed, include_unknown=False)
        if not sugg or sugg[0].count < self.min_count or sugg[0].term == low:
            return None
        return pre + T.apply_case(T.case_pattern(c), sugg[0].term) + suf

    def correct_words(self, words: list[Word], skip_capitalized: bool = True,
                      protect=None) -> tuple[list[Word], list[dict]]:
        """Correct OOV words; words confirmed by the audio (`protect`) are never changed."""
        out, edits = [], []
        sent_start = True
        for w in words:
            new = None
            if not w.meta.get("event") and not (protect and protect(w)):
                c = T.core(w.text)
                capitalized_mid = skip_capitalized and not sent_start and c[:1].isupper()
                if not capitalized_mid:
                    new = self.suggest(w.text)
            if new and new != w.text:
                nw = w.copy(text=new, source="spell")
                nw.meta.setdefault("history", []).append({"stage": "symspell", "from": w.text})
                nw.meta["stage"] = "symspell"
                out.append(nw)
                edits.append({"stage": "symspell", "op": "sub", "old": w.text, "new": new, "index": len(out) - 1})
            else:
                out.append(w)
            sent_start = T.ends_sentence(w.text)
        return out, edits


# ------------------------------------------------------------------ LanguageTool
def _java_major(exe: Path) -> int:
    try:
        r = subprocess.run([str(exe), "-version"], capture_output=True, text=True, timeout=30)
    except Exception:
        return 0
    m = re.search(r'version "(\d+)(?:\.(\d+))?', r.stderr + r.stdout)
    if not m:
        return 0
    major = int(m.group(1))
    return int(m.group(2) or 0) if major == 1 else major


@lru_cache(maxsize=1)
def ensure_java(min_version: int = 17) -> str | None:
    """Find a Java >= 17 runtime (LanguageTool >= 6.6) and put it first on PATH for this process."""
    homes: list[Path] = []
    if os.environ.get("JAVA_HOME"):
        homes.append(Path(os.environ["JAVA_HOME"]))
    for base in [r"C:\Program Files\Eclipse Adoptium", r"C:\Program Files\Java", r"C:\Program Files\Microsoft",
                 r"C:\Program Files\Zulu", r"C:\Program Files\Amazon Corretto", "/usr/lib/jvm"]:
        b = Path(base)
        if b.exists():
            homes += sorted([p for p in b.iterdir() if p.is_dir()], reverse=True)
    for home in homes:
        exe = home / "bin" / ("java.exe" if os.name == "nt" else "java")
        if exe.exists() and _java_major(exe) >= min_version:
            os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
            os.environ["JAVA_HOME"] = str(home)
            return str(exe)
    return None


def _attr(m, *names, default=None):
    for n in names:
        if hasattr(m, n):
            return getattr(m, n)
    return default


class LanguageToolChecker:
    def __init__(self, language: str = "en-US", categories: list[str] | None = None,
                 extra_rules: list[str] | None = None):
        if not ensure_java():
            raise RuntimeError("LanguageTool needs Java >= 17 (set JAVA_HOME)")
        import language_tool_python
        self.tool = language_tool_python.LanguageTool(language)
        self.categories = set(categories or ["TYPOS", "GRAMMAR", "CONFUSED_WORDS"])
        self.extra_rules = set(extra_rules or ["EN_A_VS_AN"])

    def check(self, text: str) -> list[dict]:
        out = []
        for m in self.tool.check(text):
            cat = _attr(m, "category", default="")
            reps = list(_attr(m, "replacements", default=[]) or [])
            rule = _attr(m, "ruleId", "rule_id", default="")
            if (cat not in self.categories and rule not in self.extra_rules) or not reps:
                continue
            out.append({"offset": int(_attr(m, "offset", default=0)),
                        "length": int(_attr(m, "errorLength", "error_length", default=0)),
                        "replacement": reps[0], "rule": _attr(m, "ruleId", "rule_id", default=""),
                        "category": cat, "message": _attr(m, "message", default="")})
        return out

    def correct(self, text: str) -> tuple[str, list[dict]]:
        matches = sorted(self.check(text), key=lambda d: d["offset"], reverse=True)
        applied, limit = [], len(text) + 1
        for m in matches:
            end = m["offset"] + m["length"]
            if end > limit:
                continue
            text = text[:m["offset"]] + m["replacement"] + text[end:]
            limit = m["offset"]
            applied.append(m)
        return text, applied[::-1]

    def correct_words(self, words: list[Word], max_words: int = 60, protect=None) -> tuple[list[Word], list[dict]]:
        """Apply LT per sentence-ish chunk, then re-map tokens onto Words."""
        out: list[Word] = []
        edits: list[dict] = []
        start = 0
        for k, w in enumerate(words):
            end_here = T.ends_sentence(w.text) or (k - start + 1) >= max_words or k == len(words) - 1
            if not end_here:
                continue
            chunk = words[start:k + 1]
            start = k + 1
            if protect is not None and all(protect(x) for x in chunk):
                out.extend(chunk)
                continue
            src = " ".join(x.text for x in chunk)
            fixed, applied = self.correct(src)
            if not applied or fixed == src:
                out.extend(chunk)
                continue
            new_words, ed = retokenize(chunk, fixed.split(), "languagetool", offset=len(out), protect=protect)
            for e, m in zip([e for e in ed if e["op"] != "blocked"], applied):
                e["rule"] = m["rule"]
            out.extend(new_words)
            edits.extend(ed)
        return out, edits

    def close(self) -> None:
        try:
            self.tool.close()
        except Exception:
            pass

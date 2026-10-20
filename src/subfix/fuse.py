"""Evidence fusion (Mode A): decide which subtitle words to change — imp.md 4.1 / 8.4.

After alignment every position is a match, substitution (subtitle s vs ASR a), deletion
(subtitle-only word) or insertion (ASR-only word). Decisions are deliberately conservative
because professional subtitles are often condensed rather than verbatim:

  sub : score(a over s) = w_conf*p_asr(a) + w_phon*phon_sim(s,a) + w_lm*[LM(a|ctx) - LM(s|ctx)]
                          + w_oov*[s OOV and a in-vocab] + w_lex*[a in domain lexicon];
        replace if score > tau_sub
  del : delete only duplicates ("the the"), or when the ASR is confident around the position
        (mean p > del_span_conf) AND the LM prefers the sentence without the word
  ins : insert non-filler ASR words with p > tau_ins that the LM supports
  match: keep the subtitle surface form (ASR timing + casing hint attached)

Merged/split spellings ("alot" vs "a lot", "25" vs "twenty five") are detected as units and only
changed when the subtitle side is out-of-vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import text as T
from .align import Op, phon_sim, sub_cost
from .subio import Word


@dataclass
class Candidate:
    kind: str                     # sub | ins | del | split | merge
    i: int | None                 # subtitle index
    j: int | None                 # ASR index
    i2: int | None = None         # second subtitle index (merge)
    j2: int | None = None         # second ASR index (split)
    pos: int = 0                  # subtitle insertion position (ins)
    feats: dict = field(default_factory=dict)
    score: float = 0.0
    accept: bool = False
    rule: str = ""


@dataclass
class FusionResult:
    words: list[Word]
    edits: list[dict]
    candidates: list[Candidate]
    units: list[tuple]


def transplant(s_text: str, a_text: str, spell=None) -> str:
    """ASR word core placed into the subtitle token's punctuation, with sensible casing."""
    pre, cs, suf = T.split_affixes(s_text)
    _, ca, _ = T.split_affixes(a_text)
    if not ca:
        return s_text
    ps, pa = T.case_pattern(cs), T.case_pattern(ca)
    if ps in ("title", "upper"):
        new = T.apply_case(ps, ca)
    elif pa in ("title", "upper", "mixed") and not _common(ca, spell):
        new = ca                      # keep ASR casing for proper nouns / acronyms
    else:
        new = ca.lower()
    return pre + new + suf


def _common(word_core: str, spell) -> bool:
    if spell is None:
        return word_core.lower() in T.function_words()
    return spell.frequency(word_core.lower()) >= 20_000_000


def _insert_form(a_text: str, spell=None) -> str:
    _, ca, _ = T.split_affixes(a_text)
    pa = T.case_pattern(ca)
    if pa in ("title", "upper", "mixed") and not _common(ca, spell):
        return ca
    return ca.lower() if ca.lower() != "i" else "I"


class Fuser:
    def __init__(self, cfg, lm=None, spell=None, lexicon=None):
        self.f = cfg.fusion if hasattr(cfg, "fusion") else cfg
        self.lm = lm
        self.has_lm = lm is not None
        self.spell = spell
        self.lexicon = {T.canon(x) for x in (lexicon or []) if T.canon(x)}

    # ------------------------------------------------------------ grouping
    @staticmethod
    def units(ops: list[Op], sub: list[Word], asr: list[Word]) -> list[tuple]:
        """Group raw ops into decision units: (kind, i, j, i2, j2)."""
        c = T.canon
        out = []
        k = 0
        while k < len(ops):
            op, i, j = ops[k]
            nxt = ops[k + 1] if k + 1 < len(ops) else None
            if nxt is not None:
                n_op, n_i, n_j = nxt
                if op == "sub" and n_op == "ins" and c(sub[i].text) == c(asr[j].text) + c(asr[n_j].text):
                    out.append(("split", i, j, None, n_j)); k += 2; continue
                if op == "ins" and n_op == "sub" and c(sub[n_i].text) == c(asr[j].text) + c(asr[n_j].text):
                    out.append(("split", n_i, j, None, n_j)); k += 2; continue
                if op == "sub" and n_op == "del" and c(sub[i].text) + c(sub[n_i].text) == c(asr[j].text):
                    out.append(("merge", i, j, n_i, None)); k += 2; continue
                if op == "del" and n_op == "sub" and c(sub[i].text) + c(sub[n_i].text) == c(asr[n_j].text):
                    out.append(("merge", i, n_j, n_i, None)); k += 2; continue
            out.append((op, i, j, None, None))
            k += 1
        return out

    # ------------------------------------------------------------ features
    def _oov(self, tok: str) -> bool:
        return bool(self.spell.is_oov(tok)) if self.spell is not None else False

    def featurize(self, units: list[tuple], sub: list[Word], asr: list[Word]) -> list[Candidate]:
        K = self.f.lm_context
        toks = [w.text for w in sub]
        aligned_p = np.full(len(sub), np.nan)
        for kind, i, j, _, _ in units:
            if kind in ("match", "sub") and i is not None and j is not None:
                aligned_p[i] = asr[j].prob
        single_fill, phrase_fill = T.fillers()
        cands: list[Candidate] = []
        queries: list[tuple[Candidate, str, str]] = []
        last_i = -1
        for kind, i, j, i2, j2 in units:
            if kind == "match":
                last_i = i
                continue
            if kind == "sub":
                s, a = sub[i], asr[j]
                left, right = toks[max(0, i - K):i], toks[i + 1:i + 1 + K]
                a_tok = transplant(s.text, a.text, self.spell)
                c = Candidate("sub", i, j, feats=dict(
                    p=a.prob, phon=phon_sim(s.text, a.text), cost=sub_cost(s.text, a.text),
                    oov_s=self._oov(s.text), oov_a=self._oov(a.text),
                    lex=T.canon(a.text) in self.lexicon, event=bool(s.meta.get("event")), lm=0.0))
                queries.append((c, " ".join(left + [a_tok] + right), " ".join(left + [s.text] + right)))
                last_i = i
            elif kind == "ins":
                a = asr[j]
                pos = last_i + 1
                left, right = toks[max(0, pos - K):pos], toks[pos:pos + K]
                n_a = T.norm(a.text)
                prev_n = T.norm(asr[j - 1].text) if j > 0 else ""
                next_n = T.norm(asr[j + 1].text) if j + 1 < len(asr) else ""
                filler = (n_a in single_fill or (prev_n, n_a) in phrase_fill or (n_a, next_n) in phrase_fill)
                c = Candidate("ins", None, j, pos=pos, feats=dict(
                    p=a.prob, filler=filler, func=n_a in T.function_words(),
                    lex=T.canon(a.text) in self.lexicon, lm=0.0))
                queries.append((c, " ".join(left + [_insert_form(a.text, self.spell)] + right),
                                " ".join(left + right)))
            elif kind == "del":
                s = sub[i]
                left, right = toks[max(0, i - K):i], toks[i + 1:i + 1 + K]
                cs = T.canon(s.text)
                dup = bool(cs) and ((i > 0 and T.canon(toks[i - 1]) == cs) or
                                    (i + 1 < len(toks) and T.canon(toks[i + 1]) == cs))
                window = aligned_p[max(0, i - 3):i + 4]
                span = float(np.nanmean(window)) if np.any(~np.isnan(window)) else 0.0
                c = Candidate("del", i, None, feats=dict(
                    dup=dup, span_conf=span, event=bool(s.meta.get("event")),
                    oov_s=self._oov(s.text), lm=0.0))
                queries.append((c, " ".join(left + right), " ".join(left + [s.text] + right)))
                last_i = i
            elif kind == "split":
                c = Candidate("split", i, j, j2=j2, feats=dict(
                    oov_s=self._oov(sub[i].text), p=min(asr[j].prob, asr[j2].prob)))
                last_i = i
            elif kind == "merge":
                c = Candidate("merge", i, j, i2=i2, feats=dict(
                    oov_s=self._oov(sub[i].text) or self._oov(sub[i2].text),
                    oov_a=self._oov(asr[j].text), p=asr[j].prob))
                last_i = i2
            else:  # pragma: no cover
                continue
            cands.append(c)
        if self.lm is not None and queries:
            texts = [q for _, x, y in queries for q in (x, y)]
            scores = self.lm.score(texts)
            for n, (c, _, _) in enumerate(queries):
                c.feats["lm"] = float(scores[2 * n] - scores[2 * n + 1])
        return cands

    # ------------------------------------------------------------ decisions
    def sub_score(self, ft: dict) -> float:
        f = self.f
        lm = float(np.clip(ft.get("lm", 0.0), -f.lm_clip, f.lm_clip))
        return (f.w_conf * ft["p"] + f.w_phon * ft["phon"] + f.w_lm * lm
                + f.w_oov * float(ft["oov_s"] and not ft["oov_a"]) + f.w_lex * float(ft["lex"]))

    def decide(self, c: Candidate) -> Candidate:
        f, ft = self.f, c.feats
        if c.kind == "sub":
            if ft["event"]:
                c.accept, c.rule = False, "event"
            else:
                c.score = self.sub_score(ft)
                c.accept, c.rule = c.score > f.tau_sub, "score"
        elif c.kind == "ins":
            c.score = ft["p"]
            if ft["filler"]:
                c.accept, c.rule = False, "filler"
            elif ft["p"] > f.tau_ins and (not self.has_lm or ft["lm"] > f.tau_lm_ins):
                c.accept, c.rule = True, "confident+lm"
            else:
                c.accept, c.rule = False, "weak"
        elif c.kind == "del":
            c.score = ft["span_conf"]
            if ft["event"]:
                c.accept, c.rule = False, "event"
            elif ft["dup"]:
                c.accept, c.rule = True, "duplicate"
            elif ft["span_conf"] > f.del_span_conf and (not self.has_lm or ft["lm"] > f.tau_lm_del):
                c.accept, c.rule = True, "confident+lm"
            else:
                c.accept, c.rule = False, "keep"
        elif c.kind == "split":
            c.accept, c.rule = bool(ft["oov_s"]), "split-oov" if ft["oov_s"] else "split-equivalent"
        elif c.kind == "merge":
            ok = bool(ft["oov_s"]) and not ft["oov_a"]
            c.accept, c.rule = ok, "merge-oov" if ok else "merge-equivalent"
        return c

    # ------------------------------------------------------------ main entry
    def fuse(self, sub: list[Word], asr: list[Word], ops: list[Op], caseless: bool = False) -> FusionResult:
        units = self.units(ops, sub, asr)
        cands = [self.decide(c) for c in self.featurize(units, sub, asr)]
        by_key = {(c.kind, c.i, c.j): c for c in cands}
        out: list[Word] = []
        edits: list[dict] = []

        def cue_of(i: int | None) -> int | None:
            if i is not None:
                return sub[i].meta.get("cue")
            return out[-1].meta.get("cue") if out else (sub[0].meta.get("cue") if sub else None)

        def from_sub(i: int, j: int | None, op: str) -> Word:
            s = sub[i]
            w = s.copy(source="sub")
            w.meta = {"orig_index": i, "cue": s.meta.get("cue"), "line": s.meta.get("line", 0), "op": op}
            if s.meta.get("event"):
                w.meta["event"] = True
            w.meta["sub_start"], w.meta["sub_end"] = s.start, s.end
            if j is not None:
                a = asr[j]
                w.start, w.end = a.start, a.end
                w.prob = a.prob
                w.meta.update(asr_index=j, asr_text=a.text, asr_prob=a.prob)
                if caseless and T.canon(a.text) == T.canon(s.text):
                    w.meta["case_hint"] = a.text
            return w

        def log(c: Candidate, old, new):
            edits.append({"stage": "fusion", "op": c.kind, "decision": "accept" if c.accept else "reject",
                          "rule": c.rule, "old": old, "new": new, "sub_index": c.i, "asr_index": c.j,
                          "prob": round(float(c.feats.get("p", 0.0)), 3), "score": round(float(c.score), 3),
                          "lm": round(float(c.feats.get("lm", 0.0)), 3),
                          "time": asr[c.j].start if c.j is not None else sub[c.i].start,
                          "index": len(out) - 1})

        for kind, i, j, i2, j2 in units:
            if kind == "match":
                out.append(from_sub(i, j, "match"))
                continue
            c = by_key.get((kind, i, j))
            if c is None:  # pragma: no cover
                continue
            if kind == "sub":
                w = from_sub(i, j, "sub")
                if c.accept:
                    w.text = transplant(sub[i].text, asr[j].text, self.spell)
                    w.source = "asr"
                    w.meta.update(op="sub", orig_text=sub[i].text, fusion_score=c.score, rule=c.rule)
                    w.meta.pop("case_hint", None)
                else:
                    w.meta.update(op="keep-sub", fusion_score=c.score)
                out.append(w)
                log(c, sub[i].text, w.text if c.accept else None)
            elif kind == "ins":
                if c.accept:
                    a = asr[j]
                    tok = _insert_form(a.text, self.spell)
                    if out:   # move sentence punctuation behind the inserted word if ASR continues
                        prev = out[-1]
                        prev_asr_end = T.ends_sentence(asr[j - 1].text) if j > 0 else True
                        mark = T.trailing_punct(prev.text)
                        if mark and not prev_asr_end and not prev.meta.get("event"):
                            prev.text = T.strip_trailing_punct(prev.text)
                            tok = tok + mark
                    w = Word(tok, a.start, a.end, a.prob, "asr",
                             {"op": "ins", "asr_index": j, "asr_text": a.text, "asr_prob": a.prob,
                              "cue": cue_of(None), "fusion_score": c.score, "rule": c.rule})
                    out.append(w)
                    log(c, None, tok)
                else:
                    log(c, None, None)
            elif kind == "del":
                if c.accept:
                    log(c, sub[i].text, None)
                else:
                    out.append(from_sub(i, None, "keep-del"))
                    log(c, sub[i].text, None)
            elif kind == "split":
                if c.accept:
                    pre, _, suf = T.split_affixes(sub[i].text)
                    a1, a2 = asr[j], asr[j2]
                    t1 = pre + _insert_form(a1.text, self.spell)
                    t2 = _insert_form(a2.text, self.spell) + suf
                    for tok, a, jj in ((t1, a1, j), (t2, a2, j2)):
                        out.append(Word(tok, a.start, a.end, a.prob, "asr",
                                        {"op": "split", "orig_index": i, "asr_index": jj, "cue": sub[i].meta.get("cue"),
                                         "orig_text": sub[i].text, "rule": c.rule}))
                    log(c, sub[i].text, f"{t1} {t2}")
                else:
                    w = from_sub(i, j, "keep-split")
                    w.end = asr[j2].end
                    out.append(w)
                    log(c, sub[i].text, None)
            elif kind == "merge":
                if c.accept:
                    pre, _, _ = T.split_affixes(sub[i].text)
                    _, _, suf = T.split_affixes(sub[i2].text)
                    a = asr[j]
                    tok = pre + _insert_form(a.text, self.spell) + suf
                    out.append(Word(tok, a.start, a.end, a.prob, "asr",
                                    {"op": "merge", "orig_index": i, "asr_index": j, "cue": sub[i].meta.get("cue"),
                                     "orig_text": f"{sub[i].text} {sub[i2].text}", "rule": c.rule}))
                    log(c, f"{sub[i].text} {sub[i2].text}", tok)
                else:
                    a = asr[j]
                    mid = 0.5 * (a.start + a.end)
                    w1, w2 = from_sub(i, None, "keep-merge"), from_sub(i2, None, "keep-merge")
                    w1.start, w1.end, w2.start, w2.end = a.start, mid, mid, a.end
                    out.extend([w1, w2])
                    log(c, f"{sub[i].text} {sub[i2].text}", None)
        return FusionResult(out, edits, cands, units)


def passthrough(sub: list[Word]) -> list[Word]:
    """No fusion: subtitle words as they are (baseline B1 / ablations)."""
    out = []
    for i, s in enumerate(sub):
        w = s.copy()
        w.meta = {"orig_index": i, "cue": s.meta.get("cue"), "line": s.meta.get("line", 0), "op": "match",
                  "sub_start": s.start, "sub_end": s.end}
        if s.meta.get("event"):
            w.meta["event"] = True
        out.append(w)
    return out

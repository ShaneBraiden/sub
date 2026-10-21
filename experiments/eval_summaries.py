"""Evaluate the summaries (src/subfix/summarize.py) on the AMI meeting corpus — README "Summaries".

Data (`python -I tools/fetch_ami.py`): 139 meetings with manual transcripts (timed words, speakers),
topic segments and human summaries: an abstract (~8 sentences, ~160 words) plus actions, decisions
and problems (~290 words in all), every summary sentence linked to the dialogue acts it summarises.
Splits: 95 train (they train the LSA embeddings), 20 dev (tuning), 20 test (reported), 4 others unused.

Granularities and their references:
* meeting - the whole meeting at word budgets 25-300 against the abstract (the length curve; 25 words
  is the headline, 150 the overview), and at 300 words against all four sections ("detailed")
* topic   - every annotated topic segment (the gold boundaries, so segmentation errors do not count)
  against the summary sentences whose linked dialogue acts lie mostly inside it (any section), at 50
  words (the chapter budget; those references average 46 words)
Systems: lead (first sentences), random sentences (mean of 5 seeds), TextRank, TF-IDF centroid, LSA
embeddings (ours), a greedy ROUGE oracle (the extractive upper bound), and the local LLM
(qwen3:4b-instruct through Ollama): hierarchical over our meeting-mode chapters (the shipped design)
and the whole transcript in one prompt. Every summary is cut to the budget (as in the DUC
evaluations); ROUGE-1/2/L/Lsum F1 with stemming (metrics/rouge.py); 95% bootstrap CIs over meetings;
paired bootstrap for differences.

  python experiments/eval_summaries.py --stage tune     # extractive settings on the dev meetings
  python experiments/eval_summaries.py --stage test     # every system on the test meetings
  python experiments/eval_summaries.py --stage test --no-llm
Writes results/summaries/{tuning.json, tuned.json, results.md, results.csv, per_meeting.csv,
examples.md, outputs_test.json} and results/plots/summary_lengths.png. The LLM replies are cached
(cache/llm/), so a rerun only scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

from metrics.rouge import rouge, tokenize, truncate_words  # noqa: E402
from subfix.config import Config  # noqa: E402
from subfix.subio import Word  # noqa: E402
from subfix.summarize import (LSA, Part, Scorer, Sentence, llm_combine, llm_direct, llm_headline,  # noqa: E402
                              llm_part, make_sentences, members, pick)

AMI = ROOT / "data" / "ami"
OUT = ROOT / "results" / "summaries"
PLOTS = ROOT / "results" / "plots"
BUDGETS = [25, 50, 100, 150, 200, 300]
HEADLINE, OVERVIEW, DETAILED, TOPIC = 25, 150, 300, 50
METRICS = ("rouge1", "rouge2", "rougeL", "rougeLsum")
SEEDS = 5
EXTRACTIVE = ("textrank", "tfidf", "embed")
SYSTEMS = ("lead", "random", "textrank", "tfidf", "embed", "llm", "llm_direct", "oracle")
LABEL = {"lead": "Lead (first sentences)", "random": "Random sentences", "textrank": "TextRank",
         "tfidf": "TF-IDF centroid + MMR", "embed": "LSA embeddings + MMR (ours)",
         "llm": "LLM, chapter by chapter (ours)", "llm_direct": "LLM, whole transcript at once",
         "oracle": "Extractive oracle (upper bound)"}
LEVELS = {"headline": ("meeting", HEADLINE), "overview": ("meeting", OVERVIEW), "detailed": ("meeting-all", DETAILED),
          "chapter": ("topic", TOPIC)}            # granularity -> (scored level, word budget)
DIRECT_BUDGETS = (OVERVIEW, DETAILED)
LLM_CTX = 16384                   # the whole-transcript prompts need more context than the default 8192
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


# ------------------------------------------------------------------ data
@dataclass
class Meeting:
    id: str
    split: str
    duration: float
    words: list[Word]
    sents: list[Sentence]
    topics: list[Part]                       # gold topics, [start, next start)
    abstract: list[str]
    sections: list[str]
    topic_refs: dict[int, list[str]]         # topic position -> its reference sentences
    gold_extract: list[tuple[float, float]]
    pos: list[str] | None = None
    cache: dict = field(default_factory=dict)


def manifest() -> list[dict]:
    return json.loads((AMI / "manifest.json").read_text(encoding="utf-8"))


def load(split: str, max_sentence_words: int = 40) -> list[Meeting]:
    out = []
    for m in manifest():
        if m["split"] != split or not m.get("summary", True):
            continue
        rec = json.loads((AMI / "meetings" / f"{m['id']}.json").read_text(encoding="utf-8"))
        path = AMI / "summaries" / f"{m['id']}.json"
        if not path.exists():
            continue
        summ = json.loads(path.read_text(encoding="utf-8"))
        words = [Word(t, s, e, meta={"speaker": spk}) for t, s, e, spk in rec["words"]]
        dur = float(rec["duration"])
        tops = rec["topics"]
        topics = [Part(k + 1, t["start"], tops[k + 1]["start"] if k + 1 < len(tops) else dur, t.get("label") or "")
                  for k, t in enumerate(tops)]
        refs: dict[int, list[str]] = {}
        for s in summ["sentences"]:
            if not s["evidence"] or not topics:
                continue
            ov = [sum(max(0.0, min(e, p.end) - max(a, p.start)) for a, e in s["evidence"]) for p in topics]
            k = int(np.argmax(ov))
            if ov[k] > 0:
                refs.setdefault(k, []).append(s["text"])
        out.append(Meeting(m["id"], split, dur, words, make_sentences(words, max_sentence_words), topics,
                           [s["text"] for s in summ["sentences"] if s["section"] == "abstract"],
                           [s["text"] for s in summ["sentences"]], refs,
                           [(a["start"], a["end"]) for a in summ["extractive"]]))
    if not out:
        raise SystemExit("no AMI summaries: run `python -I tools/fetch_ami.py` first")
    return out


def reference(m: Meeting, kind: str, topic: int | None = None) -> list[str]:
    return {"abstract": m.abstract, "sections": m.sections}.get(kind) or m.topic_refs.get(topic, [])


def scopes(m: Meeting) -> list[tuple[str, list[int]]]:
    """('all', every sentence) and ('t<k>', the sentences of topic k) for the topics with a reference."""
    out = [("all", list(range(len(m.sents))))]
    for k, p in enumerate(m.topics):
        if k in m.topic_refs:
            idx = members(p, m.sents, k == len(m.topics) - 1)
            if idx:
                out.append((f"t{k}", idx))
    return out


# ------------------------------------------------------------------ systems
@dataclass(frozen=True)
class Params:
    min_words: int = 6
    mmr_lambda: float = 0.7
    redundancy: float = 0.5


def prepared(m: Meeting, scorer: Scorer, key: str) -> dict[str, tuple[list[int], object, np.ndarray]]:
    """Per scope: (sentence indices, their vectors, their relevance) - computed once per scorer."""
    if key not in m.cache:
        V = scorer.vectors(m.sents)
        prep = {}
        for name, idx in scopes(m):
            Vs = V[idx]
            prep[name] = (idx, Vs, scorer.relevance([m.sents[i] for i in idx], Vs))
        m.cache[key] = prep
    return m.cache[key]


def extract(m: Meeting, prep: dict, scope: str, budget: int, p: Params, method: str) -> list[int]:
    idx, V, rel = prep[scope]
    lengths = [m.sents[i].n_words for i in idx]
    el = [n for n, i in enumerate(idx) if m.sents[i].n_words >= p.min_words and len(m.sents[i].terms) >= 2]
    lam = 1.0 if method == "textrank" else p.mmr_lambda
    return [idx[n] for n in pick(el or list(range(len(idx))), rel, V, lengths, budget, lam, p.redundancy,
                                 overfill=True)]


def lead(m: Meeting, idx: list[int], budget: int) -> list[int]:
    out, n = [], 0
    for i in idx:
        if n >= budget:
            break
        out.append(i)
        n += m.sents[i].n_words
    return out


def random_pick(m: Meeting, idx: list[int], budget: int, seed: int, min_words: int) -> list[int]:
    el = [i for i in idx if m.sents[i].n_words >= min_words] or list(idx)
    rng = random.Random(seed)
    rng.shuffle(el)
    out, n = [], 0
    for i in el:
        if n >= budget:
            break
        out.append(i)
        n += m.sents[i].n_words
    return sorted(out)


def oracle(m: Meeting, idx: list[int], budget: int, ref: list[str]) -> list[int]:
    """Greedy extractive oracle: keep adding the sentence that most raises (ROUGE-1 F1 + ROUGE-2 F1) / 2
    against the reference, until nothing helps or the budget is used."""
    from collections import Counter
    rt = [t for s in ref for t in tokenize(s)]
    r1, r2 = Counter(rt), Counter(zip(rt, rt[1:]))
    n1, n2 = max(sum(r1.values()), 1), max(sum(r2.values()), 1)
    grams = {}
    for i in idx:
        t = tokenize(m.sents[i].text)
        if t:
            grams[i] = (Counter(t), Counter(zip(t, t[1:])), len(t))
    sel1, sel2 = Counter(), Counter()
    hit1 = hit2 = c1 = c2 = 0
    chosen, words, best = [], 0, 0.0

    def f(h, c, n):
        p, r = h / max(c, 1), h / n
        return 2 * p * r / (p + r) if p + r else 0.0
    while words < budget:
        cand = None
        for i, (g1, g2, n) in grams.items():
            if i in chosen:
                continue
            h1 = hit1 + sum(min(r1[g], sel1[g] + k) - min(r1[g], sel1[g]) for g, k in g1.items() if g in r1)
            h2 = hit2 + sum(min(r2[g], sel2[g] + k) - min(r2[g], sel2[g]) for g, k in g2.items() if g in r2)
            value = (f(h1, c1 + n, n1) + f(h2, c2 + max(n - 1, 0), n2)) / 2
            if value > best + 1e-9 and (cand is None or value > cand[0]):
                cand = (value, i, h1, h2, g1, g2, n)
        if cand is None:
            break
        best, i, hit1, hit2, g1, g2, n = cand
        sel1.update(g1)
        sel2.update(g2)
        c1, c2 = c1 + n, c2 + max(n - 1, 0)
        chosen.append(i)
        words += m.sents[i].n_words
    return sorted(chosen)


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text.strip()) if s]


def score(sents: list[str], ref: list[str], budget: int, metrics: tuple[str, ...] = METRICS) -> dict[str, float]:
    sc = rouge(truncate_words(sents, budget), ref, types=metrics)
    return {k: sc[k].fmeasure for k in metrics}


def gold_hits(m: Meeting, picked: list[int]) -> float | None:
    """Share of picked sentences that overlap (by at least half their duration) a dialogue act the AMI
    annotators put in the meeting's extractive summary."""
    if not picked:
        return None
    hits = 0
    for i in picked:
        s = m.sents[i]
        dur = max(s.end - s.start, 1e-3)
        ov = sum(max(0.0, min(s.end, b) - max(s.start, a)) for a, b in m.gold_extract)
        hits += ov >= 0.5 * dur
    return hits / len(picked)


# ------------------------------------------------------------------ scoring loops
def jobs(m: Meeting) -> list[tuple[str, str, int, str, int | None]]:
    """(level, scope, budget, reference kind, topic) for every evaluation of one meeting."""
    out = [("meeting", "all", b, "abstract", None) for b in BUDGETS]
    out.append(("meeting-all", "all", DETAILED, "sections", None))
    for name, _ in scopes(m)[1:]:
        out.append(("topic", name, TOPIC, "topic", int(name[1:])))
    return out


def run_extractive(meetings: list[Meeting], method: str, p: Params, scorer: Scorer, key: str,
                   budgets_only: tuple[int, ...] | None = None, metrics: tuple[str, ...] = METRICS) -> list[dict]:
    rows = []
    for m in meetings:
        prep = prepared(m, scorer, key)
        for level, scope, budget, kind, topic in jobs(m):
            if budgets_only and (level != "meeting" or budget not in budgets_only) and level != "topic":
                continue
            picked = extract(m, prep, scope, budget, p, method)
            row = {"meeting": m.id, "system": method, "level": level, "budget": budget, "scope": scope,
                   "words": sum(m.sents[i].n_words for i in picked),
                   **score([m.sents[i].text for i in picked], reference(m, kind, topic), budget, metrics)}
            if level == "meeting" and budget == OVERVIEW:
                row["gold_hits"] = gold_hits(m, picked)
            row["picked"] = picked
            rows.append(row)
    return rows


def run_baselines(meetings: list[Meeting], p: Params) -> list[dict]:
    rows = []
    for m in meetings:
        sc = dict(scopes(m))
        for level, scope, budget, kind, topic in jobs(m):
            ref = reference(m, kind, topic)
            idx = sc[scope]
            for name in ("lead", "random", "oracle"):
                if name == "random":
                    runs = [random_pick(m, idx, budget, seed, p.min_words) for seed in range(SEEDS)]
                else:
                    runs = [lead(m, idx, budget) if name == "lead" else oracle(m, idx, budget, ref)]
                scores = [score([m.sents[i].text for i in r], ref, budget) for r in runs]
                row = {"meeting": m.id, "system": name, "level": level, "budget": budget, "scope": scope,
                       "words": float(np.mean([sum(m.sents[i].n_words for i in r) for r in runs])),
                       **{k: float(np.mean([s[k] for s in scores])) for k in METRICS}}
                if level == "meeting" and budget == OVERVIEW:
                    row["gold_hits"] = float(np.mean([gold_hits(m, r) or 0.0 for r in runs]))
                row["picked"] = runs[0]
                rows.append(row)
    return rows


def per_meeting(rows: list[dict], system: str, level: str, budget: int, metric: str) -> dict[str, float]:
    """Mean per meeting (topic-level rows are averaged within their meeting first)."""
    acc: dict[str, list[float]] = {}
    for r in rows:
        if r["system"] == system and r["level"] == level and r["budget"] == budget and r.get(metric) is not None:
            acc.setdefault(r["meeting"], []).append(r[metric])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def boot_ci(values: list[float], boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(a: dict[str, float], b: dict[str, float], boot: int = 5000, seed: int = 0) -> dict:
    keys = sorted(set(a) & set(b))
    d = np.asarray([a[k] - b[k] for k in keys])
    if not len(d):
        return {"n": 0}
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (boot, len(d)))].mean(axis=1)
    p = min(1.0, 2 * min(float((means <= 0).mean()), float((means >= 0).mean())))
    return {"n": len(d), "delta": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "p": p}


def objective(rows: list[dict]) -> float:
    """Tuning target: mean ROUGE-1/2/L F1 of the overview (150 words) and of the chapter level."""
    vals = []
    for level, budget in (("meeting", OVERVIEW), ("topic", TOPIC)):
        for metric in ("rouge1", "rouge2", "rougeL"):
            pm = per_meeting(rows, rows[0]["system"], level, budget, metric)
            vals.append(np.mean(list(pm.values())))
    return float(np.mean(vals))


# ------------------------------------------------------------------ tuning (dev)
GRID_LAMBDA = [0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
GRID_REDUNDANCY = [0.5, 1.0]
GRID_MIN_WORDS = [6, 8, 10, 12, 15, 20]
GRID_DIMS = [200, 300, 400]


def tune(cfg) -> dict:
    from train_summarizer import corpus, fit
    dev = load("dev", cfg.summary.max_sentence_words)
    t0 = time.time()
    docs, _ = corpus()
    lsas = {}
    for dims in GRID_DIMS:
        model = fit(docs, dims)
        lsas[dims] = LSA.from_arrays(model["terms"], model["idf"], model["components"])
    print(f"LSA models fitted in {time.time() - t0:.0f} s")
    results: dict[str, list] = {}
    for method in EXTRACTIVE:
        lams = GRID_LAMBDA if method != "textrank" else [1.0]
        dims_grid = GRID_DIMS if method == "embed" else [200]
        for dims, lam, red, mw in product(dims_grid, lams, GRID_REDUNDANCY, GRID_MIN_WORDS):
            scorer = Scorer(method, lsas[dims], cfg.summary.damping)
            p = Params(mw, lam, red)
            rows = run_extractive(dev, method, p, scorer, f"{method}-{dims}", budgets_only=(OVERVIEW,),
                                  metrics=("rouge1", "rouge2", "rougeL"))
            results.setdefault(method, []).append({"dims": dims, "mmr_lambda": lam, "redundancy": red,
                                                   "min_words": mw, "objective": objective(rows)})
        results[method].sort(key=lambda r: -r["objective"])
        best = results[method][0]
        print(f"{method:9s} best {best}  ({len(results[method])} settings)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tuning.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    tuned = {m: {k: v for k, v in results[m][0].items() if k != "objective"} | {"dev_objective": results[m][0]["objective"]}
             for m in results}
    (OUT / "tuned.json").write_text(json.dumps(tuned, indent=1), encoding="utf-8")
    return tuned


# ------------------------------------------------------------------ LLM
def predicted_chapters(m: Meeting, cfg, models) -> list[Part]:
    from subfix.chapters import make_chapters
    from subfix.segment import pos_tags
    if m.pos is None:
        m.pos = pos_tags(m.words, models.pos_tagger())
    c = cfg.override({"chapters.mode": "meeting"})
    chs = make_chapters(m.words, c, m.pos, models.background(), None, m.duration)
    return [Part(ch.index, ch.start, ch.end, ch.title) for ch in chs]


def run_llm(meetings: list[Meeting], cfg, models, client, direct_client) -> tuple[list[dict], dict]:
    s = cfg.summary
    rows, outputs = [], {}
    for n, m in enumerate(meetings, 1):
        t0 = time.time()
        parts = predicted_chapters(m, cfg, models)
        summaries = []
        for k, p in enumerate(parts):
            idx = members(p, m.sents, k == len(parts) - 1)
            if idx:
                text = llm_part(client, m.sents, idx, "meeting", s.chapter_words, f"part {k + 1} of {len(parts)}",
                                cfg.llm.chunk_words)
                summaries.append((p.start, p.end, text))
        out = {"chapters": [{"start": a, "end": b, "text": t} for a, b, t in summaries], "overview": {},
               "topics": {}, "direct": {}}
        for b in BUDGETS:
            if b == HEADLINE:
                continue
            out["overview"][b] = llm_combine(client, summaries, "meeting", b) if len(summaries) > 1 else \
                llm_part(client, m.sents, list(range(len(m.sents))), "meeting", b, "", cfg.llm.chunk_words)
        out["overview"][HEADLINE] = llm_headline(client, out["overview"][OVERVIEW], "meeting", HEADLINE)
        sc = dict(scopes(m))
        for name, idx in sc.items():
            if name == "all":
                continue
            k = int(name[1:])
            out["topics"][name] = llm_part(client, m.sents, idx, "meeting", TOPIC, f"part {k + 1} of {len(m.topics)}",
                                           cfg.llm.chunk_words)
        for b in DIRECT_BUDGETS:
            out["direct"][b] = llm_direct(direct_client, m.sents, "meeting", b)
        outputs[m.id] = out
        for level, scope, budget, kind, topic in jobs(m):
            ref = reference(m, kind, topic)
            text = out["topics"][scope] if level == "topic" else out["overview"][budget]
            rows.append({"meeting": m.id, "system": "llm", "level": level, "budget": budget, "scope": scope,
                         "words": len(text.split()), **score(split_sentences(text), ref, budget)})
            if level != "topic" and budget in DIRECT_BUDGETS and (level == "meeting-all") == (kind == "sections"):
                dt = out["direct"][budget]
                rows.append({"meeting": m.id, "system": "llm_direct", "level": level, "budget": budget,
                             "scope": scope, "words": len(dt.split()), **score(split_sentences(dt), ref, budget)})
        print(f"  [{n}/{len(meetings)}] {m.id}: {len(parts)} chapters, {time.time() - t0:.0f} s "
              f"(LLM calls {client.stats.calls + direct_client.stats.calls}, cached {client.stats.cached + direct_client.stats.cached})",
              flush=True)
    return rows, outputs


# ------------------------------------------------------------------ test stage
def run_test(cfg, use_llm: bool) -> None:
    from subfix.models import Models
    tuned_path = OUT / "tuned.json"
    tuned = json.loads(tuned_path.read_text(encoding="utf-8")) if tuned_path.exists() else {}
    test = load("test", cfg.summary.max_sentence_words)
    lsa = LSA(ROOT / cfg.summary.lsa_model)
    if "embed" in tuned and tuned["embed"].get("dims") not in (None, lsa.dims):
        print(f"warning: tuned dims {tuned['embed']['dims']} != {lsa.dims} of {cfg.summary.lsa_model}: "
              f"python experiments/train_summarizer.py --dims {tuned['embed']['dims']}")
    rows: list[dict] = []
    params = {}
    for method in EXTRACTIVE:
        t = tuned.get(method, {})
        p = Params(t.get("min_words", cfg.summary.min_words), t.get("mmr_lambda", cfg.summary.mmr_lambda),
                   t.get("redundancy", cfg.summary.redundancy))
        params[method] = p
        rows += run_extractive(test, method, p, Scorer(method, lsa, cfg.summary.damping), method)
    rows += run_baselines(test, params["embed"])
    llm_info, outputs = None, {}
    if use_llm:
        from subfix.llm import Ollama
        client = Ollama.from_config(cfg.llm)
        st = client.status()
        if st["installed"]:
            direct = replace(Ollama.from_config(cfg.llm), num_ctx=LLM_CTX)
            models = Models(cfg)
            t0 = time.time()
            llm_rows, outputs = run_llm(test, cfg, models, client, direct)
            rows += llm_rows
            llm_info = {"model": cfg.llm.model, "seconds": round(time.time() - t0, 1),
                        "hierarchical": client.stats.to_json(), "direct": direct.stats.to_json()}
            client.unload()
        else:
            print(f"LLM skipped: {st['error']}")
    write_results(test, rows, params, tuned, llm_info, outputs)


# ------------------------------------------------------------------ reports
def table_rows(rows: list[dict], systems: list[str]) -> list[dict]:
    out = []
    for system in systems:
        for level, budget in [("meeting", b) for b in BUDGETS] + [("meeting-all", DETAILED), ("topic", TOPIC)]:
            entry = {"system": system, "level": level, "budget": budget}
            have = False
            for metric in METRICS + ("words", "gold_hits"):
                pm = per_meeting(rows, system, level, budget, metric)
                if not pm:
                    continue
                have = True
                mean, lo, hi = boot_ci(list(pm.values()))
                entry[metric] = mean
                entry[f"{metric}_lo"], entry[f"{metric}_hi"] = lo, hi
                entry["meetings"] = len(pm)
            if have:
                out.append(entry)
    return out


def write_results(test: list[Meeting], rows: list[dict], params: dict, tuned: dict, llm_info: dict | None,
                  outputs: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    systems = [s for s in SYSTEMS if any(r["system"] == s for r in rows)]
    table = table_rows(rows, systems)
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["system", "level", "budget", "meetings"] + [c for m in METRICS + ("words", "gold_hits")
                                                           for c in (m, f"{m}_lo", f"{m}_hi")]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in table:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()})
    with open(OUT / "per_meeting.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["meeting", "system", "level", "budget", "scope", "words", *METRICS, "gold_hits"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    comparisons = {}
    for a, b in (("embed", "textrank"), ("embed", "tfidf"), ("embed", "lead"), ("llm", "embed"),
                 ("llm", "llm_direct"), ("oracle", "embed")):
        if a in systems and b in systems:
            for name, level, budget in (("overview", "meeting", OVERVIEW), ("chapter", "topic", TOPIC),
                                        ("detailed", "meeting-all", DETAILED)):
                for metric in ("rouge1", "rouge2", "rougeL"):
                    d = paired(per_meeting(rows, a, level, budget, metric), per_meeting(rows, b, level, budget, metric))
                    if d.get("n"):
                        comparisons[f"{a} vs {b} | {name} | {metric}"] = d
    data = {"meetings": len(test), "systems": systems, "table": table, "comparisons": comparisons,
            "params": {k: vars(v) for k, v in params.items()}, "tuned": tuned, "llm": llm_info,
            "budgets": BUDGETS, "levels": LEVELS}
    (OUT / "results.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    save_outputs(test, rows, outputs)
    write_markdown(data)
    write_examples(test, rows, outputs)
    plot_lengths(table, systems)


def save_outputs(test: list[Meeting], rows: list[dict], outputs: dict) -> None:
    """Every system's text per meeting and topic (the human study samples from this file)."""
    data = {}
    for m in test:
        d = {"duration": m.duration, "abstract": m.abstract, "sections": m.sections, "meeting": {}, "topics": {}}
        for name, idx in scopes(m)[1:]:
            k = int(name[1:])
            d["topics"][name] = {"index": k + 1, "start": m.topics[k].start, "end": m.topics[k].end,
                                 "label": m.topics[k].title, "reference": m.topic_refs[k],
                                 "transcript": [[round(m.sents[i].start, 2), m.sents[i].speaker, m.sents[i].text]
                                                for i in idx], "systems": {}}
        data[m.id] = d
    for r in rows:
        if "picked" not in r:
            continue
        m = next(x for x in test if x.id == r["meeting"])
        text = [m.sents[i].text for i in r["picked"]]
        if r["level"] == "topic":
            data[m.id]["topics"][r["scope"]]["systems"][r["system"]] = text
        elif r["level"] == "meeting":
            data[m.id]["meeting"].setdefault(r["system"], {})[r["budget"]] = text
    for mid, out in outputs.items():
        for name, text in out["topics"].items():
            data[mid]["topics"][name]["systems"]["llm"] = split_sentences(text)
        data[mid]["meeting"]["llm"] = {b: split_sentences(t) for b, t in out["overview"].items()}
        data[mid]["meeting"]["llm_direct"] = {b: split_sentences(t) for b, t in out["direct"].items()}
        data[mid]["llm_chapters"] = out["chapters"]
    (OUT / "outputs_test.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _cell(e: dict, metric: str) -> str:
    if metric not in e:
        return "-"
    return f"{100 * e[metric]:.1f} ({100 * e[metric + '_lo']:.1f}-{100 * e[metric + '_hi']:.1f})"


def write_markdown(data: dict) -> None:
    table, systems = data["table"], data["systems"]

    def get(system, level, budget):
        return next((e for e in table if e["system"] == system and e["level"] == level and e["budget"] == budget), None)
    L = ["# Summaries on the AMI meeting corpus", "",
         f"{data['meetings']} unseen test meetings (official AMI split), manual transcripts. ROUGE F1 x 100 with "
         "stemming; mean over meetings with a 95% bootstrap CI. Every summary is cut to the word budget.", "",
         "## Overview (150 words) against the human abstract", "",
         "| System | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | words | picked sentences in the human extract |",
         "|---|---|---|---|---|---|---|"]
    for s in systems:
        e = get(s, "meeting", OVERVIEW)
        if e:
            hits = f"{100 * e['gold_hits']:.0f}%" if "gold_hits" in e else "-"
            L.append(f"| {LABEL[s]} | {_cell(e, 'rouge1')} | {_cell(e, 'rouge2')} | {_cell(e, 'rougeL')} | "
                     f"{_cell(e, 'rougeLsum')} | {e['words']:.0f} | {hits} |")
    L += ["", "Words = mean length before the cut (the LLM is asked for at most the budget; extractive systems "
          "fill it). Picked sentences in the human extract = share of the sentences a system picks that overlap a "
          "dialogue act the AMI annotators marked as summary-worthy.", "",
          "## Granularity: ROUGE-2 F1 at each level", "",
          "| System | headline (25 w, abstract) | overview (150 w, abstract) | detailed (300 w, all sections) | "
          "chapter (50 w, topic sentences) |", "|---|---|---|---|---|"]
    for s in systems:
        cells = []
        for name in ("headline", "overview", "detailed", "chapter"):
            e = get(s, *LEVELS[name])
            cells.append(_cell(e, "rouge2") if e else "-")
        L.append(f"| {LABEL[s]} | " + " | ".join(cells) + " |")
    L += ["", "ROUGE-1 F1 at the same levels:", "",
          "| System | headline | overview | detailed | chapter |", "|---|---|---|---|---|"]
    for s in systems:
        cells = []
        for name in ("headline", "overview", "detailed", "chapter"):
            e = get(s, *LEVELS[name])
            cells.append(f"{100 * e['rouge1']:.1f}" if e else "-")
        L.append(f"| {LABEL[s]} | " + " | ".join(cells) + " |")
    L += ["", "![ROUGE against summary length](../plots/summary_lengths.png)", "",
          "## Paired differences", "", "Mean difference over meetings (x 100) with a 95% paired-bootstrap CI and "
          "two-sided p.", "", "| Comparison | level | metric | difference | 95% CI | p |", "|---|---|---|---|---|---|"]
    for key, d in data["comparisons"].items():
        pair, level, metric = key.split(" | ")
        L.append(f"| {pair} | {level} | {metric} | {100 * d['delta']:+.2f} | {100 * d['lo']:+.2f} to "
                 f"{100 * d['hi']:+.2f} | {d['p']:.3f} |")
    L += ["", "## Settings", ""]
    for method, p in data["params"].items():
        t = data["tuned"].get(method, {})
        dims = f", {t['dims']} LSA dimensions" if method == "embed" and "dims" in t else ""
        L.append(f"- {LABEL[method]}: min {p['min_words']} words per sentence, MMR lambda {p['mmr_lambda']}, "
                 f"redundancy cut {p['redundancy']}{dims} (tuned on the 20 dev meetings: "
                 f"objective {t.get('dev_objective', float('nan')):.4f})")
    if data["llm"]:
        x = data["llm"]
        h, d = x["hierarchical"], x["direct"]
        L.append(f"- LLM: {x['model']} through Ollama, temperature 0, seed 0; chapter by chapter = our meeting-mode "
                 f"chapters summarised in 50 words each, then combined; {h['calls'] + d['calls']} new calls "
                 f"({h['cached'] + d['cached']} from the cache), {x['seconds'] / 60:.0f} min, "
                 f"{h['prompt_tokens'] + d['prompt_tokens']:,} prompt + {h['output_tokens'] + d['output_tokens']:,} "
                 "output tokens")
    (OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_examples(test: list[Meeting], rows: list[dict], outputs: dict) -> None:
    m = test[0]
    L = [f"# Example: meeting {m.id} (test)", "", "## Human abstract", "", " ".join(m.abstract), ""]
    for s in SYSTEMS:
        r = next((r for r in rows if r["meeting"] == m.id and r["system"] == s and r["level"] == "meeting"
                  and r["budget"] == OVERVIEW), None)
        if r is None:
            continue
        if "picked" in r:
            text = " ".join(m.sents[i].text for i in r["picked"])
        elif s == "llm" and m.id in outputs:
            text = outputs[m.id]["overview"][OVERVIEW]
        elif s == "llm_direct" and m.id in outputs:
            text = outputs[m.id]["direct"][OVERVIEW]
        else:
            continue
        L += [f"## {LABEL[s]} (ROUGE-1 {100 * r['rouge1']:.1f}, ROUGE-2 {100 * r['rouge2']:.1f})", "",
              " ".join(truncate_words(text, OVERVIEW)), ""]
    if m.id in outputs:
        L += ["## LLM headline (25 words)", "", outputs[m.id]["overview"][HEADLINE], ""]
    (OUT / "examples.md").write_text("\n".join(L), encoding="utf-8")


def plot_lengths(table: list[dict], systems: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vizstyle import BASELINE, CATEGORICAL, INK, INK_2, MUTED, SURFACE, style_axes
    ours = [s for s in ("textrank", "embed", "llm", "llm_direct") if s in systems]
    colors = {s: CATEGORICAL[k] for k, s in enumerate(ours)}          # fixed slot order, never cycled
    context = {"lead": (BASELINE, "s"), "random": (BASELINE, "^"), "oracle": (MUTED, "o")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), dpi=150, facecolor=SURFACE)
    for ax, metric, title in zip(axes, ("rouge1", "rouge2"), ("ROUGE-1 F1", "ROUGE-2 F1")):
        for s in [s for s in context if s in systems] + ours:
            pts = sorted((e["budget"], e[metric]) for e in table if e["system"] == s and e["level"] == "meeting")
            if not pts:
                continue
            x, y = zip(*pts)
            y = [100 * v for v in y]
            if s in colors:
                ax.plot(x, y, color=colors[s], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, solid_capstyle="round",
                        label=LABEL[s], zorder=3)
            else:
                color, marker = context[s]
                ax.plot(x, y, color=color, lw=1.5, marker=marker, ms=4, label=LABEL[s], zorder=2)
        ax.set_xticks(BUDGETS)
        ax.set_xlim(15, 315)
        ax.set_xlabel("summary length (words)")
        ax.set_title(f"{title} against the human abstract", fontsize=10, color=INK, loc="left")
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5, frameon=False, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "summary_lengths.png", facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["tune", "test", "all"], default="all")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM systems")
    args = ap.parse_args()
    cfg = Config.load()
    if args.stage in ("tune", "all"):
        tune(cfg)
    if args.stage in ("test", "all"):
        run_test(cfg, not args.no_llm)
        print(f"wrote {OUT / 'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

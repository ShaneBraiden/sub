"""Evaluate the chapter markers (topic segmentation, src/subfix/chapters.py) — README "Chapter markers".

No benchmark recording has chapter annotations, so test documents are built the standard way for
topic segmentation (Choi 2000): excerpts of different recordings joined end to end, so every
junction is a known topic change. Excerpts (90-300 s, cut at sentence boundaries) come from the
benchmark clips (LibriSpeech chapters and TED talks, each a different book or talk); the pause at a
junction is drawn from the pauses between sentences inside the recordings, so pause length gives
nothing away. Dev clips build the tuning documents, test clips the reported ones.

Words come from the pipeline outputs cached by experiments/run_eval.py (Mode B: audio only,
Mode A: corrected subtitles) or from the reference transcripts (MFA timings; the TED references
have no punctuation, so their sentences are cut at pauses).

Real topic boundaries: the AMI meeting corpus topic annotations (`--stage ami`, after
`python -I tools/fetch_ami.py`): the shipped settings zero-shot on all 139 annotated meetings, a
penalty sweep, and settings tuned on the 20 AMI development meetings tested on the 20 unseen ones.

  python experiments/eval_chapters.py                     # tune on dev, test, ablations, examples (+ AMI)
  python experiments/eval_chapters.py --stage tune|test|examples|ami

Missing dev outputs:  python experiments/run_eval.py --manifest dev --systems modeb_ours ours
Writes results/chapters/{tuning.json, tuned.json, results.md, results.csv, per_doc.csv,
examples.md, ami.md, ami.json, ami_per_meeting.csv} and results/plots/chapters_example.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from metrics.segmentation import boundary_prf, f1, pk, windowdiff  # noqa: E402
from subfix.chapters import Background, Document, make_chapters, sentence_spans, to_youtube  # noqa: E402
from subfix.config import Config  # noqa: E402
from subfix.subio import Word  # noqa: E402

OUT = ROOT / "results" / "chapters"
SOURCES = {
    "modeb": ("experiments/outputs/modeb_ours/default/{id}.words.json", "Mode B output (audio only)"),
    "modea": ("experiments/outputs/ours/default/{id}.words.json", "Mode A output (corrected subtitles)"),
    "ref": ("data/refs/{id}.words.json", "Reference transcripts (MFA timings)"),
}
N_DOCS = {"dev": 80, "test": 200}
SEED = {"dev": 11, "test": 23}
PARTS = (3, 6)              # excerpts per document
EXCERPT_S = (90.0, 300.0)   # excerpt length
MIN_CHAPTER = 60.0          # every method (and baseline) keeps chapters >= this long
TOL = 10.0                  # boundary F1 tolerance (s)
TOL_WIDE = 30.0             # second tolerance, reported for the AMI meetings (topic shifts there are gradual)
CONTENT = {"N": ["NOUN", "PROPN"], "NA": ["NOUN", "PROPN", "ADJ"], "NVA": ["NOUN", "PROPN", "VERB", "ADJ"],
           "NVAR": ["NOUN", "PROPN", "VERB", "ADJ", "ADV"],
           "ALL": ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "PRON", "AUX", "ADP", "DET", "CCONJ", "SCONJ",
                   "PART", "INTJ", "X", "SYM", "PUNCT"]}     # ALL = no POS filter (closed-class words still dropped)
CONTENT_LABEL = {"N": "nouns", "NA": "nouns + adjectives", "NVA": "nouns + verbs + adjectives",
                 "NVAR": "nouns + verbs + adjectives + adverbs", "ALL": "every word (no POS filter)"}
GENERIC = [0.0, -7.0, -8.0, -9.0, -10.0]
DP_PENALTY = [0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0, 32.0]
TILING = list(product([100, 175, 250, 400, 600], [1, 2, 3, 4], [0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5]))


# ------------------------------------------------------------------ data
@dataclass
class Item:
    id: str
    group: str                      # libri | ted
    words: list[Word]
    pos: list[str]
    sents: list[tuple[int, int]]
    st: list[float]
    en: list[float]


@dataclass
class Doc:
    name: str
    words: list[Word]
    pos: list[str]
    sents: list[tuple[int, int]]
    duration: float
    ref: list[int]                  # word index of every true chapter start (first excluded)
    ref_times: list[float]
    groups: list[str]               # group of each excerpt
    parts: list[str] = field(default_factory=list)


def manifest(name: str) -> list[dict]:
    path = ROOT / "data" / "benchmark" / f"{name}.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_items(split: str, source: str, tagger) -> list[Item]:
    items = []
    for it in manifest(split):
        path = ROOT / SOURCES[source][0].format(id=it["id"])
        if not path.exists():
            raise SystemExit(f"{path} missing: python experiments/run_eval.py --manifest {split} "
                             "--systems modeb_ours ours")
        if it["duration"] < 60:
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        words = [Word(r["text"], r["start"], r["end"]) for r in rows if r["text"] and r["start"] is not None]
        st = [w.start for w in words]
        en = [max(w.end or w.start, w.start) for w in words]
        items.append(Item(it["id"], "ted" if it["dataset"].startswith("ted") else "libri", words,
                          tagger.tag_tokens([w.text for w in words]), sentence_spans(words, st, en), st, en))
    return items


def pause_pool(items: list[Item]) -> np.ndarray:
    gaps = [it.st[b] - it.en[b - 1] for it in items for b, _ in it.sents[1:]]
    return np.sort(np.clip(np.asarray(gaps), 0.0, 3.0))


def doc_specs(ids: list[str], n: int, seed: int) -> list[list[tuple]]:
    """Per document: [(item id, excerpt length, start quantile, junction-gap quantile), ...]."""
    rng = random.Random(seed)
    specs = []
    for _ in range(n):
        k = rng.randint(PARTS[0], min(PARTS[1], len(ids)))
        specs.append([(i, rng.uniform(*EXCERPT_S), rng.random(), rng.random()) for i in rng.sample(ids, k)])
    return specs


def build_doc(name: str, spec: list[tuple], items: dict[str, Item], pauses: np.ndarray) -> Doc:
    words, pos, sents, ref, ref_times, groups, parts = [], [], [], [], [], [], []
    end = 0.0
    for p, (iid, length, u, g) in enumerate(spec):
        it = items[iid]
        starts = [it.st[a] for a, _ in it.sents]
        total = it.en[-1] - it.st[0]
        s0 = 0
        if total > length:
            t0 = it.st[0] + u * (total - length)
            s0 = min(next((k for k, t in enumerate(starts) if t >= t0), 0), len(it.sents) - 1)
        s1 = s0 + 1
        while s1 < len(it.sents) and it.en[it.sents[s1 - 1][1] - 1] - starts[s0] < length:
            s1 += 1
        a, b = it.sents[s0][0], it.sents[s1 - 1][1]
        gap = 0.5 if p == 0 else float(pauses[min(int(g * len(pauses)), len(pauses) - 1)])
        shift = end + gap - it.st[a]
        base = len(words)
        if p:
            ref.append(base)
            ref_times.append(it.st[a] + shift)
        words += [Word(w.text, w.start + shift, (w.end if w.end is not None else w.start) + shift)
                  for w in it.words[a:b]]
        pos += it.pos[a:b]
        sents += [(x - a + base, y - a + base) for x, y in it.sents[s0:s1]]
        end = words[-1].end
        groups.append(it.group)
        parts.append(f"{iid} {starts[s0] - it.st[0]:.0f}-{it.en[b - 1] - it.st[0]:.0f}s")
    return Doc(name, words, pos, sents, end + 0.5, ref, ref_times, groups, parts)


def build_docs(split: str, source: str, tagger) -> list[Doc]:
    items = load_items(split, source, tagger)
    by_id = {it.id: it for it in items}
    pauses = pause_pool(items)
    return [build_doc(f"{split}-{k:03d}", spec, by_id, pauses)
            for k, spec in enumerate(doc_specs(sorted(by_id), N_DOCS[split], SEED[split]))]


# ------------------------------------------------------------------ methods
_BASE = None


def chapters_cfg(**kw):
    """configs/default.yaml's chapters section with these fields replaced (loaded once)."""
    global _BASE
    if _BASE is None:
        _BASE = replace(Config.load().chapters, min_duration=MIN_CHAPTER, num_chapters=0)
    return replace(_BASE, **kw)


def prepare(doc: Doc, bg: Background | None, content: str, generic: float, stem: bool = True) -> Document:
    c = chapters_cfg(content_pos=CONTENT[content], generic_logp=generic, stem=stem)
    return Document(doc.words, doc.pos, c, bg, doc.duration, sents=doc.sents)


def boundaries(D: Document, c) -> list[int]:
    """Predicted chapter starts as sentence indices (first excluded)."""
    return D.boundaries(c)[0][1:-1]


def run(doc: Doc, D: Document, c) -> dict:
    return score(doc, D, boundaries(D, c))


def score(doc: Doc, D: Document, sent_bounds: list[int]) -> dict:
    hyp = [D.sents[b][0] for b in sent_bounds]
    times = [D.T[b] for b in sent_bounds]
    n = len(doc.words)
    hit, nr, nh = boundary_prf(doc.ref_times, times, TOL)
    return {"pk": pk(doc.ref, hyp, n), "wd": windowdiff(doc.ref, hyp, n), "hit": hit, "n_ref": nr, "n_hyp": nh,
            "hit30": boundary_prf(doc.ref_times, times, TOL_WIDE)[0]}


def snap(D: Document, times: list[float]) -> list[int]:
    """Sentence starts nearest to the given times (distinct, increasing)."""
    T = np.asarray(D.T[1:-1])
    out = []
    for t in times:
        k = int(np.argmin(np.abs(T - t))) + 1
        if k not in out:
            out.append(k)
    return sorted(out)


def baselines(doc: Doc, D: Document, period: float, rng: random.Random) -> dict[str, list[int]]:
    K = len(doc.ref) + 1
    S = len(D.sents)
    out = {"single": [], "even_oracle_k": snap(D, [doc.duration * j / K for j in range(1, K)]),
           "periodic": snap(D, list(np.arange(period, doc.duration - MIN_CHAPTER, period)))}
    for r in range(5):
        out[f"random_oracle_k#{r}"] = sorted(rng.sample(range(1, S), min(K - 1, S - 1)))
    return out


# ------------------------------------------------------------------ aggregation
def summarize(rows: list[dict], boot: int = 0, seed: int = 0) -> dict:
    hit = sum(r["hit"] for r in rows)
    nr = sum(r["n_ref"] for r in rows)
    nh = sum(r["n_hyp"] for r in rows)
    p, r_, f = f1(hit, nr, nh)
    out = {"docs": len(rows), "pk": statistics.mean(r["pk"] for r in rows),
           "wd": statistics.mean(r["wd"] for r in rows), "p": p, "r": r_, "f1": f,
           "k_ref": statistics.mean(r["n_ref"] + 1 for r in rows),
           "k_hyp": statistics.mean(r["n_hyp"] + 1 for r in rows),
           "k_abs_err": statistics.mean(abs(r["n_hyp"] - r["n_ref"]) for r in rows)}
    out["p30"], out["r30"], out["f1_30"] = f1(sum(r["hit30"] for r in rows), nr, nh)
    if boot:
        rng = np.random.default_rng(seed)
        wd = np.asarray([r["wd"] for r in rows])
        means = [wd[rng.integers(0, len(wd), len(wd))].mean() for _ in range(boot)]
        out["wd_ci"] = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    return out


def mean_random(per_doc: list[dict[str, dict]]) -> list[dict]:
    """Average the 5 random draws of each document into one row."""
    rows = []
    for d in per_doc:
        draws = [v for k, v in d.items() if k.startswith("random_oracle_k#")]
        rows.append({k: statistics.mean(x[k] for x in draws) for k in ("pk", "wd", "hit", "n_ref", "n_hyp", "hit30")})
    return rows


# ------------------------------------------------------------------ tuning (dev)
_G: dict = {}


def _init(docs, bg):
    _G["docs"], _G["bg"] = docs, bg


def _tune_doc(k: int) -> dict:
    doc, bg = _G["docs"][k], _G["bg"]
    res = {}
    for content, generic in product(CONTENT, GENERIC):
        D = prepare(doc, bg, content, generic)
        for pen in DP_PENALTY:
            res[("dp", content, generic, pen)] = run(doc, D, chapters_cfg(method="dp", penalty=pen))
        for w, sm, cut in TILING:
            c = chapters_cfg(method="tiling", window_words=w, smoothing=sm, cutoff=cut)
            res[("tiling", content, generic, w, sm, cut)] = run(doc, D, c)
    return res


def tune(tagger, bg, workers: int) -> dict:
    t0 = time.time()
    docs = build_docs("dev", "modeb", tagger)
    print(f"dev: {len(docs)} documents, {statistics.mean(len(d.ref) + 1 for d in docs):.1f} chapters, "
          f"{statistics.mean(d.duration for d in docs) / 60:.1f} min on average", flush=True)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(docs, bg)) as ex:
        per_doc = list(ex.map(_tune_doc, range(len(docs)), chunksize=2))
    grid = {}
    for key in per_doc[0]:
        grid[key] = summarize([d[key] for d in per_doc])
    best = {}
    for method in ("dp", "tiling"):
        keys = [k for k in grid if k[0] == method]
        kbest = min(keys, key=lambda k: (grid[k]["wd"], grid[k]["pk"]))
        best[method] = {"key": list(kbest), **grid[kbest]}
        print(method, kbest, {m: round(grid[kbest][m], 3) for m in ("wd", "pk", "f1", "k_hyp", "k_ref")}, flush=True)
    period = statistics.mean(d.duration / (len(d.ref) + 1) for d in docs)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"key": list(k), **v} for k, v in sorted(grid.items(), key=lambda kv: kv[1]["wd"])]
    (OUT / "tuning.json").write_text(json.dumps({"docs": len(docs), "period": period, "best": best, "grid": rows},
                                                indent=1), encoding="utf-8")
    tuned = {"dp": best["dp"]["key"], "tiling": best["tiling"]["key"], "period": period,
             "method": min(best, key=lambda m: (best[m]["wd"], best[m]["pk"]))}
    (OUT / "tuned.json").write_text(json.dumps(tuned, indent=1), encoding="utf-8")
    print(f"tuning: {time.time() - t0:.0f} s; chosen method {tuned['method']}", flush=True)
    return tuned


def method_cfg(key: list) -> tuple[str, float, object]:
    """(content set, generic_logp, chapters config) of a tuned grid key."""
    if key[0] == "dp":
        _, content, generic, pen = key
        return content, generic, chapters_cfg(method="dp", penalty=pen)
    _, content, generic, w, sm, cut = key
    return content, generic, chapters_cfg(method="tiling", window_words=w, smoothing=sm, cutoff=cut)


# ------------------------------------------------------------------ test
def run_test(tagger, bg, tuned: dict) -> None:
    t0 = time.time()
    main_key = tuned[tuned["method"]]
    content, generic, main_c = method_cfg(main_key)
    rows_out, per_doc_csv, results = [], [], {}

    def add(section: str, name: str, label: str, rows: list[dict], boot: bool = False):
        s = summarize(rows, boot=1000 if boot else 0, seed=len(rows_out))
        results[(section, name)] = s
        rows_out.append({"section": section, "system": name, "label": label, **s})

    junction = {}
    for source in ("modeb", "modea", "ref"):
        docs = build_docs("test", source, tagger)
        if source == "modeb":
            print(f"test: {len(docs)} documents, {statistics.mean(len(d.ref) + 1 for d in docs):.1f} chapters, "
                  f"{statistics.mean(d.duration for d in docs) / 60:.1f} min on average", flush=True)
            stats_docs = docs
        per: dict[str, list[dict]] = {}
        rand: list[dict[str, dict]] = []
        timing: dict[str, list[float]] = {"dp": [], "tiling": []}
        rng = random.Random(7)
        for doc in docs:
            res = {}
            D = prepare(doc, bg, content, generic)
            res["main"] = run(doc, D, main_c)
            if source == "modeb":
                for m in ("dp", "tiling"):
                    cm, gm, cfg_m = method_cfg(tuned[m])
                    Dm = prepare(doc, bg, cm, gm)
                    t = time.perf_counter()
                    res[m] = score(doc, Dm, boundaries(Dm, cfg_m))
                    timing[m].append(time.perf_counter() - t)
                    res[f"{m}_oracle_k"] = run(doc, Dm, replace(cfg_m, num_chapters=len(doc.ref) + 1))
                base = baselines(doc, D, tuned["period"], rng)
                for name, b in base.items():
                    if not name.startswith("random"):
                        res[name] = score(doc, D, b)
                rand.append({k: score(doc, D, b) for k, b in base.items() if k.startswith("random")})
                # ablations of the chosen method
                res["abl_no_stem"] = run(doc, prepare(doc, bg, content, generic, stem=False), main_c)
                if generic != 0.0:
                    res["abl_no_generic"] = run(doc, prepare(doc, bg, content, 0.0), main_c)
                if generic == 0.0:
                    res["abl_generic_8"] = run(doc, prepare(doc, bg, content, -8.0), main_c)
                for other in CONTENT:
                    if other != content:
                        res[f"abl_content_{other}"] = run(doc, prepare(doc, bg, other, generic), main_c)
                for md in (30.0, 90.0):
                    res[f"abl_min_{md:.0f}"] = run(doc, D, replace(main_c, min_duration=md))
                # which junctions are found (recall per pair of recording types)
                hyp_t = [D.T[b] for b in boundaries(D, main_c)]
                for j, t in enumerate(doc.ref_times):
                    kind = "-".join(sorted((doc.groups[j], doc.groups[j + 1])))
                    found = any(abs(h - t) <= TOL for h in hyp_t)
                    junction.setdefault(kind, []).append(found)
            for name, r in res.items():
                per.setdefault(name, []).append(r)
                per_doc_csv.append({"source": source, "doc": doc.name, "system": name, "minutes": round(doc.duration / 60, 2),
                                    "chapters": len(doc.ref) + 1, **{k: round(v, 4) if isinstance(v, float) else v
                                                                     for k, v in r.items()}})
        label = SOURCES[source][1]
        if source == "modeb":
            add("main", "single", "No chapters (one chapter)", per["single"])
            add("main", "periodic", f"Fixed interval (every {tuned['period'] / 60:.1f} min, dev mean)", per["periodic"])
            add("main", "random_oracle_k", "Random boundaries, true number (5 draws)", mean_random(rand))
            add("main", "even_oracle_k", "Even split, true number", per["even_oracle_k"])
            add("main", "tiling", "TextTiling (Hearst 1997), tuned", per["tiling"], boot=True)
            add("main", "dp", "Utiyama-Isahara DP, tuned", per["dp"], boot=True)
            add("main", "tiling_oracle_k", "TextTiling, true number of chapters", per["tiling_oracle_k"])
            add("main", "dp_oracle_k", "Utiyama-Isahara DP, true number of chapters", per["dp_oracle_k"])
            add("ablation", "main", f"Chosen ({tuned['method']}, tuned; topic words = {CONTENT_LABEL[content]})",
                per["main"])
            add("ablation", "abl_no_stem", "- stemming", per["abl_no_stem"])
            if generic != 0.0:
                add("ablation", "abl_no_generic", "- generic-word filter (our LM's unigrams)", per["abl_no_generic"])
            else:
                add("ablation", "abl_generic_8", "+ generic-word filter (LM unigram log-prob > -8 dropped)",
                    per["abl_generic_8"])
            for other in CONTENT:
                if other != content:
                    add("ablation", f"abl_content_{other}", f"topic words = {CONTENT_LABEL[other]}",
                        per[f"abl_content_{other}"])
            add("ablation", "abl_min_30", "shortest chapter 30 s", per["abl_min_30"])
            add("ablation", "abl_min_90", "shortest chapter 90 s", per["abl_min_90"])
            results["timing"] = {m: (statistics.mean(v), max(v)) for m, v in timing.items()}
        add("source", source, label, per["main"], boot=True)
        print(source, {k: round(v, 3) for k, v in results[("source", source)].items() if isinstance(v, float)}, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["section", "system", "label", "docs", "pk", "wd", "p", "r", "f1", "k_ref", "k_hyp", "k_abs_err"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    with open(OUT / "per_doc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_doc_csv[0]))
        w.writeheader()
        w.writerows(per_doc_csv)
    write_markdown(rows_out, results, tuned, junction, stats_docs)
    print(f"test: {time.time() - t0:.0f} s", flush=True)


def _fmt(s: dict) -> str:
    ci = f" ({s['wd_ci'][0]:.3f}-{s['wd_ci'][1]:.3f})" if "wd_ci" in s else ""
    return (f"| {s['pk']:.3f} | {s['wd']:.3f}{ci} | {s['p']:.2f} | {s['r']:.2f} | {s['f1']:.2f} | "
            f"{s['k_hyp']:.1f} | {s['k_abs_err']:.2f} |")


def write_markdown(rows, results, tuned, junction, docs) -> None:
    head = ("| System | Pk ↓ | WindowDiff ↓ (95% CI) | P | R | F1 @±10 s | chapters | abs. count error |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|")
    lines = ["# Chapter markers: topic segmentation results", "",
             "Generated by `experiments/eval_chapters.py`. Test documents: "
             f"{len(docs)} joins of {PARTS[0]}-{PARTS[1]} excerpts ({EXCERPT_S[0]:.0f}-{EXCERPT_S[1]:.0f} s, cut at "
             "sentence ends) of different test recordings (16 LibriSpeech chapters, 10 TED talks); each junction is "
             "a true chapter boundary and its pause is drawn from the pauses between sentences. "
             f"Mean length {statistics.mean(d.duration for d in docs) / 60:.1f} min, "
             f"{statistics.mean(len(d.ref) + 1 for d in docs):.1f} chapters. Settings were tuned on "
             f"{N_DOCS['dev']} documents built the same way from the dev recordings (Mode B words, lowest "
             f"WindowDiff). Every method keeps chapters at least {MIN_CHAPTER:.0f} s long. Pk and WindowDiff are "
             "error rates over words (k = half the mean true chapter length); P/R/F1 count a predicted chapter "
             f"start as correct within ±{TOL:.0f} s of an unmatched true one. Bootstrap CIs resample documents "
             "(documents share recordings, so they are optimistic).", "",
             f"Chosen: **{tuned['method']}** with `{tuned[tuned['method']]}`; "
             f"TextTiling best `{tuned['tiling']}`, DP best `{tuned['dp']}` "
             "(key = method, content words, generic_logp, parameters).", "",
             "## Methods (Mode B words: ASR + our text models + CTC timing)", "", head]
    for r in rows:
        if r["section"] == "main":
            lines.append(f"| {r['label']} " + _fmt(r))
    lines += ["", "## Input words (chosen method)", "", head]
    for r in rows:
        if r["section"] == "source":
            lines.append(f"| {r['label']} " + _fmt(r))
    lines += ["", "## Ablations (chosen method, Mode B words)", "", head]
    for r in rows:
        if r["section"] == "ablation":
            lines.append(f"| {r['label']} " + _fmt(r))
    lines += ["", "## Which junctions are found (recall @±10 s, chosen method, Mode B)", "",
              "| Junction | boundaries | recall |", "|---|---:|---:|"]
    for kind, v in sorted(junction.items()):
        lines.append(f"| {kind.replace('-', ' / ')} | {len(v)} | {sum(v) / len(v):.2f} |")
    t = results.get("timing", {})
    if t:
        lines += ["", "Run time per document (segmentation only, after tagging): " +
                  ", ".join(f"{m} {a * 1000:.0f} ms mean / {b * 1000:.0f} ms max" for m, (a, b) in t.items()) + "."]
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ examples
def examples(tagger, bg, tuned: dict) -> None:
    content, generic, c = method_cfg(tuned[tuned["method"]])
    cfg = Config.load()
    cfg.chapters = replace(c, content_pos=CONTENT[content], generic_logp=generic)
    lines = ["# Chapter markers on whole recordings (no ground truth)", "",
             "Generated by `experiments/eval_chapters.py --stage examples` with the tuned settings, on the Mode B "
             "words of the longest test talks. Format: the timestamp list written next to the subtitles "
             "(`*.chapters.txt`), then each chapter's description sentence.", ""]
    talks = sorted((it for it in manifest("test") if it["dataset"].startswith("ted")), key=lambda it: -it["duration"])[:4]
    for it in talks:
        rows = json.loads((ROOT / SOURCES["modeb"][0].format(id=it["id"])).read_text(encoding="utf-8"))
        words = [Word(r["text"], r["start"], r["end"]) for r in rows if r["text"]]
        chs = make_chapters(words, cfg, tagger.tag_tokens([w.text for w in words]), bg, None, it["duration"])
        lines += [f"## {it['id']} ({it['duration'] / 60:.0f} min)", "", "```", to_youtube(chs).rstrip(), "```", ""]
        lines += [f"{ch.index}. {ch.summary}" for ch in chs] + [""]
    (OUT / "examples.md").write_text("\n".join(lines), encoding="utf-8")
    plot_example(tagger, bg, tuned)


def plot_example(tagger, bg, tuned: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    docs = build_docs("test", "modeb", tagger)
    doc = docs[3]
    content, generic, c = method_cfg(tuned["tiling"])
    D = prepare(doc, bg, content, generic)
    depth = D.depth(c.window_words, c.smoothing)
    cd, gd, cdp = method_cfg(tuned["dp"])
    Dd = prepare(doc, bg, cd, gd)
    dp_b = [Dd.T[b] / 60 for b in boundaries(Dd, cdp)]
    tl_b = [D.T[b] / 60 for b in boundaries(D, c)]
    fig, ax = plt.subplots(figsize=(9, 3.1), dpi=150)
    ax.plot([D.T[g] / 60 for g in range(1, len(D.sents))], depth, color="#4f46e5", lw=1.2, label="TextTiling depth score")
    for k, t in enumerate(doc.ref_times):
        ax.axvline(t / 60, color="#1a7f37", lw=2.5, alpha=0.35, label="true chapter start" if k == 0 else None)
    for k, t in enumerate(dp_b):
        ax.axvline(t, color="#d1495b", lw=1, ls="--", label="DP chapter start" if k == 0 else None)
    ax.plot(tl_b, [depth.max() * 1.05] * len(tl_b), "v", color="#4f46e5", ms=6, label="TextTiling chapter start")
    ax.set_xlabel("minutes")
    ax.set_ylabel("depth")
    ax.set_title(f"Test document {doc.name}: {len(doc.ref) + 1} joined excerpts (Mode B words)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = ROOT / "results" / "plots" / "chapters_example.png"
    fig.savefig(path)
    print("wrote", path)


# ------------------------------------------------------------------ AMI meetings (real topic boundaries)
AMI = ROOT / "data" / "ami"
AMI_PENALTIES = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0]


def ami_docs(tagger, splits: tuple[str, ...]) -> list[Doc]:
    """AMI meetings (tools/fetch_ami.py) as documents: true chapter starts = top-level topic starts."""
    manifest = AMI / "manifest.json"
    if not manifest.exists():
        raise SystemExit("data/ami missing: python -I tools/fetch_ami.py")
    docs = []
    for m in json.loads(manifest.read_text(encoding="utf-8")):
        if m["split"] not in splits:
            continue
        rec = json.loads((AMI / "meetings" / f"{m['id']}.json").read_text(encoding="utf-8"))
        words = [Word(t, s, e) for t, s, e, _ in rec["words"]]
        ref = [t["first_word"] for t in rec["topics"][1:]]
        docs.append(Doc(m["id"], words, tagger.tag_tokens([w.text for w in words]),
                        sentence_spans(words, [w.start for w in words], [w.end for w in words]),
                        max(rec["duration"], words[-1].end), ref, [words[i].start for i in ref], [m["split"]],
                        [t["type"] or "other" for t in rec["topics"]]))
    return docs


def longest_pauses(D: Document, pauses: np.ndarray, K: int) -> list[int]:
    """K-1 chapter starts at the longest silences, keeping chapters >= MIN_CHAPTER (audio-only baseline)."""
    chosen = [0, len(D.sents)]
    for g in np.argsort(-pauses, kind="stable"):
        if len(chosen) - 1 >= K:
            break
        b = int(g) + 1
        k = int(np.searchsorted(chosen, b))
        if D.T[b] - D.T[chosen[k - 1]] >= MIN_CHAPTER and D.T[chosen[k]] - D.T[b] >= MIN_CHAPTER:
            chosen.insert(k, b)
    return chosen[1:-1]


def gap_pauses(D: Document) -> np.ndarray:
    """Silence before every sentence k >= 1: its start minus the latest end of any earlier word (s)."""
    run_end = np.maximum.accumulate(np.asarray(D.en))
    return np.asarray([max(0.0, D.st[a] - run_end[a - 1]) for a, _ in D.sents[1:]])


def ami_stage(tagger, bg, tuned: dict, workers: int) -> None:
    from scipy.stats import mannwhitneyu

    from subfix import text as stext
    t0 = time.time()
    docs = ami_docs(tagger, ("train", "dev", "test", "nonscenario"))
    content, generic, main_c = method_cfg(tuned[tuned["method"]])
    tc, tg, tile_c = method_cfg(tuned["tiling"])
    per: dict[str, list[dict]] = {}
    rand, csv_rows = [], []
    sweep: dict[float, list[dict]] = {p_: [] for p_ in AMI_PENALTIES}
    by_split: dict[str, list[dict]] = {}
    type_found: dict[str, list[bool]] = {}
    pause_b, pause_o, first_b, first_o = [], [], [], []
    lengths = [b - a for d in docs for a, b in zip([0.0] + d.ref_times, d.ref_times + [d.duration])]
    rng = random.Random(5)
    for doc in docs:
        D, Dt = prepare(doc, bg, content, generic), prepare(doc, bg, tc, tg)
        K = len(doc.ref) + 1
        pauses = gap_pauses(D)
        res = {"dp": run(doc, D, main_c), "tiling": run(doc, Dt, tile_c),
               "dp_oracle_k": run(doc, D, replace(main_c, num_chapters=K)),
               "tiling_oracle_k": run(doc, Dt, replace(tile_c, num_chapters=K)),
               "dp_min30": run(doc, D, replace(main_c, min_duration=30.0)),
               "pauses_oracle_k": score(doc, D, longest_pauses(D, pauses, K))}
        base = baselines(doc, D, tuned["period"], rng)
        res.update({k: score(doc, D, b) for k, b in base.items() if not k.startswith("random")})
        rand.append({k: score(doc, D, b) for k, b in base.items() if k.startswith("random")})
        for p_ in AMI_PENALTIES:
            sweep[p_].append(run(doc, D, replace(main_c, penalty=p_)))
        by_split.setdefault(doc.groups[0], []).append(res["dp"])
        hyp_t = [D.T[b] for b in boundaries(D, main_c)]
        for j, t in enumerate(doc.ref_times):
            type_found.setdefault(doc.parts[j + 1], []).append(any(abs(h - t) <= TOL_WIDE for h in hyp_t))
        # diagnostics: silence and first word at the true topic starts vs every other sentence start
        starts = np.asarray([a for a, _ in D.sents])
        near = {int(np.argmin(np.abs(starts - r))) for r in doc.ref}
        for k in range(1, len(D.sents)):
            first = stext.norm(doc.words[D.sents[k][0]].text)
            (pause_b if k in near else pause_o).append(float(pauses[k - 1]))
            (first_b if k in near else first_o).append(first)
        for name, r in res.items():
            per.setdefault(name, []).append(r)
            csv_rows.append({"meeting": doc.name, "split": doc.groups[0], "system": name,
                             "minutes": round(doc.duration / 60, 1), "topics": K,
                             **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}})
    print(f"AMI zero-shot: {len(docs)} meetings in {time.time() - t0:.0f} s", flush=True)

    # tuned on the 20 AMI development meetings, tested on the 20 unseen ones
    dev = [d for d in docs if d.groups[0] == "dev"]
    test = [d for d in docs if d.groups[0] == "test"]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(dev, bg)) as ex:
        per_dev = list(ex.map(_tune_doc, range(len(dev)), chunksize=1))
    grid = {key: summarize([d[key] for d in per_dev]) for key in per_dev[0]}
    best = {m: list(min((k for k in grid if k[0] == m), key=lambda k: (grid[k]["wd"], grid[k]["pk"])))
            for m in ("dp", "tiling")}
    print("AMI dev best", best, {m: round(grid[tuple(best[m])]["wd"], 3) for m in best}, flush=True)
    shipped_key = tuple(tuned[tuned["method"]][:3])
    p_match = min((k for k in grid if k[:3] == shipped_key),
                  key=lambda k: abs(grid[k]["k_hyp"] - grid[k]["k_ref"]))[3]
    names = [d.name for d in docs]
    on_test: dict[str, list[dict]] = {}
    for doc in test:
        i = names.index(doc.name)
        row = {k: per[k][i] for k in ("single", "even_oracle_k", "pauses_oracle_k", "tiling", "dp", "dp_oracle_k")}
        for m in ("dp", "tiling"):
            cm, gm, cfg_m = method_cfg(best[m])
            row[f"ami_{m}"] = run(doc, prepare(doc, bg, cm, gm), cfg_m)
        row["dp_count"] = run(doc, prepare(doc, bg, content, generic), replace(main_c, penalty=p_match))
        for k, v in row.items():
            on_test.setdefault(k, []).append(v)

    cb, co = Counter(first_b), Counter(first_o)
    cues = [(w, cb[w] / len(first_b), co[w] / len(first_o)) for w, n in cb.most_common(40) if n >= 15]
    results = {
        "meetings": {"n": len(docs), "hours": sum(d.duration for d in docs) / 3600,
                     "topics": statistics.mean(len(d.ref) + 1 for d in docs),
                     "minutes": statistics.mean(d.duration for d in docs) / 60},
        "zero": {k: summarize(v, boot=1000 if k in ("dp", "tiling") else 0) for k, v in per.items()},
        "random": summarize(mean_random(rand)),
        "sweep": {p_: summarize(v) for p_, v in sweep.items()},
        "split": {k: summarize(v) for k, v in by_split.items()},
        "test": {k: summarize(v, boot=1000 if k in ("dp", "ami_dp", "tiling", "ami_tiling") else 0)
                 for k, v in on_test.items()},
        "best": best, "dev_grid_best": {m: grid[tuple(best[m])] for m in best}, "p_match": p_match,
        "types": {k: (len(v), sum(v) / len(v)) for k, v in type_found.items()},
        "pause": {"auc": float(mannwhitneyu(pause_b, pause_o).statistic / (len(pause_b) * len(pause_o))),
                  "median_boundary": float(np.median(pause_b)), "median_other": float(np.median(pause_o)),
                  "over_2s_boundary": float(np.mean(np.asarray(pause_b) > 2.0)),
                  "over_2s_other": float(np.mean(np.asarray(pause_o) > 2.0)), "n_boundary": len(pause_b)},
        "lengths": {"topics": len(lengths), "under_60": float(np.mean(np.asarray(lengths) < 60)),
                    "under_30": float(np.mean(np.asarray(lengths) < 30)), "median": float(np.median(lengths))},
        "cues": sorted(cues, key=lambda x: -x[1] / max(x[2], 1e-9))[:10],
    }
    results["cue_experiment"] = ami_cue_experiment(docs, bg, tuned, workers, tagger)
    print("AMI cue experiment", results["cue_experiment"]["chosen"],
          {k: (round(v["wd"], 3), round(v["f1_30"], 3)) for k, v in results["cue_experiment"]["test"].items()},
          flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ami_per_meeting.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    (OUT / "ami.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    write_ami_markdown(results, tuned)
    print(f"AMI: {time.time() - t0:.0f} s", flush=True)


def write_ami_markdown(R: dict, tuned: dict) -> None:
    head = ("| System | Pk ↓ | WindowDiff ↓ (95% CI) | F1 @±10 s | F1 @±30 s | chapters |\n"
            "|---|---:|---:|---:|---:|---:|")

    def row(label: str, s: dict) -> str:
        ci = f" ({s['wd_ci'][0]:.3f}-{s['wd_ci'][1]:.3f})" if "wd_ci" in s else ""
        return (f"| {label} | {s['pk']:.3f} | {s['wd']:.3f}{ci} | {s['f1']:.2f} | {s['f1_30']:.2f} | "
                f"{s['k_hyp']:.1f} (true {s['k_ref']:.1f}) |")
    M, Z, P, L = R["meetings"], R["zero"], R["pause"], R["lengths"]
    lines = [
        "# Chapter markers on real topic boundaries: AMI meetings", "",
        "Generated by `experiments/eval_chapters.py --stage ami` from the AMI Meeting Corpus manual annotations "
        "(CC BY 4.0; `python -I tools/fetch_ami.py`). Each meeting's manual transcript (forced-aligned word times, "
        "all speakers merged into one stream of utterances) is split into chapters and compared with its "
        f"top-level topic segmentation. {M['n']} annotated meetings, {M['hours']:.1f} h, {M['minutes']:.0f} min and "
        f"{M['topics']:.1f} topics per meeting on average. The topics are finer than chapters: median "
        f"{L['median'] / 60:.1f} min, {L['under_60']:.0%} shorter than the 60 s minimum chapter length "
        f"({L['under_30']:.0%} under 30 s: openings, agenda, chitchat), so those cannot be found as chapters.", "",
        "Same metrics as `results.md`; F1 is also given at ±30 s because a topic change in a meeting is gradual.", "",
        f"## Zero-shot: the shipped settings (tuned on LibriSpeech/TED joins), all {M['n']} meetings", "", head,
        row("No chapters (one chapter)", Z["single"]),
        row(f"Fixed interval (every {tuned['period'] / 60:.1f} min)", Z["periodic"]),
        row("Random boundaries, true number (5 draws)", R["random"]),
        row("Even split, true number", Z["even_oracle_k"]),
        row("Longest silences, true number (audio only)", Z["pauses_oracle_k"]),
        row("TextTiling (shipped settings)", Z["tiling"]),
        row("**Utiyama-Isahara DP (shipped default)**", Z["dp"]),
        row("DP, shortest chapter 30 s", Z["dp_min30"]),
        row("TextTiling, true number of chapters", Z["tiling_oracle_k"]),
        row("DP, true number of chapters", Z["dp_oracle_k"]), "",
        "By split (DP, shipped settings): " + ", ".join(
            f"{k} {v['docs']} meetings WD {v['wd']:.3f}" for k, v in sorted(R["split"].items())) + ".", "",
        "## DP penalty (all meetings)", "",
        "| penalty | WindowDiff | P @±30 s | R @±30 s | F1 @±30 s | chapters |", "|---:|---:|---:|---:|---:|---:|"]
    for p_, s in R["sweep"].items():
        mark = " (default)" if float(p_) == float(tuned[tuned["method"]][-1]) else ""
        lines.append(f"| {float(p_):g}{mark} | {s['wd']:.3f} | {s['p30']:.2f} | {s['r30']:.2f} | {s['f1_30']:.2f} | "
                     f"{s['k_hyp']:.1f} (true {s['k_ref']:.1f}) |")
    precisions = [s["p30"] for s in R["sweep"].values()]
    lines += ["", f"Precision stays between {min(precisions):.2f} and {max(precisions):.2f} at every penalty (random "
              f"boundaries: {R['random']['p30']:.2f}): about one chapter start in three lands within 30 s of a topic "
              "change, however many there are, and recall grows with the number of chapters. WindowDiff keeps "
              f"falling as chapters get fewer (one chapter: {Z['single']['wd']:.3f}) and F1 keeps rising as they get "
              "more, so neither metric picks a sensible count on these meetings: the count is a choice of "
              "granularity (penalty 1 matches AMI's topics, the shipped 6 gives half as many, longer chapters)."]
    T_ = R["test"]
    lines += ["", "## Tuned on AMI: 20 development meetings for tuning, 20 unseen meetings for testing", "",
              f"Best on the development meetings: DP `{R['best']['dp']}`, TextTiling `{R['best']['tiling']}` "
              "(key = method, topic words, generic_logp, parameters).", "", head,
              row("No chapters", T_["single"]), row("Even split, true number", T_["even_oracle_k"]),
              row("TextTiling, shipped settings", T_["tiling"]), row("TextTiling, tuned on AMI", T_["ami_tiling"]),
              row("DP, shipped settings", T_["dp"]), row("DP, tuned on AMI", T_["ami_dp"]),
              row(f"DP, penalty {R['p_match']:g} (chapter count matched to AMI on dev)", T_["dp_count"]),
              row("Longest silences, true number (audio only)", T_["pauses_oracle_k"]),
              row("DP, true number of chapters", T_["dp_oracle_k"]), "",
              "## Which topics are found (DP, shipped settings, ±30 s)", "",
              "| Topic type (AMI ontology) | topic starts | found |", "|---|---:|---:|"]
    for k, (n, r) in sorted(R["types"].items(), key=lambda kv: -kv[1][0]):
        if n >= 15:
            lines.append(f"| {k} | {n} | {r:.2f} |")
    lines += ["", "## Do silences or cue words mark the topic changes?", "",
              f"Silence before a sentence (no speaker talking): median {P['median_boundary']:.2f} s at the "
              f"{P['n_boundary']} true topic starts vs {P['median_other']:.2f} s elsewhere; longer than 2 s at "
              f"{P['over_2s_boundary']:.0%} of topic starts vs {P['over_2s_other']:.0%} of other sentence starts "
              f"(AUC {P['auc']:.2f}).", "",
              "| First word of the sentence | share at topic starts | share elsewhere |", "|---|---:|---:|"]
    for w_, a, b in R["cues"]:
        lines.append(f"| {w_} | {a:.1%} | {b:.1%} |")
    if "cue_experiment" in R:
        lines += cue_markdown(R["cue_experiment"], T_["pauses_oracle_k"])
    (OUT / "ami.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# discourse markers that open a new topic (Hirschberg & Litman 1993), fixed before looking at the data
CUE_WORDS = frozenset({"okay", "ok", "so", "now", "well", "right", "alright", "anyway"})
PAUSE_CAP = 3.0             # s of silence counted at most
W_PAUSE = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]      # nats per second of silence before a chapter start
W_CUE = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]      # nats for a chapter start that opens with a discourse marker


def paired_delta(a: list[dict], b: list[dict], boot: int = 2000, seed: int = 0) -> dict:
    """a minus b on the same documents: mean WindowDiff and micro F1 @±10 s / ±30 s, each with a paired
    bootstrap 95% CI over documents."""
    wd = np.asarray([x["wd"] for x in a]) - np.asarray([x["wd"] for x in b])
    nr = np.asarray([x["n_ref"] for x in a], float)
    na, nb = np.asarray([x["n_hyp"] for x in a], float), np.asarray([x["n_hyp"] for x in b], float)

    def F(h, i, n):
        p, q = h[i].sum() / max(n[i].sum(), 1.0), h[i].sum() / max(nr[i].sum(), 1.0)
        return 2 * p * q / (p + q) if p + q else 0.0
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(a), len(a)) for _ in range(boot)]
    allx = np.arange(len(a))
    out = {"wd": (float(wd.mean()), *np.percentile([wd[i].mean() for i in idx], [2.5, 97.5]).tolist())}
    for key in ("hit", "hit30"):
        ha, hb = np.asarray([x[key] for x in a], float), np.asarray([x[key] for x in b], float)
        ds = [F(ha, i, na) - F(hb, i, nb) for i in idx]
        out["f1" if key == "hit" else "f1_30"] = (F(ha, allx, na) - F(hb, allx, nb),
                                                  *np.percentile(ds, [2.5, 97.5]).tolist())
    return out


def cue_features(doc: Doc, D: Document) -> tuple[np.ndarray, np.ndarray]:
    """Per sentence k (0 for k = 0): silence before it (capped) and whether it opens with a discourse marker."""
    from subfix import text as stext
    pause = np.concatenate([[0.0], np.minimum(gap_pauses(D), PAUSE_CAP)])
    cue = np.asarray([0.0] + [float(stext.norm(doc.words[a].text) in CUE_WORDS) for a, _ in D.sents[1:]])
    return pause, cue


def run_bonus(doc: Doc, D: Document, c, feats, wp: float, wc: float, num: int = 0) -> dict:
    pause, cue = feats
    return score(doc, D, D.segment_dp(c.min_duration, c.penalty, num, bonus=wp * pause + wc * cue)[1:-1])


def _init_cues(docs, bg, setup):
    _G["docs"], _G["bg"], _G["setup"] = docs, bg, setup


def _cue_doc(k: int) -> dict:
    """Every (silence weight, cue weight) with the true number of chapters."""
    doc, bg, (content, generic, c) = _G["docs"][k], _G["bg"], _G["setup"]
    D = prepare(doc, bg, content, generic)
    feats = cue_features(doc, D)
    return {(wp, wc): run_bonus(doc, D, c, feats, wp, wc, num=len(doc.ref) + 1) for wp, wc in product(W_PAUSE, W_CUE)}


def ami_cue_experiment(docs: list[Doc], bg, tuned: dict, workers: int, tagger) -> dict:
    """With the true number of chapters (so only where they start counts): does evidence other than the
    words - silence before a sentence, a discourse marker opening it - move chapter starts onto the real
    topic changes? Weights chosen on the AMI development meetings (WindowDiff), tested on the unseen ones."""
    setup = method_cfg(tuned[tuned["method"]])

    def grid(ds):
        with ProcessPoolExecutor(workers, initializer=_init_cues, initargs=(ds, bg, setup)) as ex:
            per_doc = list(ex.map(_cue_doc, range(len(ds)), chunksize=1))
        return per_doc, {key: summarize([d[key] for d in per_doc]) for key in per_doc[0]}

    _, dev_grid = grid([d for d in docs if d.groups[0] == "dev"])

    def pick(pred):
        return min((k for k in dev_grid if pred(k)), key=lambda k: (dev_grid[k]["wd"], -dev_grid[k]["f1_30"]))
    chosen = {"words": (0.0, 0.0), "silence": pick(lambda k: k[1] == 0), "cues": pick(lambda k: k[0] == 0),
              "both": pick(lambda k: True)}
    per_test, _ = grid([d for d in docs if d.groups[0] == "test"])
    out = {"chosen": {k: list(v) for k, v in chosen.items()},
           "dev": {k: dev_grid[v] for k, v in chosen.items()},
           "test": {k: summarize([d[v] for d in per_test], boot=1000) for k, v in chosen.items()},
           "delta": {k: paired_delta([d[chosen[k]] for d in per_test], [d[(0.0, 0.0)] for d in per_test])
                     for k in ("silence", "cues", "both")}}
    # the same weights on the LibriSpeech/TED joins, also with the true number of chapters
    content, generic, c = setup
    joins: dict[str, list[dict]] = {k: [] for k in chosen}
    for doc in build_docs("test", "modeb", tagger):
        D = prepare(doc, bg, content, generic)
        feats = cue_features(doc, D)
        for k, (wp, wc) in chosen.items():
            joins[k].append(run_bonus(doc, D, c, feats, wp, wc, num=len(doc.ref) + 1))
    out["joins"] = {k: summarize(v) for k, v in joins.items()}
    return out


def cue_markdown(X: dict, silence_only: dict) -> list[str]:
    head = ("| True number of chapters, DP with | weights (dev) | Pk ↓ | WindowDiff ↓ (95% CI) | F1 @±10 s | F1 @±30 s |\n"
            "|---|---|---:|---:|---:|---:|")

    def row(label: str, s: dict, w: str) -> str:
        ci = f" ({s['wd_ci'][0]:.3f}-{s['wd_ci'][1]:.3f})" if "wd_ci" in s else ""
        return f"| {label} | {w} | {s['pk']:.3f} | {s['wd']:.3f}{ci} | {s['f1']:.2f} | {s['f1_30']:.2f} |"

    def wts(k: str) -> str:
        wp, wc = X["chosen"][k]
        return f"silence {wp:g}, cue {wc:g}"

    def delta(k: str) -> str:
        d = X["delta"][k]
        return (f"WindowDiff {d['wd'][0]:+.3f} ({d['wd'][1]:+.3f} to {d['wd'][2]:+.3f}), "
                f"F1 @±30 s {d['f1_30'][0]:+.3f} ({d['f1_30'][1]:+.3f} to {d['f1_30'][2]:+.3f})")
    J, T_ = X["joins"], X["test"]
    return [
        "", "## Experiment: silence and discourse markers in the DP (not shipped)", "",
        "Each chapter start earns a bonus of `silence weight × silence before the sentence (s, at most "
        f"{PAUSE_CAP:g})` + `cue weight` if the sentence opens with a discourse marker "
        f"({', '.join(sorted(CUE_WORDS))}: the classic list, fixed in advance). The number of chapters is fixed "
        "at the true one, so only placement counts. Weights were chosen on the 20 development meetings "
        "(WindowDiff) and tested on the 20 unseen meetings:", "", head,
        row("words only", T_["words"], "–"),
        row("words + silence", T_["silence"], wts("silence")),
        row("words + discourse markers", T_["cues"], wts("cues")),
        row("words + silence + discourse markers", T_["both"], wts("both")),
        row("silence only (longest silences)", silence_only, "–"), "",
        "Change against words only (paired bootstrap over the 20 meetings, 95% CI): "
        f"+ silence: {delta('silence')}; + markers: {delta('cues')}; + both: {delta('both')}.", "",
        "The same weights on the LibriSpeech/TED joins (Mode B, 200 test documents, true number of chapters "
        "there too) give WindowDiff " + ", ".join(f"{J[k]['wd']:.3f} ({name})" for k, name in (
            ("words", "words only"), ("silence", "+ silence"), ("cues", "+ markers"), ("both", "+ both"))) +
        ". There, every excerpt starts at a random sentence and junction pauses are drawn from ordinary pauses, "
        "so neither cue carries signal by construction: the change shows what the bonus costs when the cues "
        "are absent."]


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["all", "tune", "test", "examples", "ami"], default="all")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from subfix.postag import PerceptronTagger
    tagger = PerceptronTagger.load(ROOT / "models" / "pos" / "perceptron.json.gz")
    bg = Background(ROOT / "models" / "lm" / "kn4.npz")
    if args.stage in ("all", "tune"):
        tuned = tune(tagger, bg, args.workers)
    else:
        tuned = json.loads((OUT / "tuned.json").read_text(encoding="utf-8"))
    if args.stage in ("all", "test"):
        run_test(tagger, bg, tuned)
    if args.stage in ("all", "examples"):
        examples(tagger, bg, tuned)
    if args.stage == "ami" or (args.stage == "all" and (AMI / "manifest.json").exists()):
        ami_stage(tagger, bg, tuned, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())

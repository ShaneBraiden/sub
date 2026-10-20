"""Run systems over benchmark manifests and compute the imp.md 6.1 metrics.

  python experiments/run_eval.py --manifest test --systems noop b1_symspell_lt b2_ffsubsync b3_raw_whisper ours
  python experiments/run_eval.py --manifest test --ablation timing_source --datasets libri-test-clean ted-test
  python experiments/run_eval.py --sweep 0.05 0.10 0.20

System outputs are cached in experiments/outputs/<system>/<variant>/<id>.{srt,words.json,run.json};
tables go to results/<manifest>/ (main) or results/ablations/ (ablations).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import yaml  # noqa: E402

from metrics import format_metrics as FM  # noqa: E402
from metrics import text_metrics as TM  # noqa: E402
from metrics import timing as TIM  # noqa: E402
from subfix import text as T  # noqa: E402
from subfix.config import Config  # noqa: E402
from subfix.paths import CONFIG_DIR, DATA_DIR, RESULTS_DIR  # noqa: E402
from subfix.subio import read_srt, tokenize, write_srt  # noqa: E402
from subfix.utils.cache import load_json, save_json  # noqa: E402

log = logging.getLogger("run_eval")
OUT = ROOT / "experiments" / "outputs"
SYSTEMS = yaml.safe_load((ROOT / "experiments" / "systems.yaml").read_text(encoding="utf-8"))


def load_manifest(name: str) -> list[dict]:
    path = DATA_DIR / "benchmark" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ablation_tuned(name: str) -> dict:
    path = CONFIG_DIR / "tuned_ablations.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get(name, {}) or {}


# ------------------------------------------------------------------ running systems
def _paths(sys_name: str, item: dict) -> dict:
    d = OUT / sys_name / item["variant"]
    return {"srt": d / f"{item['id']}.srt", "words": d / f"{item['id']}.words.json",
            "run": d / f"{item['id']}.run.json", "dir": d}


def run_system(item: dict, sys_name: str, sysdef: dict, models_cache: dict, force: bool = False) -> dict:
    p = _paths(sys_name, item)
    if p["run"].exists() and p["srt"].exists() and not force:
        return p
    p["dir"].mkdir(parents=True, exist_ok=True)
    audio = str(ROOT / item["audio"])
    corrupted = str(ROOT / item["corrupted_srt"])
    kind = sysdef["type"]
    overrides = dict(sysdef.get("overrides") or {})
    cfg = Config.load(overrides=overrides)
    info: dict = {"system": sys_name, "type": kind, "duration": item["duration"]}
    t0 = time.perf_counter()
    if kind == "noop":
        shutil.copy(corrupted, p["srt"])
    elif kind == "raw_whisper":
        from subfix.asr import transcribe
        from subfix.pipeline import raw_whisper_cues
        asr = transcribe(audio, model=cfg.asr.model)
        write_srt(raw_whisper_cues(asr), p["srt"])
        save_json(p["words"], [{"text": w.text, "start": w.start, "end": w.end} for w in asr.words])
        info["asr_elapsed"] = asr.info.get("elapsed", 0.0)
    elif kind == "ffsubsync":
        from subfix.sync import run_ffsubsync
        ok = run_ffsubsync(audio, corrupted, str(p["srt"]))
        info["ok"] = ok
        if not ok:
            shutil.copy(corrupted, p["srt"])
    elif kind == "pipeline":
        from subfix.models import Models
        from subfix.pipeline import run
        models = models_cache.get("models")
        if models is None:
            models = models_cache["models"] = Models(cfg)
        models.cfg = cfg
        res = run(audio, corrupted if sysdef.get("mode", "A") == "A" else None, cfg, models)
        write_srt(res.cues, p["srt"])
        save_json(p["words"], [{"text": w.text, "start": w.start, "end": w.end} for w in res.words])
        info.update({"stage_times": res.stage_times, "sync": res.sync.to_json() if res.sync else None,
                     "edit_counts": res.edit_counts(), "timing": res.timing_stats,
                     "asr_elapsed": res.asr.info.get("elapsed", 0.0) if res.asr else 0.0})
    else:
        raise ValueError(kind)
    info["wall"] = time.perf_counter() - t0
    save_json(p["run"], info)
    return p


# ------------------------------------------------------------------ metrics per item
def _surface(cues) -> list[str]:
    return [t for c in cues for t in tokenize(c.text.replace("\n", " "))]


def item_parts(item: dict, paths: dict, sysdef: dict, do_errant: bool) -> dict:
    ref = read_srt(ROOT / item["ref_srt"])
    hyp = read_srt(paths["srt"])
    src = read_srt(ROOT / item["corrupted_srt"])
    ref_s, hyp_s, src_s = _surface(ref), _surface(hyp), _surface(src)
    mode_b = sysdef.get("mode", "A") == "B"
    parts: dict = {"id": item["id"], "dataset": item["dataset"], "has_punct": item["has_punct"],
                   "ref_text": " ".join(ref_s), "hyp_text": " ".join(hyp_s), "src_text": " ".join(src_s),
                   "ref_norm": [T.norm(t) for t in ref_s if T.norm(t)],
                   "hyp_norm": [T.norm(t) for t in hyp_s if T.norm(t)],
                   "src_norm": [T.norm(t) for t in src_s if T.norm(t)], "mode_b": mode_b}
    if do_errant and not mode_b:
        src_chunks = [tokenize(c.text.replace("\n", " ")) for c in src]
        e_ref, e_hyp = ref_s, hyp_s
        if not item["has_punct"]:
            # lowercase, unpunctuated references (TED): score word edits only, so systems are not
            # penalised for the punctuation/casing they add
            def strip(toks):
                return [n for n in (T.norm(t) for t in toks) if n]
            src_chunks = [strip(ch) for ch in src_chunks]
            e_ref, e_hyp = strip(ref_s), strip(hyp_s)
        try:
            parts["errant"] = TM.errant_scores(src_chunks, e_ref, e_hyp)
        except Exception as exc:
            log.warning("ERRANT failed on %s: %s", item["id"], exc)
    if item["has_punct"]:
        parts["punct_pairs"] = TM.punct_pairs(ref_s, hyp_s)
    ref_words = load_json(ROOT / item["ref_words"])
    parts["cue_timing"] = TIM.cue_timing(ref, hyp, ref_words)
    if paths["words"].exists():
        parts["word_timing"] = TIM.word_timing(ref_words, load_json(paths["words"]))
    parts["format"] = FM.format_counts(hyp, Config.load())
    run = load_json(paths["run"]) if paths["run"].exists() else {}
    proc = sum((run.get("stage_times") or {}).values()) if run.get("stage_times") else run.get("wall", 0.0)
    parts["rtf"] = (proc + (run.get("asr_elapsed") or 0.0)) / max(item["duration"], 1e-6) if run else float("nan")
    if not mode_b:
        parts["recall"] = error_recall(load_json(ROOT / item["corruption_log"]), ref_s, hyp_s, src_s)
    return parts


def error_recall(clog: dict, ref_s: list[str], hyp_s: list[str], src_s: list[str]) -> dict:
    """Per-error-type: was the injected error repaired in the hypothesis?"""
    from rapidfuzz.distance import Levenshtein
    rn = [T.norm(t) or t for t in ref_s]
    hn = [T.norm(t) or t for t in hyp_s]
    sn = [T.norm(t) or t for t in src_s]
    ref_ok = set()
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(rn, hn):
        if tag == "equal":
            ref_ok.update(range(i1, i2))
    deleted_src = set()
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(sn, hn):
        if tag == "delete" or (tag == "replace" and (i2 - i1) > (j2 - j1)):
            deleted_src.update(range(i1, i2))
    fixed, total = Counter(), Counter()
    for e in clog.get("text_errors", []):
        t = e["type"]
        total[t] += 1
        if t == "dup":
            s = e["src_index"]
            fixed[t] += int(s in deleted_src or (s + 1) in deleted_src)
        else:
            fixed[t] += int(e["ref_index"] in ref_ok)
    return {"fixed": dict(fixed), "total": dict(total)}


# ------------------------------------------------------------------ aggregation
def aggregate(parts: list[dict]) -> dict:
    if not parts:
        return {}
    out: dict = {"n_files": len(parts)}
    out.update(TM.wer_cer([p["ref_text"] for p in parts], [p["hyp_text"] for p in parts]))
    if not parts[0]["mode_b"]:
        out.update(TM.edit_scores([p["src_norm"] for p in parts], [p["ref_norm"] for p in parts],
                                  [p["hyp_norm"] for p in parts]))
    er = [p["errant"] for p in parts if "errant" in p]
    if er:
        tp = sum(e["errant_tp"] for e in er)
        fp = sum(e["errant_fp"] for e in er)
        fn = sum(e["errant_fn"] for e in er)
        P, R, F = TM.prf(tp, fp, fn)
        out.update({"errant_p": P, "errant_r": R, "errant_f05": F})
        types: dict = defaultdict(lambda: Counter())
        for e in er:
            for typ, d in e["errant_types"].items():
                types[typ].update({"tp": d["tp"], "fp": d["fp"], "fn": d["fn"]})
        out["errant_types"] = {k: dict(v) for k, v in types.items()}
    pairs = [x for p in parts if p.get("punct_pairs") for x in p["punct_pairs"]]
    if pairs:
        out.update(TM.punct_scores(pairs))
    out.update(TIM.summarize_cue_timing([p["cue_timing"] for p in parts]))
    wt = [p["word_timing"] for p in parts if "word_timing" in p]
    if wt:
        out.update(TIM.summarize_word_timing(wt))
    out.update(FM.summarize_format([p["format"] for p in parts]))
    rtfs = [p["rtf"] for p in parts if not math.isnan(p["rtf"])]
    out["rtf"] = sum(rtfs) / len(rtfs) if rtfs else float("nan")
    rec = [p["recall"] for p in parts if "recall" in p]
    if rec:
        fixed, total = Counter(), Counter()
        for r in rec:
            fixed.update(r["fixed"])
            total.update(r["total"])
        out["recall_by_type"] = {k: {"fixed": fixed[k], "total": total[k], "recall": fixed[k] / total[k]}
                                 for k in sorted(total) if total[k]}
    return out


# ------------------------------------------------------------------ tables
MAIN_COLS = [("wer", "WER", 100, ".2f"), ("cer", "CER", 100, ".2f"), ("errant_f05", "ERRANT F0.5", 100, ".1f"),
             ("edit_f05", "Edit F0.5", 100, ".1f"), ("overcorrection", "Over-corr %", 100, ".2f"),
             ("punct_macro_f1", "Punct F1", 100, ".1f"), ("casing_acc", "Case acc", 100, ".1f"),
             ("t_mean_ms", "Onset |Δt| ms", 1, ".0f"), ("t_within_100", "% onset ≤100ms", 100, ".1f"),
             ("t_within_250", "% onset ≤250ms", 100, ".1f"), ("w_boundary_mean_ms", "Word |Δt| ms", 1, ".0f"), ("fmt_cps17_ok", "CPS≤17 %", 100, ".1f"),
             ("fmt_cps20_ok", "CPS≤20 %", 100, ".1f"), ("fmt_lines_ok", "Lines ok %", 100, ".1f"),
             ("fmt_all_ok", "All fmt ok %", 100, ".1f"), ("rtf", "RTF", 1, ".3f")]


def fmt_val(v, scale, f) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return format(v * scale, f)


def markdown_table(rows: list[tuple[str, dict]], cols=MAIN_COLS) -> str:
    head = "| System | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    lines = [head, sep]
    for label, m in rows:
        lines.append(f"| {label} | " + " | ".join(fmt_val(m.get(c[0]), c[2], c[3]) for c in cols) + " |")
    return "\n".join(lines)


def recall_table(rows: list[tuple[str, dict]]) -> str:
    types = sorted({t for _, m in rows for t in (m.get("recall_by_type") or {})})
    if not types:
        return ""
    lines = ["| System | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for label, m in rows:
        rb = m.get("recall_by_type") or {}
        lines.append(f"| {label} | " + " | ".join(
            f"{rb[t]['recall'] * 100:.1f} ({rb[t]['fixed']}/{rb[t]['total']})" if t in rb else "–" for t in types) + " |")
    return "\n".join(lines)


def write_results(out_dir: Path, results: dict, labels: dict, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "metrics.json", results)
    datasets = sorted({d for r in results.values() for d in r})
    md = [f"# {title}", ""]
    csv_rows = []
    for ds in datasets:
        rows = [(labels[s], results[s][ds]) for s in results if ds in results[s]]
        md += [f"## {ds}", "", markdown_table(rows), ""]
        rt = recall_table(rows)
        if rt:
            md += ["Per-error-type recall (% repaired, fixed/total):", "", rt, ""]
        for s in results:
            if ds in results[s]:
                flat = {k: v for k, v in results[s][ds].items() if not isinstance(v, dict)}
                csv_rows.append({"system": s, "dataset": ds, **flat})
    (out_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(csv_rows).to_csv(out_dir / "results.csv", index=False)


# ------------------------------------------------------------------ driver
def evaluate(manifest: str, systems: dict[str, dict], datasets: list[str] | None, limit: int | None,
             do_errant: bool, force: bool, out_dir: Path, title: str, labels: dict | None = None) -> dict:
    items = load_manifest(manifest)
    if datasets:
        items = [it for it in items if it["dataset"] in datasets]
    if limit:
        by: dict = defaultdict(list)
        for it in items:
            by[it["dataset"]].append(it)
        items = [it for v in by.values() for it in v[:limit]]
    models_cache: dict = {}
    results: dict = {}
    per_item_rows = []
    for name, sd in systems.items():
        parts_by_ds: dict = defaultdict(list)
        for it in items:
            t0 = time.perf_counter()
            paths = run_system(it, name, sd, models_cache, force)
            parts = item_parts(it, paths, sd, do_errant)
            parts_by_ds[it["dataset"]].append(parts)
            m = aggregate([parts])
            per_item_rows.append({"system": name, "id": it["id"], "dataset": it["dataset"],
                                  **{k: v for k, v in m.items() if not isinstance(v, dict)}})
            print(f"[{name}] {it['id']}: WER {m.get('wer', float('nan')) * 100:.2f}  "
                  f"|Δt| {m.get('t_mean_ms', float('nan')):.0f}ms  ({time.perf_counter() - t0:.1f}s)", flush=True)
        results[name] = {ds: aggregate(ps) for ds, ps in parts_by_ds.items()}
        allp = [p for ps in parts_by_ds.values() for p in ps]
        if len(parts_by_ds) > 1:
            results[name]["ALL"] = aggregate(allp)
    labels = labels or {n: sd.get("label", n) for n, sd in systems.items()}
    write_results(out_dir, results, labels, title)
    import pandas as pd
    pd.DataFrame(per_item_rows).to_csv(out_dir / "per_item.csv", index=False)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="test")
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--ablation", default=None)
    ap.add_argument("--sweep", nargs="*", type=float, default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-errant", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.sweep:
        systems = {"noop": SYSTEMS["systems"]["noop"], "ours": SYSTEMS["systems"]["ours"]}
        for rate in args.sweep:
            name = f"test_rate{rate:.2f}"
            evaluate(name, systems, args.datasets, args.limit, False, args.force,
                     RESULTS_DIR / "ablations" / f"corruption_rate_{rate:.2f}", f"Corruption rate {rate:.0%}")
        return
    if args.ablation:
        variants = SYSTEMS["ablations"][args.ablation]
        base = SYSTEMS["systems"]["ours"].get("overrides") or {}
        systems = {}
        for vname, ov in variants.items():
            ov = dict(base, **(ov or {}))
            if args.ablation == "fusion_signals":
                ov.update(ablation_tuned(vname))
            systems[f"abl_{args.ablation}_{vname}"] = {"type": "pipeline", "label": vname, "overrides": ov}
        evaluate(args.manifest, systems, args.datasets, args.limit, not args.no_errant, args.force,
                 RESULTS_DIR / "ablations" / args.ablation, f"Ablation: {args.ablation}")
        return
    names = args.systems or list(SYSTEMS["systems"])
    systems = {n: SYSTEMS["systems"][n] for n in names}
    evaluate(args.manifest, systems, args.datasets, args.limit, not args.no_errant, args.force,
             RESULTS_DIR / args.manifest, f"Results: {args.manifest}")


if __name__ == "__main__":
    main()

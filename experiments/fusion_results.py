"""Dev-set experiments of the neural fusion network -> results/fusion/ (the test split is never touched here).

python experiments/fusion_results.py [--stages variants pipeline evidence holdout report]

  variants  trains each variant with three seeds (experiments/train_fusion.py, cached in
            cache/train/fusion/runs/): the spread of the dev numbers, and the WER each variant reaches
            at a given precision. The first seed of each goes to models/fusion (the network the
            pipeline runs) or models/fusion_abl/ (the ablation in experiments/systems.yaml).
  pipeline  the rule and every variant through the whole pipeline on the dev clips
  evidence  how well single signals rank the candidates per dataset, including the audio our
            CTC model trained on (is its evidence too good there?)
  holdout   leaves one error type out of the training corruptions, on the smaller training set of
            dev-clean + TED: does the network still repair an error type it never saw?
  report    results/fusion/results.md
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "experiments")]

from fusion_data import FEATURES as FEATURE_DIR, WORK  # noqa: E402
from subfix.fusenet import FEATURE_INDEX, KINDS, SCORED, NeuralFusion, UnitTable  # noqa: E402
from subfix.paths import MODELS_DIR  # noqa: E402
from train_fusion import OUT_OF_KIND, Corpus, auc, rule_numbers, threshold_counts  # noqa: E402

RES = ROOT / "results" / "fusion"
RULE_TUNING = ROOT / "results" / "tuning" / "fusion_grid.json"
RUNS = WORK / "runs"
SEEDS = (0, 1, 2)
PRECISIONS = (0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98)
ERROR_TYPES = ("entity", "asr", "homophone", "spelling", "drop", "dup")
HOLDOUT_SPLITS = ["dev-clean", "ted"]
# name -> train_fusion.py arguments (the defaults are the shipped network)
VARIANTS = {"bilstm": [], "bilstm_64": ["--hidden", "64"], "bilstm_128": ["--hidden", "128"],
            "words": ["--arch", "words"], "mlp": ["--arch", "mlp"], "logistic": ["--arch", "linear"],
            "no_ctc": ["--no-ctc"]}
LABELS = {"rule": "Linear rule (same selection score)", "rule_f05": "Linear rule tuned on F0.5 alone", "bilstm": "**BiLSTM over the signals, 32 hidden units (shipped)**",
          "bilstm_64": "BiLSTM, 64 hidden units", "bilstm_128": "BiLSTM, 128 hidden units",
          "words": "BiLSTM + word and character embeddings", "mlp": "MLP per candidate (no context)",
          "logistic": "Logistic regression", "no_ctc": "BiLSTM without the CTC evidence"}


def script(name: str, *args: str, check: bool = True) -> None:
    env = dict(os.environ, PYTHONUTF8="1")
    subprocess.run([sys.executable, str(ROOT / "experiments" / name), *args], check=check, env=env)


def save(name: str, data) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    (RES / name).write_text(json.dumps(data, indent=1), encoding="utf-8")


def load(name: str):
    path = RES / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def train(run: str, *args: str) -> dict:
    out = RUNS / run
    if not (out / "metrics.json").exists():
        # no exit-code check: on Windows, training a CUDA LSTM can crash at interpreter exit
        # (0xC0000409, also with a few lines of plain PyTorch), after everything has been saved
        script("train_fusion.py", "--out", str(out), *args, check=False)
    return json.loads((out / "metrics.json").read_text(encoding="utf-8"))


def dev_probabilities(run: Path) -> tuple[Corpus, np.ndarray]:
    """P(edit helps) of each dev unit, through the class the pipeline uses."""
    net = NeuralFusion(str(run))
    forms: dict[str, int] = {"": 0}
    dev = Corpus("dev", None, forms)
    names = list(forms)
    prob = np.zeros(len(dev.kinds), np.float32)
    for s in dev.seqs:
        a, b = s["start"], s["end"]
        table = UnitTable(dev.kinds[a:b], [names[k] for k in dev.s[a:b]], [names[k] for k in dev.a[a:b]], dev.num[a:b],
                          np.full(b - a, -1))
        prob[a:b] = net.probabilities(table)[np.arange(b - a), OUT_OF_KIND[dev.kinds[a:b]]]
    return dev, prob


def wer_at_precision(run: Path) -> dict[str, float | None]:
    """The lowest WER estimate over all thresholds at each precision (None if no threshold reaches
    it), so the comparison does not depend on where the selection score put each variant."""
    dev, prob = dev_probabilities(run)
    cand = dev.rule >= 0
    out = np.where(dev.scored, OUT_OF_KIND[dev.kinds], -1)[cand]
    base = ((dev.rule == 1) & ~dev.scored)[cand]
    tp, fp, net = threshold_counts(prob[cand], dev.gain[cand], out, base)
    err, n_ref = sum(s["err"] for s in dev.seqs), sum(s["n_ref"] for s in dev.seqs)
    precision, wer = tp / np.maximum(tp + fp, 1), 100 * (err - net) / n_ref
    return {f"{p:.2f}": round(float(wer[precision >= p].min()), 3) if (precision >= p).any() else None
            for p in PRECISIONS}


def mean_sd(values: list[float]) -> dict:
    return {"mean": round(statistics.mean(values), 4), "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0}


# ------------------------------------------------------------------ stages
def stage_variants() -> None:
    table = {}
    for name, args in VARIANTS.items():
        runs = [train(f"{name}_s{seed}", "--seed", str(seed), *args) for seed in SEEDS]
        best = [r["best"] for r in runs]
        curves = [wer_at_precision(RUNS / f"{name}_s{seed}") for seed in SEEDS]
        table[name] = {
            "params": runs[0]["params"], "minutes": mean_sd([r["minutes"] for r in runs]),
            "epoch": mean_sd([b["epoch"] for b in best]),
            **{k: mean_sd([b["all"][k] for b in best]) for k in ("f05", "precision", "recall", "wer")},
            "libri_wer": mean_sd([b["libri"]["wer"] for b in best]), "ted_wer": mean_sd([b["ted"]["wer"] for b in best]),
            "auc": {k: mean_sd([b["auc"][k] for b in best]) for k in SCORED},
            "wer_at_precision": {p: mean_sd([c[p] for c in curves]) if all(c[p] is not None for c in curves)
                                 else None for p in curves[0]},
            "thresholds_first_seed": best[0]["thresholds"]}
        target = MODELS_DIR / "fusion" if name == "bilstm" else MODELS_DIR / "fusion_abl" / name
        target.mkdir(parents=True, exist_ok=True)
        for f in ("model.pt", "metrics.json"):
            shutil.copy(RUNS / f"{name}_s0" / f, target / f)
    first = json.loads((RUNS / "bilstm_s0" / "metrics.json").read_text(encoding="utf-8"))
    # the rule's numbers come from the features, since re-tuning the rule does not retrain the network
    rule = rule_numbers(Corpus("dev", None, {"": 0}))
    tuned = json.loads(RULE_TUNING.read_text(encoding="utf-8"))["best"]["dev_f05"]
    if abs(rule["all"]["f05"] - tuned) > 1e-3:
        raise SystemExit(f"cache/train/fusion/features/dev hold another rule's decisions (F0.5 {rule['all']['f05']}, tuned "
                         f"{tuned}): delete that folder and run experiments/fusion_data.py --stages features")
    save("variants.json", {"variants": table, "rule": rule, "perfect_decisions_wer": first["oracle_wer"],
                           "train": {k: first[k] for k in ("train_sequences", "train_units", "train_candidates")}})


def pipeline_row(metrics: dict) -> dict:
    keep = ("wer", "cer", "edit_f05", "overcorrection", "punct_macro_f1", "t_mean_ms", "rtf")
    return {**{k: metrics.get(k) for k in keep},
            "recall": {t: v["recall"] for t, v in metrics.get("recall_by_type", {}).items()}}


def run_dev(systems: dict[str, dict], out_dir: Path, title: str) -> dict:
    """Whole pipeline on the dev clips; returns name -> dataset -> numbers."""
    from run_eval import OUT, evaluate
    defs = {f"fusiondev_{name}": {"type": "pipeline", "label": name, "overrides": over} for name, over in systems.items()}
    res = evaluate("dev", defs, None, None, False, True, out_dir, title)
    for key in defs:
        shutil.rmtree(OUT / key, ignore_errors=True)
    return {name: {ds: pipeline_row(m) for ds, m in res[f"fusiondev_{name}"].items()} for name in systems}


def stage_pipeline() -> None:
    from run_eval import ablation_tuned
    systems = {"rule": {"fusion.method": "rule"}, "rule_f05": {"fusion.method": "rule", **ablation_tuned("rule_f05")}}
    systems.update({name: {"fusion.model": str(RUNS / f"{name}_s0")} for name in VARIANTS})
    save("pipeline.json", run_dev(systems, RES / "dev_pipeline", "Fusion variants, dev clips"))


def stage_evidence() -> None:
    """AUC of single signals (is the edit good?) on the candidates of each dataset."""
    groups: dict[str, dict[str, list]] = {}
    for folder, tag in (("train", ""), ("dev", "benchmark ")):
        for path in sorted((FEATURE_DIR / folder).glob("*.npz")):
            with np.load(path) as z:
                name = tag + ("libri" if tag and "libri" in str(z["dataset"]) else
                              "ted" if tag else str(z["dataset"]))
                g = groups.setdefault(name, {"kinds": [], "num": [], "gain": [], "scored": []})
                for k in g:
                    g[k].append(z[k])
    ix = FEATURE_INDEX
    out = {}
    for name, g in groups.items():
        kinds, num, gain, scored = (np.concatenate(g[k]) for k in ("kinds", "num", "gain", "scored"))
        row = {"fit_before": round(float(num[scored & (num[:, ix["ctc_ok"]] > 0), ix["ctc_fit_before"]].mean()), 3)}
        for kind in SCORED:
            m = scored & (kinds == KINDS.index(kind))
            y = (gain[m] > 0).astype(np.int64)
            conf = "span_conf" if kind == "del" else "p"
            row[kind] = {"n": int(m.sum()), "good": round(float(y.mean()), 3),
                         "ctc": round(auc(y, num[m, ix["ctc_delta"]]), 3),
                         "whisper": round(auc(y, num[m, ix[conf]]), 3),
                         "lm": round(auc(y, num[m, ix["lm"]]), 3)}
        m = scored & (kinds == KINDS.index("del")) & (num[:, ix["dup"]] > 0)
        y = (gain[m] > 0).astype(np.int64)
        row["duplicate del"] = {"n": int(m.sum()), "good": round(float(y.mean()), 3),
                                "ctc": round(auc(y, num[m, ix["ctc_delta"]]), 3),
                                "whisper": round(auc(y, num[m, ix["span_conf"]]), 3),
                                "lm": round(auc(y, num[m, ix["lm"]]), 3)}
        out[name] = row
    save("evidence.json", out)


def stage_holdout() -> None:
    subset = ["--datasets", *HOLDOUT_SPLITS]
    train("holdout_all", *subset)
    systems = {"rule": {"fusion.method": "rule"}, "all_types": {"fusion.model": str(RUNS / "holdout_all")}}
    for t in ERROR_TYPES:
        script("fusion_data.py", "--stages", "features", "--splits", *HOLDOUT_SPLITS, "--holdout", t)
        train(f"holdout_{t}", *subset, "--train-group", f"train_no_{t}")
        systems[f"no_{t}"] = {"fusion.model": str(RUNS / f"holdout_{t}")}
    save("holdout.json", run_dev(systems, RES / "dev_holdout", "Fusion, error types left out of training"))


# ------------------------------------------------------------------ report
def pm(entry: dict, nd: int = 3) -> str:
    return f"{entry['mean']:.{nd}f} ± {entry['sd']:.{nd}f}"


def report() -> None:
    lines = ["# Neural fusion on the dev clips", "",
             "Written by `experiments/fusion_results.py`. All of this is dev data, used to choose the network; the "
             "test numbers are in `results/test/` and `results/ablations/fusion_method/`.", ""]
    v = load("variants.json")
    if v:
        lines += ["## Variants (three seeds, fusion stage only)", "",
                  f"{v['train']['train_candidates']:,} training candidates in {v['train']['train_units']:,} aligned words. "
                  "WER is the estimate after fusion alone (errors of the corrupted subtitle minus the gain of "
                  f"the accepted edits). Perfect decisions on the official corruption would give {v['perfect_decisions_wer']['official_all']:.2f}.", "",
                  "| Who decides | Params | F0.5 | Precision | Recall | WER % | AUC sub | AUC ins | AUC del |", "|---|---|---|---|---|---|---|---|---|"]
        r = v["rule"]["all"]
        lines.append(f"| {LABELS['rule']} | 9 | {r['f05']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['wer']:.2f} | – | – | – |")
        o = json.loads(RULE_TUNING.read_text(encoding="utf-8"))["f05_only"]
        lines.append(f"| {LABELS['rule_f05']} | 9 | {o['dev_f05']:.3f} | {o['dev_precision']:.3f} | {o['dev_recall']:.3f} | {o['dev_wer']:.2f} | – | – | – |")
        for name, row in v["variants"].items():
            lines.append(f"| {LABELS[name]} | {row['params']:,} | {pm(row['f05'])} | {pm(row['precision'])} | {pm(row['recall'])} | "
                         f"{pm(row['wer'], 2)} | {pm(row['auc']['sub'])} | {pm(row['auc']['ins'])} | {pm(row['auc']['del'])} |")
        lines += ["", "**Lowest WER % at a given precision** (over all thresholds; the rule reaches "
                  f"precision {r['precision']:.3f} at WER {r['wer']:.2f})", "",
                  "| Who decides | " + " | ".join(f"≥ {p:.2f}" for p in PRECISIONS) + " |", "|---|" + "---|" * len(PRECISIONS)]
        for name, row in v["variants"].items():
            curve = row["wer_at_precision"]
            lines.append(f"| {LABELS[name]} | " + " | ".join(f"{curve[f'{p:.2f}']['mean']:.2f}" if curve[f"{p:.2f}"] else "–"
                                                             for p in PRECISIONS) + " |")
        lines.append("")
    p = load("pipeline.json")
    if p:
        lines += ["## Whole pipeline (first seed of each variant, all 12 dev clips)", "",
                  "| Who decides | WER % | CER % | Edit F0.5 | Over-corr. % | Punct F1 | RTF | " +
                  " | ".join(f"{t} recall" for t in ERROR_TYPES) + " |", "|---|---|---|---|---|---|---|" + "---|" * len(ERROR_TYPES)]
        for name, by_ds in p.items():
            m = by_ds["ALL"]
            lines.append(f"| {LABELS[name]} | {m['wer'] * 100:.2f} | {m['cer'] * 100:.2f} | {m['edit_f05'] * 100:.1f} | {m['overcorrection'] * 100:.2f} | "
                         f"{(m['punct_macro_f1'] or 0) * 100:.1f} | {m['rtf']:.3f} | " +
                         " | ".join(f"{m['recall'].get(t, float('nan')) * 100:.0f}" for t in ERROR_TYPES) + " |")
        lines.append("")
    e = load("evidence.json")
    if e:
        lines += ["## What single signals can tell (AUC: is this edit good?)", "",
                  "`train-clean-100` is the audio our CTC model trained on; the others are audio it never saw.", "",
                  "| Candidates | Dataset | n | Good | CTC evidence | Whisper confidence | LM |", "|---|---|---|---|---|---|---|"]
        for kind in (*SCORED, "duplicate del"):
            for name, row in e.items():
                c = row[kind]
                lines.append(f"| {kind} | {name} | {c['n']:,} | {c['good'] * 100:.0f}% | {c['ctc']:.3f} | {c['whisper']:.3f} | {c['lm']:.3f} |")
        lines.append("")
    h = load("holdout.json")
    if h:
        lines += ["## An error type the network never saw (whole pipeline, dev clips)", "",
                  f"Trained on {' + '.join(HOLDOUT_SPLITS)} only. Each column is the recall of that error type: by the rule, by the "
                  "network trained on all types, and by the network whose training corruptions left that type out.", "",
                  "| Error type | Rule | Network, all types | Network, type left out |", "|---|---|---|---|"]
        for t in ERROR_TYPES:
            lines.append(f"| {t} | {h['rule']['ALL']['recall'].get(t, float('nan')) * 100:.0f} | {h['all_types']['ALL']['recall'].get(t, float('nan')) * 100:.0f} | "
                         f"{h[f'no_{t}']['ALL']['recall'].get(t, float('nan')) * 100:.0f} |")
        lines += ["", "| Network trained without | WER % | Edit F0.5 | Over-corr. % |", "|---|---|---|---|"]
        for name in ("rule", "all_types", *(f"no_{t}" for t in ERROR_TYPES)):
            m = h[name]["ALL"]
            lines.append(f"| {name} | {m['wer'] * 100:.2f} | {m['edit_f05'] * 100:.1f} | {m['overcorrection'] * 100:.2f} |")
        lines.append("")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RES / 'results.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["variants", "pipeline", "evidence", "holdout", "report"],
                    choices=["variants", "pipeline", "evidence", "holdout", "report"])
    args = ap.parse_args()
    for stage in args.stages:
        report() if stage == "report" else globals()[f"stage_{stage}"]()


if __name__ == "__main__":
    main()

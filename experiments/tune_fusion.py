"""Tune the fusion rule on the dev split — imp.md 4.1 ("grid search ... to maximize F0.5").

Stage 1 (fast, default): the fusion candidates (sub / ins / del) of every dev file are featurized
once (ASR cached, LM scored once), under the official corruption and fusion_data.DEV_SEEDS, and
labelled against the reference: a candidate is correct iff applying it lowers the edit distance to
the reference (fusion_data.edit_gains). Position-based labels were wrong for duplicates, because in
"the the" either copy may go: they found 10 good deletions on dev where 107 lower the distance.
A vectorized grid search then maximizes the score the fusion network is selected on, candidate-level
F0.5 (gold edits the candidates cannot reach count as false negatives) minus WER_WEIGHT * WER%, by
coordinate descent over (sub weights + tau_sub), (tau_ins, tau_lm_ins) and (del_span_conf, tau_lm_del).
Each single-signal ablation gets its own re-tuned weights/threshold, and rule_f05 is the rule tuned
on F0.5 alone, as imp.md words it.  -> configs/tuned.yaml, configs/tuned_ablations.yaml

Stage 2 (--pipeline): full-pipeline runs on dev to choose punctuation strategy (Mode A / B) and
the GEC guard ratio.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.fusion_data import DEV_SEEDS, WER_WEIGHT, candidates, edit_gains, norm_words  # noqa: E402
from subfix.asr import transcribe  # noqa: E402
from subfix.config import Config  # noqa: E402
from subfix.models import Models  # noqa: E402
from subfix.paths import CONFIG_DIR, DATA_DIR  # noqa: E402
from subfix.subio import read_srt, tokenize  # noqa: E402
from subfix.utils.cache import load_json, save_json  # noqa: E402

FEAT_CACHE = DATA_DIR / "benchmark" / "fusion_features_dev.json"
# LM-gate thresholds (nats) for insertions / deletions. Wide enough for the n-gram LM, whose
# log-prob gains for correct insertions are lower than distilgpt2's (dev mean -1.6 vs +0.2).
LM_GATE_GRID = np.arange(-12, 3.01, 0.25)


def load_manifest(name: str) -> list[dict]:
    p = DATA_DIR / "benchmark" / f"{name}.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def extract(items: list[dict], cfg: Config, models: Models) -> dict:
    from rapidfuzz.distance import Levenshtein

    from corrupt import corrupt_clip
    rows = {"sub": [], "ins": [], "del": []}
    gold_total = n_ref = 0
    base = np.zeros(3)      # splits and merges keep their rule: right edits, wrong edits, net gain
    lm, spell = models.lm(), models.spell()
    for it in items:
        audio = str(ROOT / it["audio"])
        asr = transcribe(audio, model=cfg.asr.model)
        tiny = [w.text for w in transcribe(audio, model="tiny.en").words]
        ref_cues = read_srt(ROOT / it["ref_srt"])
        ref = [x for c in ref_cues for t in tokenize(c.text.replace("\n", " ")) for x in norm_words(t)]
        versions = [read_srt(ROOT / it["corrupted_srt"])]
        versions += [corrupt_clip(ref_cues, it["id"], it.get("has_punct", False), tiny, seed=seed, rate_scale=scale).cues
                     for seed, scale in DEV_SEEDS]
        for cues in versions:
            sub, _, cands = candidates(cues, asr, cfg, lm, spell)
            gold_total += Levenshtein.distance([x for w in sub for x in norm_words(w.text)], ref)
            n_ref += len(ref)
            for c, gain in zip(cands, edit_gains(cands, sub, asr.words, ref)):
                f = c.feats
                gain = int(np.clip(gain, -5, 5))
                if c.kind == "sub":
                    rows["sub"].append([f["p"], f["phon"], f["lm"], float(f["oov_s"] and not f["oov_a"]),
                                        float(f["event"]), gain])
                elif c.kind == "ins":
                    rows["ins"].append([f["p"], f["lm"], float(f["filler"]), gain])
                elif c.kind == "del":
                    rows["del"].append([f["span_conf"], f["lm"], float(f["dup"]), float(f["event"]), gain])
                elif c.accept:
                    base += [gain > 0, gain <= 0, gain]
        print(f"[features] {it['id']}: {sum(len(r) for r in rows.values())} candidates so far", flush=True)
    return {"rows": rows, "gold_total": gold_total, "n_ref": n_ref, "base": base.tolist()}


def f05(tp: float, fp: float, fn: float) -> float:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return 1.25 * p * r / (0.25 * p + r) if p + r else 0.0


class Tuner:
    def __init__(self, data: dict, clip: float = 5.0, wer_weight: float = WER_WEIGHT):
        self.gold, self.n_ref, self.wer_weight = float(data["gold_total"]), float(data["n_ref"]), wer_weight
        self.base = np.array(data["base"], float)
        S = np.array(data["rows"]["sub"], float).reshape(-1, 6)
        self.sub_X, self.sub_event, self.sub_g = S[:, :4].copy(), S[:, 4], S[:, 5]
        self.sub_X[:, 2] = np.clip(self.sub_X[:, 2], -clip, clip)
        I = np.array(data["rows"]["ins"], float).reshape(-1, 4)
        self.ins_p, self.ins_lm, self.ins_fill, self.ins_g = I[:, 0], I[:, 1], I[:, 2], I[:, 3]
        D = np.array(data["rows"]["del"], float).reshape(-1, 5)
        self.del_span, self.del_lm, self.del_dup, self.del_event, self.del_g = D[:, 0], D[:, 1], D[:, 2], D[:, 3], D[:, 4]

    def wer(self, net: float) -> float:
        return 100 * (self.gold - net) / self.n_ref

    def score(self, counts: np.ndarray) -> float:
        tp, fp, net = counts
        return f05(tp, fp, self.gold - tp) - self.wer_weight * self.wer(net)

    # decisions
    def sub_acc(self, w, tau):
        return ((self.sub_X @ np.asarray(w)) > tau) & (self.sub_event == 0)

    def ins_acc(self, tau_ins, tau_lm, use_lm=True):
        return (self.ins_fill == 0) & (self.ins_p > tau_ins) & ((self.ins_lm > tau_lm) if use_lm else True)

    def del_acc(self, span, tau_lm, use_lm=True):
        return (self.del_event == 0) & ((self.del_dup == 1) | ((self.del_span > span) &
                                                               ((self.del_lm > tau_lm) if use_lm else True)))

    @staticmethod
    def counts(acc, gain) -> np.ndarray:
        """Right edits, wrong edits, net gain."""
        return np.array([(acc & (gain > 0)).sum(), (acc & (gain <= 0)).sum(), gain[acc].sum()], float)

    def total(self, p: dict, use_lm=True) -> dict:
        w = [p["w_conf"], p["w_phon"], p["w_lm"], p["w_oov"]]
        tp, fp, net = (self.base + self.counts(self.sub_acc(w, p["tau_sub"]), self.sub_g)
                       + self.counts(self.ins_acc(p["tau_ins"], p["tau_lm_ins"], use_lm), self.ins_g)
                       + self.counts(self.del_acc(p["del_span_conf"], p["tau_lm_del"], use_lm), self.del_g))
        return {"dev_f05": f05(tp, fp, self.gold - tp), "dev_precision": tp / max(tp + fp, 1), "dev_recall": tp / self.gold,
                "dev_wer": self.wer(net), "dev_tp": int(tp), "dev_fp": int(fp)}

    def tune(self, p: dict, fixed: set[str] = frozenset(), use_lm: bool = True, rounds: int = 2) -> dict:
        p = dict(p)
        grids = {"w_conf": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0], "w_phon": [0.0, 0.25, 0.5, 1.0, 1.5],
                 "w_lm": [0.0, 0.1, 0.2, 0.3, 0.5, 0.8], "w_oov": [0.0, 0.25, 0.5, 1.0, 2.0]}
        taus = np.round(np.arange(-1.0, 6.01, 0.05), 3)
        for _ in range(rounds):
            # substitutions
            b = self.counts(self.ins_acc(p["tau_ins"], p["tau_lm_ins"], use_lm), self.ins_g)
            c = self.counts(self.del_acc(p["del_span_conf"], p["tau_lm_del"], use_lm), self.del_g)
            best = (-np.inf, None)
            axes = [grids[k] if k not in fixed else [p[k]] for k in ("w_conf", "w_phon", "w_lm", "w_oov")]
            for w in itertools.product(*axes):
                if not any(w):
                    continue
                scores = self.sub_X @ np.asarray(w)
                scores = np.where(self.sub_event == 0, scores, -1e9)
                for tau in taus:
                    s = self.score(self.base + self.counts(scores > tau, self.sub_g) + b + c)
                    if s > best[0]:
                        best = (s, (w, float(tau)))
            (p["w_conf"], p["w_phon"], p["w_lm"], p["w_oov"]), p["tau_sub"] = best[1]
            # insertions
            a = self.counts(self.sub_acc([p["w_conf"], p["w_phon"], p["w_lm"], p["w_oov"]], p["tau_sub"]), self.sub_g)
            best = (-np.inf, None)
            for ti in np.arange(0.3, 1.0, 0.05):
                for tl in (LM_GATE_GRID if use_lm else [0.0]):
                    s = self.score(self.base + a + self.counts(self.ins_acc(ti, tl, use_lm), self.ins_g) + c)
                    if s > best[0]:
                        best = (s, (float(ti), float(tl)))
            p["tau_ins"], p["tau_lm_ins"] = best[1]
            # deletions
            b = self.counts(self.ins_acc(p["tau_ins"], p["tau_lm_ins"], use_lm), self.ins_g)
            best = (-np.inf, None)
            for sc in np.arange(0.5, 1.0, 0.02):
                for tl in (LM_GATE_GRID if use_lm else [0.0]):
                    s = self.score(self.base + a + b + self.counts(self.del_acc(sc, tl, use_lm), self.del_g))
                    if s > best[0]:
                        best = (s, (float(sc), float(tl)))
            p["del_span_conf"], p["tau_lm_del"] = best[1]
            c = self.counts(self.del_acc(p["del_span_conf"], p["tau_lm_del"], use_lm), self.del_g)
        p.update(self.total(p, use_lm))
        return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in p.items()}


def stage_features(args) -> None:
    items = load_manifest("dev")
    cfg = Config.load(use_tuned=False)
    models = Models(cfg)
    data = extract(items, cfg, models)
    save_json(FEAT_CACHE, data, indent=None)
    n = {k: len(v) for k, v in data["rows"].items()}
    print(f"[features] saved {n}, gold edits {data['gold_total']}")


def stage_grid(args) -> None:
    data = load_json(FEAT_CACHE)
    tuner = Tuner(data)
    f = Config.load(use_tuned=False).fusion
    start = {"w_conf": f.w_conf, "w_phon": f.w_phon, "w_lm": f.w_lm, "w_oov": f.w_oov, "tau_sub": f.tau_sub,
             "tau_ins": f.tau_ins, "tau_lm_ins": f.tau_lm_ins, "del_span_conf": f.del_span_conf,
             "tau_lm_del": f.tau_lm_del}
    print("[grid] defaults:", tuner.total(start))
    best = tuner.tune(start)
    print("[grid] tuned:", best)
    keys = ["w_conf", "w_phon", "w_lm", "w_oov", "tau_sub", "tau_ins", "tau_lm_ins", "del_span_conf", "tau_lm_del"]
    tuned_path = CONFIG_DIR / "tuned.yaml"
    tuned = yaml.safe_load(tuned_path.read_text(encoding="utf-8")) if tuned_path.exists() else {}
    tuned = tuned or {}
    tuned["fusion"] = {k: best[k] for k in keys}
    header = f"# written by experiments/tune_fusion.py (dev F0.5 = {best['dev_f05']:.4f}, WER = {best['dev_wer']:.2f})\n"
    tuned_path.write_text(header + yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8")
    ablations = {}
    for name, key in (("no_conf", "w_conf"), ("no_phon", "w_phon"), ("no_lm", "w_lm"), ("no_oov", "w_oov")):
        p = dict(best, **{key: 0.0})
        r = tuner.tune(p, fixed={key}, use_lm=(key != "w_lm"))
        print(f"[grid] {name}: dev F0.5 {r['dev_f05']:.4f}, WER {r['dev_wer']:.2f}")
        ablations[name] = {f"fusion.{k}": r[k] for k in keys}
    f05_only = Tuner(data, wer_weight=0.0).tune(start)
    print("[grid] tuned on F0.5 alone:", f05_only)
    ablations["rule_f05"] = {f"fusion.{k}": f05_only[k] for k in keys}
    (CONFIG_DIR / "tuned_ablations.yaml").write_text(
        "# re-tuned fusion parameters for the single-signal ablations (experiments/tune_fusion.py)\n"
        + yaml.safe_dump(ablations, sort_keys=False), encoding="utf-8")
    save_json(ROOT / "results" / "tuning" / "fusion_grid.json", {"defaults": tuner.total(start), "best": best,
                                                                  "f05_only": f05_only, "ablations": ablations})


def stage_pipeline(args) -> None:
    """Pick punctuation strategy (A and B) and GEC guard ratio by full dev runs."""
    from experiments.run_eval import evaluate
    out = ROOT / "results" / "tuning"

    def score(res: dict, mode_b: bool = False) -> float:
        m = res.get("ALL") or next(iter(res.values()))
        pf = m.get("punct_macro_f1", 0.0) or 0.0
        return -m["wer"] + 0.5 * pf * (0.0 if np.isnan(pf) else 1.0)

    tuned_path = CONFIG_DIR / "tuned.yaml"
    tuned = (yaml.safe_load(tuned_path.read_text(encoding="utf-8")) or {}) if tuned_path.exists() else {}
    # the score uses WER and punctuation F1 only, so word timing may come from a cheaper backend
    extra = {"timing.backend": args.timing} if args.timing else {}
    choices = {}
    for mode, key, opts in (("A", "punct.strategy_a", ["fill", "replace"]), ("B", "punct.strategy_b", ["replace", "fill"])):
        best = None
        for opt in opts:
            name = f"tune_punct_{mode}_{opt}"
            sysdef = {"type": "pipeline", "mode": mode, "overrides": {key: opt, "gec.enabled": False, **extra}}
            res = evaluate("dev", {name: sysdef}, None, args.limit, False, args.force, out / name, name)[name]
            s = score(res)
            print(f"[pipeline] {key}={opt}: score {s:.4f}")
            if best is None or s > best[0]:
                best = (s, opt)
        choices[key] = best[1]
    best = None
    for ratio in (0.15, 0.25, 0.4):
        name = f"tune_gec_{ratio}"
        sysdef = {"type": "pipeline", "overrides": {"punct.strategy_a": choices["punct.strategy_a"],
                                                     "gec.max_ratio": ratio, **extra}}
        res = evaluate("dev", {name: sysdef}, None, args.limit, False, args.force, out / name, name)[name]
        s = score(res)
        print(f"[pipeline] gec.max_ratio={ratio}: score {s:.4f}")
        if best is None or s > best[0]:
            best = (s, ratio)
    tuned.setdefault("punct", {})["strategy_a"] = choices["punct.strategy_a"]
    tuned["punct"]["strategy_b"] = choices["punct.strategy_b"]
    tuned.setdefault("gec", {})["max_ratio"] = best[1]
    tuned_path.write_text("# written by experiments/tune_fusion.py\n" + yaml.safe_dump(tuned, sort_keys=False),
                          encoding="utf-8")
    print("[pipeline] tuned:", tuned)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["features", "grid"], choices=["features", "grid", "pipeline"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timing", default=None, help="timing backend for the --pipeline stage (e.g. whisper)")
    args = ap.parse_args()
    for s in args.stages:
        {"features": stage_features, "grid": stage_grid, "pipeline": stage_pipeline}[s](args)


if __name__ == "__main__":
    main()

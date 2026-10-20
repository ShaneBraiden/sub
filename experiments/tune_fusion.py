"""Tune the fusion rule on the dev split — imp.md 4.1 ("grid search ... to maximize F0.5").

Stage 1 (fast, default): for every dev file the fusion candidates (sub / ins / del) are
featurized once (ASR cached, LM scored once) and labelled against the reference:
  sub  s->a  correct iff s is wrong and a equals the reference word at that position
  ins  a     correct iff the reference has a missing word equal to a at that position
  del  s     correct iff s is not in the reference
A vectorized grid search then maximizes candidate-level F0.5 (gold edits the candidates cannot
reach count as false negatives), by coordinate descent over (sub weights + tau_sub),
(tau_ins, tau_lm_ins) and (del_span_conf, tau_lm_del). Each single-signal ablation gets its own
re-tuned weights/threshold.  -> configs/tuned.yaml, configs/tuned_ablations.yaml

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

from subfix import text as T  # noqa: E402
from subfix.align import align_words  # noqa: E402
from subfix.asr import transcribe  # noqa: E402
from subfix.config import Config  # noqa: E402
from subfix.fuse import Fuser  # noqa: E402
from subfix.models import Models  # noqa: E402
from subfix.paths import CONFIG_DIR, DATA_DIR  # noqa: E402
from subfix.subio import Cue, cues_to_words, read_srt, tokenize  # noqa: E402
from subfix.sync import sync_cues  # noqa: E402
from subfix.utils.cache import load_json, save_json  # noqa: E402

FEAT_CACHE = DATA_DIR / "benchmark" / "fusion_features_dev.json"
# LM-gate thresholds (nats) for insertions / deletions. Wide enough for the n-gram LM, whose
# log-prob gains for correct insertions are lower than distilgpt2's (dev mean -1.6 vs +0.2).
LM_GATE_GRID = np.arange(-12, 3.01, 0.25)


def load_manifest(name: str) -> list[dict]:
    p = DATA_DIR / "benchmark" / f"{name}.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def ref_alignment(src: list[str], ref: list[str]):
    """src->ref mapping on canonical tokens: 1:1 pairs, deleted src, inserted ref tokens per position."""
    from rapidfuzz.distance import Levenshtein
    pair, deleted, inserted = {}, set(), {}
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(src, ref):
        if tag == "equal":
            pair.update({i1 + k: j1 + k for k in range(i2 - i1)})
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            pair.update({i1 + k: j1 + k for k in range(n)})
            deleted.update(range(i1 + n, i2))
            if j2 - j1 > n:
                inserted.setdefault(i2, []).extend(ref[j1 + n:j2])
        elif tag == "delete":
            deleted.update(range(i1, i2))
        else:
            inserted.setdefault(i1, []).extend(ref[j1:j2])
    return pair, deleted, inserted


def extract(items: list[dict], cfg: Config, models: Models) -> dict:
    rows = {"sub": [], "ins": [], "del": []}
    gold_total = 0
    lm, spell = models.lm(), models.spell()
    for it in items:
        audio = str(ROOT / it["audio"])
        asr = transcribe(audio, model=cfg.asr.model)
        cues = read_srt(ROOT / it["corrupted_srt"])
        synced, res = sync_cues(cues, asr.words, cfg)
        sub = cues_to_words([Cue(c.index, c.start, c.end, c.text) for c in synced])
        ops = align_words(sub, asr.words, cfg, use_time=res.method != "none")
        fz = Fuser(cfg, lm=lm, spell=spell)
        units = fz.units(ops, sub, asr.words)
        cands = fz.featurize(units, sub, asr.words)
        ref_toks = [t for c in read_srt(ROOT / it["ref_srt"]) for t in tokenize(c.text.replace("\n", " "))]
        src_c = [T.canon(w.text) for w in sub]
        ref_c = [T.canon(t) for t in ref_toks]
        pair, deleted, inserted = ref_alignment(src_c, ref_c)
        from metrics.text_metrics import atomic_edits
        gold_total += sum(atomic_edits(src_c, ref_c).values())
        for c in cands:
            f = c.feats
            if c.kind == "sub":
                r = pair.get(c.i)
                y = int(r is not None and src_c[c.i] != ref_c[r] and T.canon(asr.words[c.j].text) == ref_c[r])
                rows["sub"].append([f["p"], f["phon"], f["lm"], float(f["oov_s"] and not f["oov_a"]),
                                    float(f["event"]), y])
            elif c.kind == "ins":
                y = int(T.canon(asr.words[c.j].text) in inserted.get(c.pos, []))
                rows["ins"].append([f["p"], f["lm"], float(f["filler"]), y])
            elif c.kind == "del":
                y = int(c.i in deleted)
                rows["del"].append([f["span_conf"], f["lm"], float(f["dup"]), float(f["event"]), y])
        print(f"[features] {it['id']}: sync={res.method} cands={len(cands)}", flush=True)
    return {"rows": rows, "gold_total": gold_total}


def f05(tp: float, fp: float, fn: float) -> float:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return 1.25 * p * r / (0.25 * p + r) if p + r else 0.0


def best_threshold(scores: np.ndarray, y: np.ndarray, taus: np.ndarray, rest_tp: float, rest_fp: float,
                   gold: float) -> tuple[float, float, int, int]:
    best = (-1.0, 0.0, 0, 0)
    for tau in taus:
        acc = scores > tau
        tp = int((acc & (y == 1)).sum())
        fp = int((acc & (y == 0)).sum())
        f = f05(rest_tp + tp, rest_fp + fp, gold - rest_tp - tp)
        if f > best[0]:
            best = (f, float(tau), tp, fp)
    return best


class Tuner:
    def __init__(self, data: dict, clip: float = 5.0):
        self.gold = float(data["gold_total"])
        S = np.array(data["rows"]["sub"], float).reshape(-1, 6)
        self.sub_X, self.sub_event, self.sub_y = S[:, :4].copy(), S[:, 4], S[:, 5].astype(int)
        self.sub_X[:, 2] = np.clip(self.sub_X[:, 2], -clip, clip)
        I = np.array(data["rows"]["ins"], float).reshape(-1, 4)
        self.ins_p, self.ins_lm, self.ins_fill, self.ins_y = I[:, 0], I[:, 1], I[:, 2], I[:, 3].astype(int)
        D = np.array(data["rows"]["del"], float).reshape(-1, 5)
        self.del_span, self.del_lm, self.del_dup, self.del_event, self.del_y = (D[:, 0], D[:, 1], D[:, 2], D[:, 3],
                                                                               D[:, 4].astype(int))

    # decisions -> (tp, fp)
    def sub_acc(self, w, tau):
        return ((self.sub_X @ np.asarray(w)) > tau) & (self.sub_event == 0)

    def ins_acc(self, tau_ins, tau_lm, use_lm=True):
        return (self.ins_fill == 0) & (self.ins_p > tau_ins) & ((self.ins_lm > tau_lm) if use_lm else True)

    def del_acc(self, span, tau_lm, use_lm=True):
        return (self.del_event == 0) & ((self.del_dup == 1) | ((self.del_span > span) &
                                                               ((self.del_lm > tau_lm) if use_lm else True)))

    @staticmethod
    def counts(acc, y):
        return int((acc & (y == 1)).sum()), int((acc & (y == 0)).sum())

    def total(self, p: dict, use_lm=True) -> tuple[float, int, int]:
        w = [p["w_conf"], p["w_phon"], p["w_lm"], p["w_oov"]]
        a = self.counts(self.sub_acc(w, p["tau_sub"]), self.sub_y)
        b = self.counts(self.ins_acc(p["tau_ins"], p["tau_lm_ins"], use_lm), self.ins_y)
        c = self.counts(self.del_acc(p["del_span_conf"], p["tau_lm_del"], use_lm), self.del_y)
        tp, fp = a[0] + b[0] + c[0], a[1] + b[1] + c[1]
        return f05(tp, fp, self.gold - tp), tp, fp

    def tune(self, p: dict, fixed: set[str] = frozenset(), use_lm: bool = True, rounds: int = 2) -> dict:
        p = dict(p)
        grids = {"w_conf": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0], "w_phon": [0.0, 0.25, 0.5, 1.0, 1.5],
                 "w_lm": [0.0, 0.1, 0.2, 0.3, 0.5, 0.8], "w_oov": [0.0, 0.25, 0.5, 1.0, 2.0]}
        taus = np.round(np.arange(-1.0, 6.01, 0.05), 3)
        for _ in range(rounds):
            # substitutions
            b = self.counts(self.ins_acc(p["tau_ins"], p["tau_lm_ins"], use_lm), self.ins_y)
            c = self.counts(self.del_acc(p["del_span_conf"], p["tau_lm_del"], use_lm), self.del_y)
            rest_tp, rest_fp = b[0] + c[0], b[1] + c[1]
            best = (-1.0, None)
            axes = [grids[k] if k not in fixed else [p[k]] for k in ("w_conf", "w_phon", "w_lm", "w_oov")]
            for w in itertools.product(*axes):
                if not any(w):
                    continue
                scores = self.sub_X @ np.asarray(w)
                scores = np.where(self.sub_event == 0, scores, -1e9)
                f, tau, _, _ = best_threshold(scores, self.sub_y, taus, rest_tp, rest_fp, self.gold)
                if f > best[0]:
                    best = (f, (w, tau))
            (p["w_conf"], p["w_phon"], p["w_lm"], p["w_oov"]), p["tau_sub"] = best[1]
            # insertions
            a = self.counts(self.sub_acc([p["w_conf"], p["w_phon"], p["w_lm"], p["w_oov"]], p["tau_sub"]), self.sub_y)
            best = (-1.0, None)
            for ti in np.arange(0.3, 1.0, 0.05):
                for tl in (LM_GATE_GRID if use_lm else [0.0]):
                    b = self.counts(self.ins_acc(ti, tl, use_lm), self.ins_y)
                    tp, fp = a[0] + b[0] + c[0], a[1] + b[1] + c[1]
                    f = f05(tp, fp, self.gold - tp)
                    if f > best[0]:
                        best = (f, (float(ti), float(tl)))
            p["tau_ins"], p["tau_lm_ins"] = best[1]
            # deletions
            b = self.counts(self.ins_acc(p["tau_ins"], p["tau_lm_ins"], use_lm), self.ins_y)
            best = (-1.0, None)
            for sc in np.arange(0.5, 1.0, 0.02):
                for tl in (LM_GATE_GRID if use_lm else [0.0]):
                    c2 = self.counts(self.del_acc(sc, tl, use_lm), self.del_y)
                    tp, fp = a[0] + b[0] + c2[0], a[1] + b[1] + c2[1]
                    f = f05(tp, fp, self.gold - tp)
                    if f > best[0]:
                        best = (f, (float(sc), float(tl)))
            p["del_span_conf"], p["tau_lm_del"] = best[1]
            c = self.counts(self.del_acc(p["del_span_conf"], p["tau_lm_del"], use_lm), self.del_y)
        p["dev_f05"], p["dev_tp"], p["dev_fp"] = self.total(p, use_lm)
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
    header = f"# written by experiments/tune_fusion.py (dev F0.5 = {best['dev_f05']:.4f})\n"
    tuned_path.write_text(header + yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8")
    ablations = {}
    for name, key in (("no_conf", "w_conf"), ("no_phon", "w_phon"), ("no_lm", "w_lm"), ("no_oov", "w_oov")):
        p = dict(best, **{key: 0.0})
        r = tuner.tune(p, fixed={key}, use_lm=(key != "w_lm"))
        print(f"[grid] {name}: dev F0.5 {r['dev_f05']:.4f}")
        ablations[name] = {f"fusion.{k}": r[k] for k in keys}
    (CONFIG_DIR / "tuned_ablations.yaml").write_text(
        "# re-tuned fusion parameters for the single-signal ablations (experiments/tune_fusion.py)\n"
        + yaml.safe_dump(ablations, sort_keys=False), encoding="utf-8")
    save_json(ROOT / "results" / "tuning" / "fusion_grid.json", {"defaults": tuner.total(start), "best": best,
                                                                  "ablations": ablations})


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

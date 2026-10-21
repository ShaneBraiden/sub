"""Train and evaluate our sentiment classifier (src/subfix/sentiment.py) — README "Sentiment over time".

Data (`python -I tools/fetch_sentiment.py`): the Stanford Sentiment Treebank (every phrase of its 8,544
training trees; dev and test sentences) and MELD (utterances of the TV series Friends with their times;
9,989 / 1,109 / 2,610). Three classes: negative, neutral, positive.

Model: multinomial logistic regression on TF-IDF unigrams + bigrams of negation-marked tokens, the
regularisation C and the weight of the MELD examples chosen on dev (mean macro-F1 of SST dev and MELD
dev). Compared on test with the majority class, a lexicon baseline (the SST words labelled positive or
negative on their own, counted with a sign flip after a negation) and the same classifier trained on SST
only or on MELD only. SST is also scored the binary way (SST-2: positive vs negative sentences).

Over time: on the MELD test episodes (utterances at their broadcast times), the gold labels (+1 / 0 / -1)
and the predicted scores are smoothed with the app's Gaussian kernel (30 s) and correlated per episode.

  python experiments/train_sentiment.py
Writes models/sentiment/{model.npz, metrics.json}, results/sentiment/{results.md, results.json} and
results/plots/sentiment_meld.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

from subfix.sentiment import CLASSES, SentimentModel, features, mark_negation, smooth, tokenize  # noqa: E402

DATA = ROOT / "data" / "sentiment"
MODEL = ROOT / "models" / "sentiment"
OUT = ROOT / "results" / "sentiment"
GRID_C = [0.5, 1.0, 2.0, 4.0, 8.0]
GRID_MELD_WEIGHT = [1.0, 3.0, 6.0]
TIME_BANDWIDTH = 30.0
SCORE = {"neg": -1.0, "neu": 0.0, "pos": 1.0}


def read(name: str) -> list[dict]:
    path = DATA / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing: python -I tools/fetch_sentiment.py")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ------------------------------------------------------------------ models
class Linear:
    """TF-IDF + logistic regression (sklearn for training; exported to the npz the app reads)."""

    def __init__(self, C: float = 1.0, min_df: int = 2):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(analyzer=features, sublinear_tf=True, min_df=min_df, dtype=np.float32)
        self.clf = LogisticRegression(C=C, max_iter=3000)

    def fit(self, texts: list[str], labels: list[str], weights: np.ndarray | None = None) -> "Linear":
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, labels, sample_weight=weights)
        return self

    def proba(self, texts: list[str]) -> np.ndarray:
        P = self.clf.predict_proba(self.vec.transform(texts))
        return P[:, [list(self.clf.classes_).index(c) for c in CLASSES]]

    def export(self, path: Path) -> None:
        terms = np.empty(len(self.vec.vocabulary_), dtype=object)
        for t, j in self.vec.vocabulary_.items():
            terms[j] = t
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "model.npz", terms=terms.astype(str), idf=self.vec.idf_.astype(np.float32),
                            coef=self.clf.coef_.astype(np.float32), intercept=self.clf.intercept_.astype(np.float32),
                            classes=np.asarray(self.clf.classes_).astype(str))


class Lexicon:
    """Words the SST annotators labelled on their own: +1 positive, -1 negative; flipped after a negation."""

    def __init__(self, phrases: list[dict]):
        self.lex = {}
        for r in phrases:
            toks = tokenize(r["text"])
            if len(toks) == 1 and r["fine"] != 2:
                self.lex[toks[0]] = 1.0 if r["fine"] >= 3 else -1.0

    def proba(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 3))
        for k, text in enumerate(texts):
            s = sum(-self.lex.get(t[4:], 0.0) if t.startswith("NEG_") else self.lex.get(t, 0.0)
                    for t in mark_negation(tokenize(text)))
            out[k, 2 if s > 0 else 0 if s < 0 else 1] = 1.0
        return out


class Majority:
    def __init__(self, label: str):
        self.k = CLASSES.index(label)

    def proba(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 3))
        out[:, self.k] = 1.0
        return out


# ------------------------------------------------------------------ metrics
def scores(rows: list[dict], P: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    gold = [r["label"] for r in rows]
    pred = [CLASSES[k] for k in P.argmax(axis=1)]
    out = {"n": len(rows), "accuracy": accuracy_score(gold, pred),
           "macro_f1": f1_score(gold, pred, labels=list(CLASSES), average="macro", zero_division=0),
           "weighted_f1": f1_score(gold, pred, labels=list(CLASSES), average="weighted", zero_division=0),
           "f1": dict(zip(CLASSES, f1_score(gold, pred, labels=list(CLASSES), average=None, zero_division=0).tolist())),
           "confusion": confusion_matrix(gold, pred, labels=list(CLASSES)).tolist()}
    polar = [k for k, r in enumerate(rows) if r["label"] != "neu"]
    if polar:
        bp = ["pos" if P[k, 2] >= P[k, 0] else "neg" for k in polar]
        out["binary_accuracy"] = float(np.mean([b == rows[k]["label"] for b, k in zip(bp, polar)]))
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def over_time(rows: list[dict], P: np.ndarray, h: float = TIME_BANDWIDTH, min_utts: int = 10) -> dict:
    """Pearson r of the smoothed gold and predicted curves per MELD episode (utterance midpoints), and of
    the raw utterance scores over all utterances."""
    pred = P[:, 2] - P[:, 0]
    gold = np.asarray([SCORE[r["label"]] for r in rows])
    episodes: dict[tuple, list[int]] = {}
    for k, r in enumerate(rows):
        episodes.setdefault((r["season"], r["episode"]), []).append(k)
    rs = []
    for ix in episodes.values():
        if len(ix) < min_utts:
            continue
        ix = sorted(ix, key=lambda k: rows[k]["start"])
        mids = np.asarray([(rows[k]["start"] + rows[k]["end"]) / 2 for k in ix])
        w = np.asarray([max(len(rows[k]["text"].split()), 1) for k in ix], dtype=float)
        g = np.asarray(smooth(mids, mids, gold[ix], w, h), dtype=float)
        p = np.asarray(smooth(mids, mids, pred[ix], w, h), dtype=float)
        if g.std() > 1e-9 and p.std() > 1e-9:
            rs.append(float(np.corrcoef(g, p)[0, 1]))
    return {"episodes": len(rs), "curve_r": float(np.mean(rs)) if rs else float("nan"),
            "curve_r_median": float(np.median(rs)) if rs else float("nan"),
            "utterance_r": float(np.corrcoef(gold, pred)[0, 1]) if pred.std() > 0 else float("nan")}


def evaluate(model, sst: list[dict], meld: list[dict]) -> dict:
    Ps, Pm = model.proba([r["text"] for r in sst]), model.proba([r["text"] for r in meld])
    return {"sst": scores(sst, Ps), "meld": scores(meld, Pm), "time": over_time(meld, Pm)}


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="one setting (C=2, MELD weight 3), no grid")
    args = ap.parse_args()
    t0 = time.time()
    phrases, sst_dev, sst_test = read("sst/train_phrases"), read("sst/dev"), read("sst/test")
    meld_train, meld_dev, meld_test = read("meld/train"), read("meld/dev"), read("meld/test")
    texts = [r["text"] for r in phrases] + [r["text"] for r in meld_train]
    labels = [r["label"] for r in phrases] + [r["label"] for r in meld_train]
    is_meld = np.r_[np.zeros(len(phrases)), np.ones(len(meld_train))].astype(bool)

    grid = [(2.0, 3.0)] if args.quick else list(product(GRID_C, GRID_MELD_WEIGHT))
    tuning = []
    for C, wm in grid:
        m = Linear(C).fit(texts, labels, np.where(is_meld, wm, 1.0))
        dev = evaluate(m, sst_dev, meld_dev)
        obj = (dev["sst"]["macro_f1"] + dev["meld"]["macro_f1"]) / 2
        tuning.append({"C": C, "meld_weight": wm, "objective": obj, "sst_dev_macro_f1": dev["sst"]["macro_f1"],
                       "meld_dev_macro_f1": dev["meld"]["macro_f1"]})
        print(f"C={C:<4} MELD weight={wm:<3} dev: SST macro-F1 {dev['sst']['macro_f1']:.3f}, "
              f"MELD macro-F1 {dev['meld']['macro_f1']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    best = max(tuning, key=lambda r: r["objective"])
    ours = Linear(best["C"]).fit(texts, labels, np.where(is_meld, best["meld_weight"], 1.0))
    ours.export(MODEL)
    app = SentimentModel(MODEL)                       # the exported model must give the same probabilities
    sample = [r["text"] for r in meld_test[:300] + sst_test[:300]]
    drift = float(np.abs(app.proba(sample) - ours.proba(sample)).max())
    assert drift < 1e-4, f"exported model differs from sklearn by {drift}"

    systems = {
        "majority": Majority("neu"),
        "lexicon": Lexicon(phrases),
        "sst_only": Linear(best["C"]).fit([r["text"] for r in phrases], [r["label"] for r in phrases]),
        "meld_only": Linear(best["C"]).fit([r["text"] for r in meld_train], [r["label"] for r in meld_train]),
        "ours": ours,
    }
    results = {name: evaluate(m, sst_test, meld_test) for name, m in systems.items()}
    top = {}
    coef = ours.clf.coef_
    vocab = np.empty(len(ours.vec.vocabulary_), dtype=object)
    for t, j in ours.vec.vocabulary_.items():
        vocab[j] = t
    for k, c in enumerate(ours.clf.classes_):
        top[str(c)] = [str(vocab[j]) for j in np.argsort(-coef[k])[:15]]
    metrics = {"C": best["C"], "meld_weight": best["meld_weight"], "vocabulary": len(vocab),
               "train_examples": len(texts), "test": results["ours"], "export_max_diff": drift,
               "seconds": round(time.time() - t0, 1)}
    (MODEL / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"tuning": tuning, "best": best, "results": results, "top_features": top,
            "data": {"sst_train_phrases": len(phrases), "sst_test": len(sst_test), "meld_train": len(meld_train),
                     "meld_test": len(meld_test)}}
    (OUT / "results.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    write_markdown(data)
    plot_episode(meld_test, ours.proba([r["text"] for r in meld_test]))
    print(json.dumps({k: {"sst_acc": round(v["sst"]["accuracy"], 3), "sst2_acc": round(v["sst"].get("binary_accuracy", 0), 3),
                          "meld_wf1": round(v["meld"]["weighted_f1"], 3), "meld_mf1": round(v["meld"]["macro_f1"], 3),
                          "curve_r": round(v["time"]["curve_r"], 3)} for k, v in results.items()}, indent=1))
    return 0


LABEL = {"majority": "Majority class (neutral)", "lexicon": "SST word lexicon",
         "sst_only": "Ours, trained on SST only", "meld_only": "Ours, trained on MELD only",
         "ours": "Ours, SST + MELD (shipped)"}


def write_markdown(data: dict) -> None:
    R = data["results"]
    b = data["best"]
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    L = ["# Sentiment classifier", "",
         f"Logistic regression on TF-IDF unigrams + bigrams with negation marking; C = {b['C']}, MELD examples "
         f"weighted x{b['meld_weight']:g} (chosen on dev). Trained on {data['data']['sst_train_phrases']:,} SST "
         f"phrases + {data['data']['meld_train']:,} MELD utterances. Three classes.", "",
         "| System | SST-3 accuracy | SST-3 macro-F1 | SST-2 accuracy | MELD accuracy | MELD weighted-F1 | "
         "MELD macro-F1 | curve r (MELD episodes) |", "|---|---|---|---|---|---|---|---|"]
    for name, r in R.items():
        s, m, t = r["sst"], r["meld"], r["time"]
        curve = f"{t['curve_r']:.2f}" if t["curve_r"] == t["curve_r"] else "–"     # a constant prediction has no r
        L.append(f"| {LABEL[name]} | {pct(s['accuracy'])} | {pct(s['macro_f1'])} | "
                 f"{pct(s.get('binary_accuracy', float('nan')))} | {pct(m['accuracy'])} | {pct(m['weighted_f1'])} | "
                 f"{pct(m['macro_f1'])} | {curve} |")
    o = R["ours"]
    L += ["", f"SST test: {o['sst']['n']:,} sentences; MELD test: {o['meld']['n']:,} utterances. SST-2 = the "
          "positive and negative SST test sentences, labelled by whichever of P(positive), P(negative) is higher. "
          f"Curve r = Pearson correlation of the smoothed gold and predicted sentiment ({TIME_BANDWIDTH:.0f} s "
          f"Gaussian, as in the app) over the utterances of each MELD test episode, averaged over "
          f"{o['time']['episodes']} episodes; utterance-level r = {o['time']['utterance_r']:.2f}.", "",
          "Per-class F1 of the shipped model:", "",
          "| | negative | neutral | positive |", "|---|---|---|---|"]
    for ds in ("sst", "meld"):
        f = o[ds]["f1"]
        L.append(f"| {ds.upper()} | {pct(f['neg'])} | {pct(f['neu'])} | {pct(f['pos'])} |")
    L += ["", "Confusion matrix on MELD test (rows gold, columns predicted; neg / neu / pos):", ""]
    for row, name in zip(o["meld"]["confusion"], ("neg", "neu", "pos")):
        L.append(f"- {name}: {row}")
    L += ["", "Strongest features per class:", ""]
    for c, feats in data["top_features"].items():
        L.append(f"- {c}: {', '.join(feats)}")
    L += ["", "![MELD test episode](../plots/sentiment_meld.png)", ""]
    (OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def plot_episode(rows: list[dict], P: np.ndarray) -> None:
    """The MELD test episode with the most utterances: gold vs predicted sentiment, smoothed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from vizstyle import CATEGORICAL, INK, INK_2, SURFACE, style_axes
    eps: dict[tuple, list[int]] = {}
    for k, r in enumerate(rows):
        eps.setdefault((r["season"], r["episode"]), []).append(k)
    key, ix = max(eps.items(), key=lambda kv: len(kv[1]))
    ix = sorted(ix, key=lambda k: rows[k]["start"])
    mids = np.asarray([(rows[k]["start"] + rows[k]["end"]) / 2 for k in ix])
    w = np.asarray([max(len(rows[k]["text"].split()), 1) for k in ix], dtype=float)
    gold = np.asarray([SCORE[rows[k]["label"]] for k in ix])
    pred = P[ix, 2] - P[ix, 0]
    grid = np.linspace(mids.min() - 20, mids.max() + 20, 600)
    g = np.asarray([np.nan if v is None else v for v in smooth(grid, mids, gold, w, TIME_BANDWIDTH)])
    p = np.asarray([np.nan if v is None else v for v in smooth(grid, mids, pred, w, TIME_BANDWIDTH)])
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150, facecolor=SURFACE)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.plot(grid, g, color=CATEGORICAL[1], lw=2, label="gold (annotators)")
    ax.plot(grid, p, color=CATEGORICAL[0], lw=2, label="ours (predicted)")
    ax.scatter(mids, pred, s=9, color=CATEGORICAL[0], alpha=0.35, linewidths=0)
    r = float(np.corrcoef(np.nan_to_num(g), np.nan_to_num(p))[0, 1])
    ax.set_title(f"MELD test, season {key[0]} episode {key[1]}: {len(ix)} utterances, curve r = {r:.2f}",
                 fontsize=10, color=INK, loc="left")
    ax.set_ylabel("positive - negative", fontsize=8, color=INK_2)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", ncol=2, labelcolor=INK_2)

    def clock(v, _):
        s = int(max(v, 0))
        return f"{s // 60}:{s % 60:02d}"
    ax.xaxis.set_major_formatter(FuncFormatter(clock))
    style_axes(ax)
    fig.tight_layout()
    path = ROOT / "results" / "plots" / "sentiment_meld.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

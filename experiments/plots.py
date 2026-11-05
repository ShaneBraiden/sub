"""Ablation plots (imp.md 6.3 / 14.2): reads results/ablations/*/metrics.json -> results/plots/*.png"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / "results" / "ablations"
OUT = ROOT / "results" / "plots"
COLORS = ["#2a6fdb", "#e07a1f", "#2e9e5b", "#c23b5a", "#7b5cc4", "#6b7785"]


def load(name: str) -> dict | None:
    p = ABL / name / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pick(res: dict, metric: str, dataset: str = "ALL") -> dict[str, float]:
    out = {}
    for sysname, by_ds in res.items():
        m = by_ds.get(dataset) or next(iter(by_ds.values()))
        v = m.get(metric)
        label = sysname
        for abl in ("fusion_signals", "fusion_method", "gec_model", "asr_size", "timing_source", "segmenter",
                    "audio_guard"):
            if sysname.startswith(f"abl_{abl}_"):
                label = sysname[len(f"abl_{abl}_"):]
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            out[label] = v
    return out


def bars(ax, data: dict[str, float], title: str, ylabel: str, scale: float = 1.0, fmt: str = "{:.1f}"):
    names = list(data)
    vals = [data[n] * scale for n in names]
    b = ax.bar(names, vals, color=COLORS[: len(names)])
    for rect, v in zip(b, vals):
        ax.annotate(fmt.format(v), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    r = load("timing_source")
    if r:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        bars(axes[0], pick(r, "w_boundary_mean_ms"), "Word boundary error vs MFA", "mean |Δt| (ms)", fmt="{:.0f}")
        bars(axes[1], pick(r, "t_within_250"), "Cue onsets within ±250 ms", "%", 100)
        fig.tight_layout()
        fig.savefig(OUT / "timing_source.png", dpi=150)
        made.append("timing_source.png")
    r = load("gec_model")
    if r:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        bars(axes[0], pick(r, "wer"), "WER with / without our corrector", "WER (%)", 100, "{:.2f}")
        bars(axes[1], pick(r, "overcorrection"), "Over-correction rate", "% correct words changed", 100, "{:.2f}")
        fig.tight_layout()
        fig.savefig(OUT / "gec_model.png", dpi=150)
        made.append("gec_model.png")
    for name, metric, title, ylabel, scale, fmt in (
            ("asr_size", "wer", "WER by ASR size", "WER (%)", 100, "{:.2f}"),
            ("fusion_signals", "edit_f05", "Edit F0.5 without each fusion signal", "F0.5 (%)", 100, "{:.1f}"),
            ("fusion_method", "wer", "WER by who decides each edit", "WER (%)", 100, "{:.2f}"),
            ("segmenter", "fmt_all_ok", "Cues meeting all format rules", "%", 100, "{:.1f}"),
            ("audio_guard", "overcorrection", "Over-correction with/without audio guard", "% correct words changed",
             100, "{:.2f}")):
        r = load(name)
        if r:
            fig, ax = plt.subplots(figsize=(5, 3.4))
            bars(ax, pick(r, metric), title, ylabel, scale, fmt)
            fig.tight_layout()
            fig.savefig(OUT / f"{name}.png", dpi=150)
            made.append(f"{name}.png")
    variants = ROOT / "results" / "fusion" / "variants.json"
    if variants.exists():
        v = json.loads(variants.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        for k, (name, label) in enumerate((("bilstm", "BiLSTM (shipped)"), ("mlp", "MLP, no context"),
                                           ("logistic", "logistic regression"), ("no_ctc", "BiLSTM, no CTC evidence"))):
            curve = {float(c) * 100: e["mean"] for c, e in v["variants"][name]["wer_at_precision"].items() if e}
            ax.plot(list(curve), list(curve.values()), "o-", color=COLORS[k], label=label)
        rule = v["rule"]["all"]
        ax.plot([rule["precision"] * 100], [rule["wer"]], "s", color=COLORS[5], label="linear rule")
        ax.set_xlabel("precision of the accepted edits (%)")
        ax.set_ylabel("WER after fusion (%), dev clips")
        ax.set_title("What each fusion model can reach", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / "fusion_tradeoff.png", dpi=150)
        made.append("fusion_tradeoff.png")
    rates = sorted(ABL.glob("corruption_rate_*/metrics.json"))
    if rates:
        xs, noop, ours, rule = [], [], [], []
        for p in rates:
            res = json.loads(p.read_text(encoding="utf-8"))
            xs.append(float(p.parent.name.split("_")[-1]) * 100)
            noop.append(pick(res, "wer").get("noop", float("nan")) * 100)
            ours.append(pick(res, "wer").get("ours", float("nan")) * 100)
            rule.append(pick(res, "wer").get("ours_rule", float("nan")) * 100)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot(xs, noop, "o-", color=COLORS[1], label="corrupted input")
        ax.plot(xs, rule, "o-", color=COLORS[2], label="ours, linear rule")
        ax.plot(xs, ours, "o-", color=COLORS[0], label="ours, network")
        ax.set_xlabel("injected word error rate (%)")
        ax.set_ylabel("WER (%)")
        ax.set_title("Corruption-rate sweep", fontsize=10)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / "corruption_rate.png", dpi=150)
        made.append("corruption_rate.png")
    print("plots:", made or "none (run the ablations first)")


if __name__ == "__main__":
    sys.exit(main())

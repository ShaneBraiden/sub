"""Static figures for the PDF (matplotlib, light theme, print sizes).

Colour roles (validated reference palette, light mode): series-1 blue = our models,
series-2 orange = the old pretrained model / baseline. Text never uses series colours.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OURS = "#2a78d6"
OLD = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"


def _style() -> None:
    family = "Segoe UI" if any("Segoe UI" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
    })


def _save(fig, out: Path) -> Path:
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def _fmt_params(n: float) -> str:
    return f"{n / 1e6:.0f}M" if n >= 1e7 else f"{n / 1e6:.1f}M"


def params_compare(f: dict, out: Path) -> Path:
    rows = [(m["role"], m["old_params"], f[m["new_key"]]["params"]) for m in f["old_models"]
            if m["old_params"] and f.get(m["new_key"]) and f[m["new_key"]].get("params")]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    y = range(len(rows))
    h = 0.36
    ax.barh([i - h / 2 - 0.02 for i in y], [r[1] for r in rows], height=h, color=OLD, label="Pretrained (before)")
    ax.barh([i + h / 2 + 0.02 for i in y], [r[2] for r in rows], height=h, color=OURS, label="Ours, from scratch")
    for i, (_, old, new) in enumerate(rows):
        ax.text(old * 1.08, i - h / 2 - 0.02, _fmt_params(old), va="center", fontsize=8, color=INK2)
        ax.text(new * 1.08, i + h / 2 + 0.02, f"{_fmt_params(new)}  ({old / new:.0f}x smaller)", va="center",
                fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1e6, 4e9)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_title("Model size", pad=14)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
    return _save(fig, out)


def corrector_curve(f: dict, out: Path) -> Path:
    h = f["corrector"]["metrics"]["history"]
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.plot(ep, [100 * r["wer_model"] for r in h], color=OURS, lw=2, marker="o", ms=4, label="Corrector output")
    ax.axhline(100 * h[0]["wer_copy"], color=OLD, lw=2, ls=(0, (4, 3)), label="Input (no correction)")
    best = min(h, key=lambda r: r["wer_model"])
    ax.annotate(f"best {100 * best['wer_model']:.2f}% (epoch {best['epoch']})", (best["epoch"], 100 * best["wer_model"]),
                xytext=(0, 14), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.text(ep[-1], 100 * h[0]["wer_copy"] + 0.4, f"{100 * h[0]['wer_copy']:.2f}%", ha="right", fontsize=8, color=INK2)
    ax.set_ylim(0, 100 * h[0]["wer_copy"] + 2.5)
    ax.set_xticks(ep)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("WER on noisy dev sentences (%)")
    ax.set_title("Corrector: word error rate per epoch", pad=14)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
    return _save(fig, out)


def punct_curve(f: dict, out: Path) -> Path:
    h = f["punct"]["metrics"]["history"]
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(figsize=(3.25, 2.6))
    ax.plot(ep, [r["cased"]["macro"] for r in h], color=OURS, lw=2, marker="o", ms=4, label="Cased input")
    ax.plot(ep, [r["lower"]["macro"] for r in h], color=OLD, lw=2, marker="s", ms=4, label="Lower-cased input")
    ax.set_ylim(0.5, 0.9)
    ax.set_xticks(ep)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev macro-F1")
    ax.set_title("Punctuation tagger per epoch")
    ax.legend(loc="lower right")
    return _save(fig, out)


def punct_classes(f: dict, out: Path) -> Path:
    best = f["punct"]["metrics"]["best"]
    names = [".", ",", "?"]
    labels = ["Full stop", "Comma", "Question"]
    fig, ax = plt.subplots(figsize=(3.25, 2.6))
    x = range(len(names))
    w = 0.36
    b1 = ax.bar([i - w / 2 - 0.02 for i in x], [best["cased"][n] for n in names], w, color=OURS, label="Cased")
    b2 = ax.bar([i + w / 2 + 0.02 for i in x], [best["lower"][n] for n in names], w, color=OLD, label="Lower-cased")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}", ha="center",
                    fontsize=7.5, color=INK)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Dev F1")
    ax.set_title("F1 per punctuation mark")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncols=2)
    return _save(fig, out)


def pos_recall(f: dict, out: Path) -> Path:
    rec = f["pos"]["per_tag_recall"]
    items = sorted(rec.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.barh([k for k, _ in items], [v for _, v in items], color=OURS, height=0.62)
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Held-out recall")
    ax.set_title(f"POS tagger: recall per tag (overall accuracy {100 * f['pos']['heldout_accuracy']:.2f}%)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7.5)
    return _save(fig, out)


def lm_orders(f: dict, out: Path) -> Path:
    lm = f["lm"]
    k = list(range(1, lm["order"] + 1))
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    bars = ax.bar([str(i) for i in k], [lm[f"ngrams_{i}"] for i in k], color=OURS, width=0.55)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02, f"{b.get_height():,.0f}", ha="center",
                fontsize=8, color=INK)
    ax.set_ylim(0, max(lm[f"ngrams_{i}"] for i in k) * 1.18)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M"))
    ax.set_xlabel("n-gram order")
    ax.set_ylabel("Distinct n-grams stored")
    ax.set_title(f"Kneser-Ney language model: {lm['entries']:,} n-gram entries")
    ax.grid(axis="x", visible=False)
    return _save(fig, out)


def aligner_curves(f: dict, out_cer: Path, out_onset: Path) -> tuple[Path, Path] | None:
    m = f["aligner"].get("metrics")
    if not m or not m.get("history"):
        return None
    h = m["history"]
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(figsize=(3.25, 2.6))
    ax.plot(ep, [100 * r["dev_cer"] for r in h], color=OURS, lw=2, marker="o", ms=3.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Greedy CER on dev-clean (%)")
    ax.set_title("Acoustic model: character error")
    ax.set_ylim(0, None)
    _save(fig, out_cer)
    fig, ax = plt.subplots(figsize=(3.25, 2.6))
    ax.plot(ep, [r["onset_mean_abs_ms"] for r in h], color=OURS, lw=2, marker="o", ms=3.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean |onset error| vs MFA (ms)")
    ax.set_title("Word-start timing error")
    ax.set_ylim(0, None)
    _save(fig, out_onset)
    return out_cer, out_onset


def results_compare(f: dict, out: Path) -> Path | None:
    old, new = f["results"]["old"], f["results"]["new"]
    if not old or not new:
        return None

    def ours(table):
        return next((v for k, v in table["A"].items() if k.startswith("Ours") and not k.startswith("Ours +")), None)

    o, n = ours(old), ours(new)
    if not o or not n:
        return None
    metrics = [("WER", "WER (%)", False), ("ERRANT F0.5", "ERRANT F0.5", True), ("Punct F1", "Punctuation F1", True),
               ("<=250 ms", "Onsets within 250 ms (%)", True), ("Onset ms", "Mean onset error (ms)", False)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.6, 2.4))
    for ax, (key, label, higher) in zip(axes, metrics):
        vals = [o.get(key) or 0, n.get(key) or 0]
        bars = ax.bar(["Before", "Ours"], vals, color=[OLD, OURS], width=0.6)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02, f"{b.get_height():g}", ha="center",
                    fontsize=7.5, color=INK)
        ax.set_title(label, fontsize=8, loc="center")
        ax.set_ylim(0, max(vals) * 1.25 or 1)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.text(0.5, -0.32, "higher is better" if higher else "lower is better", transform=ax.transAxes,
                ha="center", fontsize=7, color=INK2)
    fig.suptitle("Mode A test results: pretrained system vs our models", x=0.01, ha="left", fontsize=10,
                 fontweight="semibold")
    return _save(fig, out)


def build_all(f: dict, out_dir: Path) -> dict[str, Path]:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Path] = {}
    if all(f.get(k) for k in ("punct", "corrector", "lm")):
        figs["params"] = params_compare(f, out_dir / "params.png")
    if f.get("corrector"):
        figs["corrector"] = corrector_curve(f, out_dir / "corrector_curve.png")
    if f.get("punct"):
        figs["punct_curve"] = punct_curve(f, out_dir / "punct_curve.png")
        figs["punct_classes"] = punct_classes(f, out_dir / "punct_classes.png")
    if f.get("pos"):
        figs["pos"] = pos_recall(f, out_dir / "pos_recall.png")
    if f.get("lm"):
        figs["lm"] = lm_orders(f, out_dir / "lm_orders.png")
    al = aligner_curves(f, out_dir / "aligner_cer.png", out_dir / "aligner_onset.png")
    if al:
        figs["aligner_cer"], figs["aligner_onset"] = al
    rc = results_compare(f, out_dir / "results.png")
    if rc:
        figs["results"] = rc
    return figs

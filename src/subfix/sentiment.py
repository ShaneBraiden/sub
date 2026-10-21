"""Sentiment over time — the fourth output (README "Sentiment over time").

Our own classifier (experiments/train_sentiment.py; nothing pretrained): multinomial logistic
regression on TF-IDF word unigrams and bigrams, with the scope of a negation marked ("not very good"
-> not NEG_very NEG_good), trained on every phrase of the Stanford Sentiment Treebank (written movie
reviews) and on MELD (spoken dialogue from the TV series Friends). Three classes; the score of a unit
is P(positive) - P(negative), from -1 to +1.

Units are the sentences of the corrected subtitles (`unit: sentence`) or their cues (`unit: cue`), or
those of any .srt/.vtt file (`subfix sentiment`; speaker labels such as "SPEAKER_00:", "JOHN:" or
WebVTT voice tags are kept). The curve over time is a Gaussian-weighted average of the unit scores
(weighted by their words) on a regular grid; the bandwidth adapts to the length of the recording
(`bandwidth: 0`) or is given in seconds. With speaker labels every speaker gets a curve, with chapter
markers every chapter its mean, and the strongest positive and negative moments are the extremes of
the curve.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import text as T

CLASSES = ("neg", "neu", "pos")
CLASS_NAME = {"neg": "negative", "neu": "neutral", "pos": "positive"}
NEGATIONS = frozenset({"not", "n't", "no", "never", "nothing", "nobody", "none", "neither", "nor", "nowhere",
                       "cannot", "without", "hardly", "barely"})
NEG_SCOPE = 4                     # words after a negation that are marked
_CLITIC = re.compile(r"(\w)'(s|re|ve|ll|d|m)\b")
_NT = re.compile(r"(\w)n't\b")
_TOKEN = re.compile(r"n't|'(?:s|re|ve|ll|d|m)\b|[a-z0-9]+(?:[-.][a-z0-9]+)*|[!?]+|[.,;:]")
_STOP_SCOPE = frozenset(".,;:!?")


# ---------------------------------------------------------------- features (shared with training)
def tokenize(text: str) -> list[str]:
    """Lowercased tokens split like the Penn Treebank ("don't" -> do n't, "it's" -> it 's)."""
    t = T.canon_quotes(text).lower()
    t = _NT.sub(r"\1 n't", t)
    t = _CLITIC.sub(r"\1 '\2", t)
    return _TOKEN.findall(t)


def mark_negation(tokens: list[str]) -> list[str]:
    out, left = [], 0
    for tok in tokens:
        if tok[0] in _STOP_SCOPE:
            left = 0
            out.append(tok)
        elif left > 0:
            out.append("NEG_" + tok)
            left -= 1
        else:
            out.append(tok)
        if tok in NEGATIONS:
            left = NEG_SCOPE
    return out


def features(text: str) -> list[str]:
    """Unigrams and bigrams of the negation-marked tokens (the analyzer of the training vectorizer)."""
    toks = mark_negation(tokenize(text))
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


class SentimentModel:
    """The trained classifier: TF-IDF (sublinear tf, L2) -> softmax(coef x + intercept)."""

    def __init__(self, path: str | Path):
        p = Path(path)
        z = np.load(p / "model.npz" if p.is_dir() else p, allow_pickle=False)
        self.terms = z["terms"].tolist()
        self.index = {t: i for i, t in enumerate(self.terms)}
        self.idf = z["idf"].astype(np.float64)
        self.coef = z["coef"].astype(np.float64)            # classes x V
        self.intercept = z["intercept"].astype(np.float64)
        self.classes = [str(c) for c in z["classes"]]
        order = [self.classes.index(c) for c in CLASSES]    # columns as neg, neu, pos
        self.coef, self.intercept = self.coef[order], self.intercept[order]

    def proba(self, texts: list[str]) -> np.ndarray:
        """(N, 3) class probabilities in the order neg, neu, pos."""
        out = np.zeros((len(texts), len(CLASSES)))
        for r, text in enumerate(texts):
            counts = Counter(j for j in (self.index.get(f) for f in features(text)) if j is not None)
            logits = self.intercept.copy()
            if counts:
                js = np.fromiter(counts, dtype=np.int64)
                vals = (1.0 + np.log(np.fromiter(counts.values(), dtype=np.float64))) * self.idf[js]
                vals /= max(np.linalg.norm(vals), 1e-12)
                logits += self.coef[:, js] @ vals
            e = np.exp(logits - logits.max())
            out[r] = e / e.sum()
        return out


# ---------------------------------------------------------------- units
@dataclass
class Unit:
    start: float
    end: float
    text: str
    speaker: str | None = None

    @property
    def words(self) -> int:
        return max(len(self.text.split()), 1)


def units_from_words(words, max_words: int = 40) -> list[Unit]:
    """Sentences of timed words (the summary's sentence splitter: punctuation, speaker changes, pauses)."""
    from .summarize import make_sentences
    return [Unit(s.start, s.end, s.text, s.speaker) for s in make_sentences(words, max_words)]


def labeled_words(cues) -> list:
    """The words of subtitle cues with times interpolated inside each cue, and the speaker of their line
    in meta["speaker"] (None without labels)."""
    from .subio import cue_words
    words = []
    for pos, c in enumerate(cues):
        for w in cue_words(c, pos):
            line = w.meta["line"]
            w.meta["speaker"] = c.speakers[line] if c.speakers and line < len(c.speakers) else None
            words.append(w)
    return words


def units_from_cues(cues, unit: str = "cue") -> list[Unit]:
    """One unit per cue (per line when a two-speaker cue gives each line a speaker), or, with
    unit="sentence", the sentences of the cue words (times interpolated inside each cue)."""
    from .subio import spoken_text
    if unit == "sentence":
        return units_from_words(labeled_words(cues))
    out = []
    for c in cues:
        lines = [ln.strip() for ln in spoken_text(c).split("\n") if ln.strip()]
        spk = list(c.speakers) if c.speakers else []
        if len(lines) > 1 and len(spk) == len(lines) and len(set(spk)) > 1:
            total = sum(len(ln) for ln in lines)
            t = c.start
            for ln, s in zip(lines, spk):
                d = (c.end - c.start) * len(ln) / max(total, 1)
                out.append(Unit(t, t + d, ln, s))
                t += d
        else:
            out.append(Unit(c.start, c.end, " ".join(lines), spk[0] if spk else None))
    return [u for u in out if u.text]


# speaker labels at the start of a subtitle line: our own (SPEAKER_00:), ALL-CAPS names (JOHN:) and
# Title-case names (Interviewer:) - the latter only when the same label opens two or more lines
_LABEL = re.compile(r"^\s*[\[(]?(?P<name>SPEAKER_\d+|[A-Z][A-Z0-9.'-]*(?: [A-Z][A-Z0-9.'-]*){0,2}|"
                    r"[A-Z][a-z]+(?: [A-Z][a-z]+){0,2})[\])]?:\s+(?P<rest>\S.*)$")
_VOICE = re.compile(r"<v(?:\.[\w.-]+)?\s+([^>]+)>")


def read_labeled_subs(path: str | Path) -> list:
    """Cues of an .srt/.vtt file with the speaker of every line (cue.speakers) when the file labels them."""
    from .subio import _TAG, Cue, _finish, clean_text
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    if str(path).lower().endswith(".vtt"):
        raw = "\n\n".join(b for b in re.split(r"\n\s*\n", raw)
                          if "-->" in b or not b.lstrip().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")))
    blocks = []
    for start, end, body_text in _parse_blocks_raw(raw):
        lines = []
        for line in body_text.split("\n"):
            voice = _VOICE.search(line)
            plain = re.sub(r"^\s*[-–—]\s+", "", _TAG.sub("", _VOICE.sub("", line))).strip()
            m = None if voice else _LABEL.match(plain)
            name = voice.group(1).strip() if voice else (m.group("name") if m else None)
            lines.append((name, clean_text(m.group("rest") if m else plain), m is not None))
        blocks.append((start, end, [ln for ln in lines if ln[1]]))
    counts = Counter(n for _, _, lines in blocks for n, _, from_label in lines if n and from_label)
    accept = {n for n, k in counts.items() if n.startswith("SPEAKER_") or n.isupper() or k >= 2}
    cues = []
    for start, end, lines in blocks:
        if not lines:
            continue
        spk, texts = [], []
        for name, body, from_label in lines:
            if from_label and name not in accept:          # "Note: ..." is text, not a speaker
                body, name = f"{name}: {body}", None
            spk.append(name)
            texts.append(body)
        cue = Cue(0, start, end, "\n".join(texts))
        if any(spk):
            cue.speakers = spk
        cues.append(cue)
    return _finish(cues)


def _parse_blocks_raw(content: str) -> list[tuple[float, float, str]]:
    from .subio import _TIMING_LINE, _parse_time
    out = []
    for block in re.split(r"\n\s*\n", content.lstrip("﻿")):
        lines = block.split("\n")
        for k, line in enumerate(lines):
            m = _TIMING_LINE.match(line)
            if m:
                out.append((_parse_time(m.group(1)), _parse_time(m.group(3)), "\n".join(lines[k + 1:])))
                break
    return out


# ---------------------------------------------------------------- the track over time
@dataclass
class SentimentTrack:
    units: list[dict]
    times: list[float]
    curve: list[float | None]
    bandwidth: float
    duration: float
    totals: dict
    speakers: list[dict] = field(default_factory=list)
    speaker_curves: dict[str, list[float | None]] = field(default_factory=dict)
    chapters: list[dict] = field(default_factory=list)
    moments: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"bandwidth": round(self.bandwidth, 2), "duration": round(self.duration, 3), "totals": self.totals,
                "times": [round(t, 2) for t in self.times],
                "curve": [None if v is None else round(v, 4) for v in self.curve],
                "speakers": self.speakers,
                "speaker_curves": {k: [None if v is None else round(v, 4) for v in c]
                                   for k, c in self.speaker_curves.items()},
                "chapters": self.chapters, "moments": self.moments, "units": self.units}


def auto_bandwidth(duration: float) -> float:
    """Gaussian smoothing width (s): a fortieth of the recording, between 4 s and 2 min."""
    return float(min(max(duration / 40.0, 4.0), 120.0))


def smooth(times: np.ndarray, mids: np.ndarray, scores: np.ndarray, weights: np.ndarray, h: float,
           min_density: float = 0.05) -> list[float | None]:
    """Kernel-weighted mean of the scores at every grid time (None where no speech is near)."""
    out: list[float | None] = []
    for t in times:
        k = weights * np.exp(-0.5 * ((t - mids) / h) ** 2)
        s = k.sum()
        out.append(float((k * scores).sum() / s) if s > min_density else None)
    return out


def _share(labels: list[str], weights: list[int]) -> dict:
    total = max(sum(weights), 1)
    return {CLASS_NAME[c]: round(sum(w for lab, w in zip(labels, weights) if lab == c) / total, 4) for c in CLASSES}


def track(units: list[Unit], model: SentimentModel, duration: float = 0.0, bandwidth: float = 0.0,
          points: int = 400, chapters=None) -> SentimentTrack:
    duration = max(duration, units[-1].end if units else 0.0)
    h = bandwidth if bandwidth > 0 else auto_bandwidth(duration)
    P = model.proba([u.text for u in units]) if units else np.zeros((0, 3))
    scores = P[:, 2] - P[:, 0]
    labels = [CLASSES[int(k)] for k in P.argmax(axis=1)]
    rows = [{"start": round(u.start, 3), "end": round(u.end, 3), "text": u.text, "speaker": u.speaker,
             "label": lab, "score": round(float(s), 4), "probs": [round(float(x), 4) for x in p]}
            for u, lab, s, p in zip(units, labels, scores, P)]
    times = np.linspace(0.0, duration, max(points, 2)) if duration > 0 else np.zeros(0)
    mids = np.asarray([(u.start + u.end) / 2 for u in units])
    weights = np.asarray([u.words for u in units], dtype=float)
    curve = smooth(times, mids, scores, weights, h) if units else [None] * len(times)
    wsum = max(weights.sum(), 1.0)
    totals = {"units": len(units), "mean": round(float((weights * scores).sum() / wsum), 4) if units else 0.0,
              "share": _share(labels, [u.words for u in units])}
    res = SentimentTrack(rows, times.tolist(), curve, h, duration, totals)
    names = sorted({u.speaker for u in units if u.speaker})
    for name in names:
        ix = [k for k, u in enumerate(units) if u.speaker == name]
        w = weights[ix]
        res.speakers.append({"speaker": name, "units": len(ix), "words": int(w.sum()),
                             "mean": round(float((w * scores[ix]).sum() / max(w.sum(), 1)), 4),
                             "share": _share([labels[k] for k in ix], [units[k].words for k in ix])})
        if len(names) > 1:
            res.speaker_curves[name] = smooth(times, mids[ix], scores[ix], w, h)
    for ch in chapters or []:
        ix = [k for k, u in enumerate(units) if ch.start <= (u.start + u.end) / 2 < ch.end]
        w = weights[ix]
        res.chapters.append({"index": ch.index, "title": ch.title, "start": round(ch.start, 3),
                             "end": round(ch.end, 3), "units": len(ix),
                             "mean": round(float((w * scores[ix]).sum() / max(w.sum(), 1)), 4) if ix else None})
    res.moments = moments(res, units, scores, h)
    return res


def moments(res: SentimentTrack, units: list[Unit], scores: np.ndarray, h: float, k: int = 3,
            threshold: float = 0.1) -> list[dict]:
    """The k most positive and k most negative peaks of the curve (at least 2 bandwidths apart), each
    with the unit near it that scores most strongly the same way."""
    out = []
    t = np.asarray(res.times)
    c = np.asarray([np.nan if v is None else v for v in res.curve])
    for sign, kind in ((1, "positive"), (-1, "negative")):
        v = sign * c
        peaks = [i for i in range(len(v)) if np.isfinite(v[i]) and v[i] > threshold
                 and (i == 0 or not np.isfinite(v[i - 1]) or v[i] >= v[i - 1])
                 and (i == len(v) - 1 or not np.isfinite(v[i + 1]) or v[i] >= v[i + 1])]
        chosen: list[int] = []
        for i in sorted(peaks, key=lambda i: -v[i]):
            if all(abs(t[i] - t[j]) >= 2 * h for j in chosen):
                chosen.append(i)
            if len(chosen) == k:
                break
        for i in chosen:
            near = [n for n, u in enumerate(units) if abs((u.start + u.end) / 2 - t[i]) <= h]
            if not near:
                continue
            best = max(near, key=lambda n: sign * scores[n])
            u = units[best]
            out.append({"kind": kind, "time": round(float(t[i]), 2), "value": round(float(c[i]), 4),
                        "start": round(u.start, 3), "end": round(u.end, 3), "text": u.text, "speaker": u.speaker,
                        "score": round(float(scores[best]), 4)})
    return out


# ---------------------------------------------------------------- output files
def _clock(t: float) -> str:
    s = int(max(t, 0))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def plot(res: SentimentTrack, path: str | Path, title: str = "Sentiment over time") -> None:
    """Score of every unit (dots by class), the smoothed curve (positive / negative areas) and, with two or
    more speakers, a panel of per-speaker curves on the same scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    surface, ink, ink2, muted, grid, base = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    pos, neg = "#2a78d6", "#e34948"
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    panels = 2 if len(res.speaker_curves) > 1 else 1
    fig, axes = plt.subplots(panels, 1, figsize=(10, 3.4 * panels), dpi=150, sharex=True, facecolor=surface,
                             squeeze=False)
    t = np.asarray(res.times)
    c = np.asarray([np.nan if v is None else v for v in res.curve])
    ax = axes[0][0]
    color = {"pos": pos, "neg": neg, "neu": base}
    for lab in ("neu", "neg", "pos"):
        pts = [((u["start"] + u["end"]) / 2, u["score"]) for u in res.units if u["label"] == lab]
        if pts:
            x, y = zip(*pts)
            ax.scatter(x, y, s=9, color=color[lab], alpha=0.55 if lab != "neu" else 0.45, linewidths=0,
                       label=CLASS_NAME[lab], zorder=2)
    ax.fill_between(t, 0, c, where=np.nan_to_num(c) > 0, color=pos, alpha=0.12, lw=0, interpolate=True)
    ax.fill_between(t, 0, c, where=np.nan_to_num(c) < 0, color=neg, alpha=0.12, lw=0, interpolate=True)
    ax.plot(t, c, color=ink, lw=2, solid_capstyle="round", label=f"smoothed ({res.bandwidth:.0f} s)", zorder=3)
    panel_axes = [ax]
    if panels == 2:
        ax2 = axes[1][0]
        for k, (name, curve) in enumerate(res.speaker_curves.items()):
            if k >= len(series):
                break
            ax2.plot(t, [np.nan if v is None else v for v in curve], color=series[k], lw=2, label=name)
        ax2.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1),
                   ncol=min(len(res.speaker_curves), 6), labelcolor=ink2)
        ax2.set_title("By speaker", fontsize=9, color=ink2, loc="left")
        panel_axes.append(ax2)
    for a in panel_axes:
        a.set_facecolor(surface)
        a.axhline(0, color=base, lw=1, zorder=1)
        a.set_ylim(-1.05, 1.05)
        a.set_yticks([-1, -0.5, 0, 0.5, 1])
        a.grid(axis="y", color=grid, lw=0.8)
        a.set_axisbelow(True)
        a.spines[["top", "right", "left"]].set_visible(False)
        a.spines["bottom"].set_color(base)
        a.tick_params(colors=muted, labelsize=8, length=0)
        a.set_xlim(0, max(res.duration, 1.0))
        for ch in res.chapters[1:]:
            a.axvline(ch["start"], color=grid, lw=1, zorder=0)
    for ch in res.chapters:
        ax.annotate(str(ch["index"]), (ch["start"], 1.02), xytext=(3, 0), textcoords="offset points",
                    fontsize=7, color=muted, va="bottom")
    ax.set_ylabel("positive - negative", color=ink2, fontsize=8)
    sh = res.totals["share"]
    ax.set_title(title, fontsize=10, color=ink, loc="left", pad=22)
    ax.text(0, 1.02, f"{sh['positive']:.0%} positive, {sh['neutral']:.0%} neutral, {sh['negative']:.0%} negative "
                     "(share of words)", transform=ax.transAxes, fontsize=8, color=ink2, va="bottom")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=4, labelcolor=ink2,
              borderaxespad=0.2)
    span = max(res.duration, 1.0)
    step = next((s for s in (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600) if span / s <= 10), 7200)
    panel_axes[-1].xaxis.set_major_locator(MultipleLocator(step))
    panel_axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: _clock(v)))
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def write_sentiment_files(res: SentimentTrack, out_dir: str | Path, stem: str, png: bool = True) -> dict[str, str]:
    from .utils.cache import save_json
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"sentiment_json": out / f"{stem}.sentiment.json", "sentiment_csv": out / f"{stem}.sentiment.csv"}
    save_json(paths["sentiment_json"], res.to_json(), indent=None)
    with open(paths["sentiment_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "speaker", "label", "score", "p_negative", "p_neutral", "p_positive", "text"])
        for u in res.units:
            w.writerow([u["start"], u["end"], u["speaker"] or "", CLASS_NAME[u["label"]], u["score"], *u["probs"],
                        u["text"]])
    if png and res.units:
        paths["sentiment_png"] = out / f"{stem}.sentiment.png"
        plot(res, paths["sentiment_png"], f"Sentiment over time: {stem}")
    return {k: str(v) for k, v in paths.items()}


def summary_line(res: SentimentTrack) -> dict:
    """Compact numbers for logs and the CLI."""
    return {"units": res.totals["units"], "mean": res.totals["mean"], "share": res.totals["share"],
            "bandwidth": round(res.bandwidth, 1),
            "moments": [{"kind": m["kind"], "at": _clock(m["time"]), "text": m["text"][:80]} for m in res.moments]}

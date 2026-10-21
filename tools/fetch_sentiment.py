"""Sentiment data for our own sentiment model (src/subfix/sentiment.py, experiments/train_sentiment.py).

* Stanford Sentiment Treebank (Socher et al. 2013; trainDevTestTrees_PTB.zip, 0.8 MB, nlp.stanford.edu):
  11,855 movie-review sentences parsed into trees with a 5-way sentiment label (0 very negative ..
  4 very positive) on every phrase. Mapped to 3 classes: 0-1 negative, 2 neutral, 3-4 positive.
  Written English; every phrase of the training trees is a training example.
* MELD (Poria et al. 2019; github.com/declare-lab/MELD, GPL-3.0): 13,708 utterances from the TV
  series Friends with their start/end times and a 3-way sentiment label (positive / neutral /
  negative). Spoken dialogue, close to what subtitles contain: the spoken-domain train/dev/test set.

Writes data/sentiment/{sst,meld}/{train,dev,test}.jsonl (+ sst/train_phrases.jsonl), one example per
line: {"text", "label": neg|neu|pos, ...}.

  python -I tools/fetch_sentiment.py
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

RAW = ROOT / "data" / "raw" / "sentiment"
OUT = ROOT / "data" / "sentiment"
SST_URL = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"
SST_SIZE = 789539
MELD_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{split}_sent_emo.csv"
MELD_SIZES = {"train": 1119290, "dev": 121640, "test": 294526}
MELD_LICENSE = "https://raw.githubusercontent.com/declare-lab/MELD/master/LICENSE"
FINE_TO_3 = {0: "neg", 1: "neg", 2: "neu", 3: "pos", 4: "pos"}
MELD_TO_3 = {"negative": "neg", "neutral": "neu", "positive": "pos"}
_PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}", "``": '"', "''": '"'}


# ------------------------------------------------------------------ SST
def parse_tree(line: str) -> tuple[int, list[str], list[tuple[list[str], int]]]:
    """'(3 (2 It) (4 ...))' -> (root label, words, [(phrase words, label) for every node])."""
    toks = re.findall(r"\(|\)|[^ ()]+", line)          # split on ASCII spaces only: "8 1\/2" is one word
    pos = 0

    def node():
        nonlocal pos
        if toks[pos] != "(":
            raise ValueError(f"bad tree near token {pos}: {line[:80]}")
        label = int(toks[pos + 1])
        pos += 2
        phrases: list[tuple[list[str], int]] = []
        if toks[pos] == "(":
            words = []
            while toks[pos] == "(":
                _, w, p = node()
                words += w
                phrases += p
        else:
            words = [_PTB.get(toks[pos], toks[pos].replace("\\/", "/").replace("\\*", "*"))]
            pos += 1
        if toks[pos] != ")":
            raise ValueError(f"bad tree near token {pos}: {line[:80]}")
        pos += 1
        phrases.append((words, label))
        return label, words, phrases

    return node()


def detokenize(words: list[str]) -> str:
    """PTB tokens -> readable text ("it 's not" -> "it's not"); our sentiment tokenizer splits both alike."""
    text = " ".join(words)
    text = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)
    text = re.sub(r" ([.,!?;:%)\]}])", r"\1", text)
    text = re.sub(r"([(\[{$]) ", r"\1", text)
    return text


def convert_sst(zpath: Path) -> dict[str, int]:
    out = OUT / "sst"
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    with zipfile.ZipFile(zpath) as z:
        for split in ("train", "dev", "test"):
            lines = z.read(f"trees/{split}.txt").decode("utf-8").splitlines()
            rows, phrases = [], {}
            for line in lines:
                if not line.strip():
                    continue
                label, words, ph = parse_tree(line)
                rows.append({"text": detokenize(words), "fine": label, "label": FINE_TO_3[label]})
                if split == "train":
                    for w, lab in ph:
                        phrases.setdefault(detokenize(w), lab)
            write_jsonl(out / f"{split}.jsonl", rows)
            counts[f"sst/{split}"] = len(rows)
            if split == "train":
                write_jsonl(out / "train_phrases.jsonl",
                            [{"text": t, "fine": lab, "label": FINE_TO_3[lab]} for t, lab in phrases.items()])
                counts["sst/train_phrases"] = len(phrases)
    return counts


# ------------------------------------------------------------------ MELD
# the MELD CSVs hold Windows-1252 punctuation decoded as Latin-1 control characters (e.g. "\x92")
_CP1252 = {"\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x85": "...", "\x96": "-", "\x97": "-",
           "\xa0": " "}


def clean_meld(text: str) -> str:
    for bad, good in _CP1252.items():
        text = text.replace(bad, good)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("…", "...").replace("—", "-").replace("–", "-")
    return " ".join(text.split())


def clock(t: str) -> float:
    """'00:16:16,518' -> seconds."""
    h, m, s = t.strip().replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def convert_meld() -> dict[str, int]:
    out = OUT / "meld"
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "dev", "test"):
        raw = (RAW / "meld" / f"{split}_sent_emo.csv").read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")
        rows = []
        for r in csv.DictReader(io.StringIO(text)):
            utt = clean_meld(r["Utterance"])
            if not utt:
                continue
            rows.append({"text": utt, "label": MELD_TO_3[r["Sentiment"].strip().lower()],
                         "emotion": r["Emotion"].strip().lower(), "speaker": clean_meld(r["Speaker"]),
                         "dialogue": int(r["Dialogue_ID"]), "utterance": int(r["Utterance_ID"]),
                         "season": int(r["Season"]), "episode": int(r["Episode"]),
                         "start": round(clock(r["StartTime"]), 3), "end": round(clock(r["EndTime"]), 3)})
        write_jsonl(out / f"{split}.jsonl", rows)
        counts[f"meld/{split}"] = len(rows)
    return counts


# ------------------------------------------------------------------ main
def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def fetch() -> Path:
    from fetch_data import download
    zpath = RAW / "trainDevTestTrees_PTB.zip"
    download([SST_URL], zpath, size=SST_SIZE)
    with zipfile.ZipFile(zpath) as z:
        if z.testzip() is not None:
            zpath.unlink()
            raise SystemExit(f"{zpath.name} is corrupt: run again")
    for split, size in MELD_SIZES.items():
        download([MELD_URL.format(split=split)], RAW / "meld" / f"{split}_sent_emo.csv", size=size)
    lic = RAW / "meld" / "LICENSE"
    if not lic.exists():
        download([MELD_LICENSE], lic)
    return zpath


def main() -> int:
    zpath = fetch()
    counts = convert_sst(zpath)
    counts.update(convert_meld())
    for k, v in counts.items():
        print(f"{k}: {v:,}")
    for name in ("sst/test", "meld/test"):
        rows = [json.loads(x) for x in (OUT / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        dist = {lab: sum(r["label"] == lab for r in rows) / len(rows) for lab in ("neg", "neu", "pos")}
        print(f"{name} labels: " + ", ".join(f"{k} {v:.0%}" for k, v in dist.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

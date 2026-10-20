"""Train our modified Kneser-Ney n-gram LM (replaces distilgpt2 in evidence fusion).

python experiments/train_lm.py [--order 4] [--min-count 2]
-> models/lm/kn4.npz  (+ models/lm/kn4.json with perplexities)

Training text: LibriSpeech-PC train-clean-100/360 + train-other-500 sentences (~8.5M words).
Perplexity is reported on the dev-clean / dev-other manifests (never used for training).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from corpus import sentences  # noqa: E402
from subfix.lm import lm_words  # noqa: E402
from subfix.ngram import KneserNeyLM  # noqa: E402
from subfix.paths import MODELS_DIR  # noqa: E402


def to_words(sent: str) -> list[str]:
    return [w for tok in sent.split() for w in lm_words(tok)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--out", default=str(MODELS_DIR / "lm" / "kn4.npz"))
    args = ap.parse_args()
    t0 = time.time()
    train = sentences()
    lm = KneserNeyLM.train(lambda: (to_words(s) for s in train), order=args.order, min_count=args.min_count)
    lm.save(args.out)
    print(f"[lm] trained + saved in {time.time() - t0:.0f}s -> {args.out}", flush=True)
    lm = KneserNeyLM.load(args.out)
    report = {"order": args.order, "min_count": args.min_count, "vocab": len(lm.vocab),
              "train_sentences": len(train)}
    for split in ("dev-clean", "dev-other"):
        dev = [to_words(s) for s in sentences((split,))]
        report[f"ppl_{split}"] = round(lm.perplexity(dev), 1)
        oov = sum(lm.word_id(w) == lm.unk for s in dev for w in s) / max(1, sum(map(len, dev)))
        report[f"oov_{split}"] = round(oov, 4)
    for k in range(1, args.order + 1):
        report[f"ngrams_{k}"] = int(len(lm.t[f"k{k}"]))
    print("[lm]", report, flush=True)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

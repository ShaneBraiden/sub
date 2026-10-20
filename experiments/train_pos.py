"""Train our averaged-perceptron POS tagger (replaces spaCy en_core_web_sm in the segmenter).

python experiments/train_pos.py [--iters 5]
-> models/pos/perceptron.json.gz (+ .metrics.json with held-out accuracy)

Data: the Brown corpus (NLTK, ~1.16M tokens), Brown tags mapped to Universal Dependencies POS.
The last 5% of the (fixed-order) sentences are held out for the accuracy report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subfix.paths import MODELS_DIR  # noqa: E402
from subfix.postag import PerceptronTagger, brown_to_ud  # noqa: E402


def load_brown() -> list[list[tuple[str, str]]]:
    import nltk
    try:
        from nltk.corpus import brown
        brown.tagged_sents()[0]
    except LookupError:
        nltk.download("brown", quiet=True)
        from nltk.corpus import brown
    return [brown_to_ud(list(s)) for s in brown.tagged_sents()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=str(MODELS_DIR / "pos" / "perceptron.json.gz"))
    args = ap.parse_args()
    t0 = time.time()
    sents = load_brown()
    n_dev = len(sents) // 20
    train, dev = sents[:-n_dev], sents[-n_dev:]
    print(f"[pos] {len(train)} train / {len(dev)} held-out sentences; tags:",
          dict(Counter(t for s in train for _, t in s).most_common()), flush=True)
    tagger = PerceptronTagger()
    tagger.train(train, n_iter=args.iters, log=lambda m: print(m, flush=True))
    tagger.save(args.out)
    tagger = PerceptronTagger.load(args.out)
    correct = total = 0
    per_tag: dict[str, list[int]] = {}
    for s in dev:
        pred = tagger.tag([w for w, _ in s])
        for (_, gold), p in zip(s, pred):
            correct += gold == p
            total += 1
            r = per_tag.setdefault(gold, [0, 0])
            r[0] += gold == p
            r[1] += 1
    report = {"heldout_accuracy": round(correct / total, 4), "heldout_tokens": total,
              "per_tag_recall": {t: round(c / n, 3) for t, (c, n) in sorted(per_tag.items())},
              "features": len(tagger.weights), "tagdict": len(tagger.tagdict),
              "train_seconds": round(time.time() - t0)}
    print("[pos]", report, flush=True)
    Path(args.out.replace(".json.gz", ".metrics.json")).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

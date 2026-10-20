"""Download the Whisper models (the only pretrained models in the system) with retries, and
report which of our self-trained models are present.

python scripts/prefetch_models.py [--only large-v3-turbo ...]

Our models are trained locally (see README "Self-trained models"):
  models/lm/kn4.npz                 experiments/train_lm.py
  models/punct/model.pt             experiments/train_punct.py
  models/corrector/model.pt         experiments/train_corrector.py
  models/pos/perceptron.json.gz     experiments/train_pos.py
  models/aligner/model.pt           experiments/train_aligner.py  (needs LibriSpeech train-clean-100)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path(__file__).resolve().parents[1]

WHISPER = ["large-v3-turbo", "small.en", "tiny.en"]
OWN_MODELS = {
    "models/lm/kn4.npz": "experiments/train_lm.py",
    "models/punct/model.pt": "experiments/train_punct.py",
    "models/corrector/model.pt": "experiments/train_corrector.py",
    "models/pos/perceptron.json.gz": "experiments/train_pos.py",
    "models/aligner/model.pt": "experiments/train_aligner.py",
}


def retry(fn, what: str, tries: int = 30):
    for k in range(tries):
        try:
            out = fn()
            print(f"[ok] {what}", flush=True)
            return out
        except Exception as exc:  # network resets
            print(f"[retry {k + 1}] {what}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            time.sleep(min(60, 5 + 3 * k))
    print(f"[FAILED] {what}", flush=True)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    from faster_whisper.utils import download_model
    for name in WHISPER:
        if args.only and name not in args.only:
            continue
        retry(lambda: download_model(name), f"whisper {name}")
    missing = 0
    for path, script in OWN_MODELS.items():
        ok = (ROOT / path).exists()
        missing += not ok
        print(f"[{'ok' if ok else 'MISSING'}] {path}" + ("" if ok else f"  -> python {script}"))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

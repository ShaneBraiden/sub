"""LibriSpeech-PC *train* text shared by the training scripts (LM, punctuation, corrector).

The manifests are punctuated and cased (Meister et al. 2023). Only the train splits are used;
dev/test books, speakers and our benchmark clips come from dev-clean / test-clean / test-other.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subfix import text as T  # noqa: E402
from subfix.paths import DATA_DIR  # noqa: E402

TRAIN_SPLITS = ("train-clean-100", "train-clean-360", "train-other-500")
PC_DIR = DATA_DIR / "raw" / "librispeech" / "pc"


def utterances(splits=TRAIN_SPLITS) -> list[tuple[str, str]]:
    """(utterance id, text) in manifest order (consecutive utterances of a chapter stay together)."""
    out = []
    for split in splits:
        path = PC_DIR / f"{split}.json"
        for line in path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            out.append((Path(d["audio_filepath"]).stem, " ".join(d["text"].split())))
    return out


def chapters(splits=TRAIN_SPLITS):
    """Token streams of whole chapters (utterances joined in order), for long-form tagging.
    A generator, so only one chapter's tokens exist as Python strings at a time."""
    groups: dict[str, list[tuple[int, str]]] = {}
    for uid, text in utterances(splits):
        spk, chap, num = uid.split("-")
        groups.setdefault(f"{spk}-{chap}", []).append((int(num), text))
    for _, rows in sorted(groups.items()):
        yield " ".join(t for _, t in sorted(rows)).split()


def sentences(splits=TRAIN_SPLITS) -> list[str]:
    """Sentences (split at . ? ! inside utterances); utterance ends also end a sentence."""
    out = []
    for _, text in utterances(splits):
        cur = []
        for tok in text.split():
            cur.append(tok)
            if T.ends_sentence(tok):
                out.append(" ".join(cur))
                cur = []
        if cur:
            out.append(" ".join(cur))
    return out

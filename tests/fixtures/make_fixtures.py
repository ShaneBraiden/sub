"""Build the 3 golden integration fixtures from LibriSpeech dev-clean clips (CC BY 4.0).

For each clip: the first utterances up to ~25 s -> clipN.flac, the MFA-timed reference words ->
clipN.ref.srt (strict DP segmentation) + clipN.words.json, and a seeded corruption with a known
offset/drift -> clipN.corrupted.srt + clipN.corrupted.log.json.

python tests/fixtures/make_fixtures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from subfix.config import Config  # noqa: E402
from subfix.segment import segment  # noqa: E402
from subfix.subio import Word, write_srt  # noqa: E402
from subfix.utils.cache import save_json  # noqa: E402
from tools.corrupt import TimingParams, corrupt_clip  # noqa: E402

CLIPS = ["libri-dev-clean-84-121550", "libri-dev-clean-3752-4944", "libri-dev-clean-6241-66616"]
MAX_S = 25.0


def main() -> None:
    cfg = Config.load(use_tuned=False)
    for n, cid in enumerate(CLIPS, 1):
        clip = json.loads((ROOT / "data" / "clips" / "libri-dev-clean" / f"{cid.split('libri-dev-clean-')[1]}.json")
                          .read_text(encoding="utf-8"))
        utts = [u for u in clip["utterances"] if u["end"] <= MAX_S] or clip["utterances"][:1]
        end = utts[-1]["end"] + 0.3
        audio, sr = sf.read(str(ROOT / clip["audio"]), dtype="float32")
        sf.write(str(HERE / f"clip{n}.flac"), audio[: int(end * sr)], sr)
        words = json.loads((ROOT / "data" / "refs" / f"{cid}.words.json").read_text(encoding="utf-8"))
        words = [w for w in words if w["end"] <= end + 0.05]
        ws = [Word(w["text"], w["start"], w["end"]) for w in words]
        cues = segment(ws, cfg, None)
        write_srt(cues, HERE / f"clip{n}.ref.srt")
        save_json(HERE / f"clip{n}.words.json", words)
        tp = TimingParams(max_offset=0.0, drift_prob=0.0, split_prob=0.0)   # timing set explicitly below
        res = corrupt_clip(cues, f"fixture{n}", True, None, seed=n, rate_scale=2.0, tp=tp)
        offset = [1.5, -2.0, 3.0][n - 1]
        for c in res.cues:                                        # known offset for the sync assertion
            c.start, c.end = max(0.0, c.start + offset), max(0.0, c.end + offset)
        res.log["timing"]["offset"] = offset
        write_srt(res.cues, HERE / f"clip{n}.corrupted.srt")
        save_json(HERE / f"clip{n}.corrupted.log.json", res.log)
        print(f"clip{n}: {end:.1f}s, {len(words)} words, {len(cues)} cues, errors {res.log['counts']}")


if __name__ == "__main__":
    main()

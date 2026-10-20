"""Build benchmark v1 end-to-end — imp.md 5 / 11 (Week 3).

stages: fetch -> clips -> asr -> refs -> corrupt -> manifest   (each stage is cached/idempotent)

  python tools/build_benchmark.py --stages all
  python tools/build_benchmark.py --stages corrupt manifest --sweep 0.05 0.10 0.20

Splits: dev = libri-dev-clean + ted-validation (tuning only); test = libri-test-clean,
libri-test-other, ted-test. Manifests: data/benchmark/<split>[_<variant>].jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from subfix.paths import DATA_DIR  # noqa: E402

DEV_SETS = {"libri-dev-clean", "ted-validation"}
SWEEP_SETS = {"libri-test-clean", "ted-test"}


def split_of(dataset: str) -> str:
    return "dev" if dataset in DEV_SETS else "test"


def stage_fetch(args) -> None:
    import fetch_data
    fetch_data.fetch_librispeech(["dev-clean", "test-clean", "test-other"])
    fetch_data.fetch_tedlium(["validation", "test"])


def stage_clips(args) -> None:
    import fetch_data
    for split in ("dev-clean", "test-clean", "test-other"):
        fetch_data.build_libri_clips(split, args.libri_n, args.libri_minutes)
    fetch_data.build_ted_clips("validation", args.ted_dev_n)
    fetch_data.build_ted_clips("test", args.ted_test_n)


def stage_asr(args) -> None:
    import fetch_data
    from subfix.asr import transcribe
    clips = fetch_data.all_clips()
    for model in args.asr_models:
        for c in clips:
            wav = str(ROOT / c["audio"])
            res = transcribe(wav, model=model)
            print(f"[asr] {model:16s} {c['id']}: {len(res.words)} words, {res.info.get('elapsed', 0):.1f}s", flush=True)


def stage_refs(args) -> None:
    import subprocess
    cmd = [sys.executable, str(ROOT / "tools" / "make_refs.py"), "--aligner", args.ref_aligner, "--jobs", str(args.jobs)]
    subprocess.run(cmd, check=True)


def stage_corrupt(args) -> None:
    import corrupt
    import fetch_data
    for c in fetch_data.all_clips():
        if not (DATA_DIR / "refs" / f"{c['id']}.srt").exists():
            print(f"[corrupt] {c['id']}: no reference, skipped")
            continue
        log = corrupt.corrupt_file(c, "default", args.seed, 1.0)
        print(f"[corrupt] {c['id']}: {log['counts']}", flush=True)
        if c["dataset"] in SWEEP_SETS:
            for rate in args.sweep:
                corrupt.corrupt_file(c, f"rate{rate:.2f}", args.seed, rate / corrupt.TOTAL_WORD_RATE)


def stage_manifest(args) -> None:
    import fetch_data
    out = DATA_DIR / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    variants = ["default"] + [f"rate{r:.2f}" for r in args.sweep]
    for variant in variants:
        rows: dict[str, list[dict]] = {"dev": [], "test": []}
        for c in fetch_data.all_clips():
            cor = DATA_DIR / "corrupted" / variant / f"{c['id']}.srt"
            ref = DATA_DIR / "refs" / f"{c['id']}.srt"
            if not (cor.exists() and ref.exists()):
                continue
            rel = lambda p: str(p.relative_to(ROOT)).replace("\\", "/")  # noqa: E731
            rows[split_of(c["dataset"])].append({
                "id": c["id"], "dataset": c["dataset"], "split": split_of(c["dataset"]), "variant": variant,
                "audio": c["audio"], "duration": c["duration"], "has_punct": c.get("has_punct", False),
                "ref_srt": rel(ref), "ref_words": rel(DATA_DIR / "refs" / f"{c['id']}.words.json"),
                "ref_meta": rel(DATA_DIR / "refs" / f"{c['id']}.meta.json"),
                "corrupted_srt": rel(cor), "corruption_log": rel(cor.with_suffix(".log.json")),
            })
        for split, items in rows.items():
            if not items:
                continue
            name = split if variant == "default" else f"{split}_{variant}"
            path = out / f"{name}.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in items) + "\n", encoding="utf-8")
            print(f"[manifest] {path.name}: {len(items)} items")


STAGES = {"fetch": stage_fetch, "clips": stage_clips, "asr": stage_asr, "refs": stage_refs,
          "corrupt": stage_corrupt, "manifest": stage_manifest}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["all"])
    ap.add_argument("--libri-n", type=int, default=8)
    ap.add_argument("--libri-minutes", type=float, default=5.0)
    ap.add_argument("--ted-dev-n", type=int, default=4)
    ap.add_argument("--ted-test-n", type=int, default=11)
    ap.add_argument("--asr-models", nargs="+", default=["large-v3-turbo", "tiny.en", "small.en"])
    ap.add_argument("--ref-aligner", default="mfa")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--sweep", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    stages = list(STAGES) if "all" in args.stages else args.stages
    for s in stages:
        print(f"===== stage: {s} =====", flush=True)
        STAGES[s](args)


if __name__ == "__main__":
    main()

"""Training data for the neural fusion network (subfix.fusenet, experiments/train_fusion.py).

python experiments/fusion_data.py [--stages clips asr refs lm features] [--splits dev-clean ted train-clean-100]

Every stage is cached per clip under cache/train/fusion/, so an interrupted run continues.
  clips     up to 5 min per chapter: LibriSpeech train-clean-100 (one chapter per speaker), dev-clean
            chapters of the speakers outside the benchmark, and the TED validation talks outside
            the benchmark. No benchmark speaker or talk is ever used for training.
  asr       Whisper large-v3-turbo (the pipeline's ASR) and tiny.en (source of the "asr" errors)
  refs      reference words timed by our CTC aligner, cut into cues by the DP segmenter
  lm        an n-gram LM without train-clean-100, which scores the candidates of that split. The
            pipeline's LM trained on those sentences and ranks their candidates far better than it
            ranks unseen text (substitutions: AUC 0.93 vs 0.83), so a network trained on its scores
            would trust the LM too much. Our CTC model shows almost no such gap (0.70 vs 0.66).
  features  for each clip, several synthetic corruptions (tools/corrupt.py, seeds 1.., at several
            error rates) go through the real sync -> alignment -> candidate code. Each candidate
            gets the network input and a label: does applying it lower the edit distance to
            the reference? Benchmark dev clips get the official corruption (seed 0) as well as extra
            seeds and are kept for model selection only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "tools")]

from subfix import text as T  # noqa: E402
from subfix.paths import DATA_DIR, cache_dir  # noqa: E402

WORK = cache_dir("train") / "fusion"
CLIPS = WORK / "clips"
REFS = WORK / "refs"
FEATURES = WORK / "features"
HELDOUT = "train-clean-100"
HELDOUT_LM = WORK / "lm_heldout" / "kn4.npz"
RAW = DATA_DIR / "raw"
# (seed, error-rate scale): 5.5%, 11% (the benchmark's rate) and 22% of the words corrupted
TRAIN_SEEDS = [(1, 1.0), (2, 1.0), (3, 0.5), (4, 2.0)]
DEV_SEEDS = [(1, 1.0), (2, 1.0)]
# selection score of both the rule and the network = F0.5 - WER_WEIGHT * WER%: 0.1 WER points are worth 0.001 of F0.5
WER_WEIGHT = 0.01


# ------------------------------------------------------------------ labels
@lru_cache(maxsize=300_000)
def norm_words(token: str) -> tuple[str, ...]:
    """A token as the WER metric sees it (Whisper's text normalizer), so labels match the metric."""
    from metrics.text_metrics import normalize_text
    return tuple(normalize_text(token).split())


def edit_gains(cands: list, sub, asr, ref: list[str], window: int = 12) -> list[int]:
    """For each candidate: edit distance to the reference without it minus with it, over a window
    of subtitle words around it (> 0 = the edit helps, 0 = no effect, < 0 = it harms). Measuring
    the effect avoids the ambiguity of position-based labels: in "the the" either copy may go."""
    from rapidfuzz.distance import Levenshtein
    S = [norm_words(w.text) for w in sub]
    A = [norm_words(w.text) for w in asr]
    flat, first = [], []
    for toks in S:
        first.append(len(flat))
        flat.extend(toks)
    first.append(len(flat))
    to_ref = [len(ref)] * (len(flat) + 1)
    for _, i1, i2, j1, j2 in Levenshtein.opcodes(flat, ref):
        for k in range(i2 - i1):
            to_ref[i1 + k] = j1 + min(k, max(j2 - j1 - 1, 0))
    gains = []
    for c in cands:
        anchor = c.i if c.i is not None else c.pos
        a, b = max(0, anchor - window), min(len(S), anchor + window + 1)
        ref_win = ref[max(0, to_ref[first[a]] - 3):min(len(ref), to_ref[first[b]] + 3)]
        before = [x for toks in S[a:b] for x in toks]
        edited = list(S[a:b])
        off = anchor - a
        if c.kind == "sub":
            edited[off] = A[c.j]
        elif c.kind == "ins":
            edited.insert(off, A[c.j])
        elif c.kind == "del":
            edited[off] = ()
        elif c.kind == "split":
            edited[off] = A[c.j] + A[c.j2]
        elif c.kind == "merge":
            edited[off] = A[c.j]
            if c.i2 - a < len(edited):
                edited[c.i2 - a] = ()
        after = [x for toks in edited for x in toks]
        gains.append(Levenshtein.distance(before, ref_win) - Levenshtein.distance(after, ref_win))
    return gains


# ------------------------------------------------------------------ clips
def benchmark_speakers() -> tuple[set[str], set[str]]:
    libri = {p.stem.split("-")[0] for p in (DATA_DIR / "clips" / "libri-dev-clean").glob("*.json")}
    ted = {p.stem for p in (DATA_DIR / "clips" / "ted-validation").glob("*.json")}
    return libri, ted


def build_libri(split: str, exclude: set[str], per_speaker: int | None, max_minutes: float = 5.0,
                min_minutes: float = 1.0, gap_s: float = 0.4, seed: int = 17) -> list[dict]:
    """Chapter clips built like the benchmark's (tools/fetch_data.py build_libri_clips)."""
    import random

    import soundfile as sf
    from fetch_data import _pc_manifest
    sdir = RAW / "librispeech" / "LibriSpeech" / split
    pc = _pc_manifest(split)
    chapters: dict[tuple[str, str], list[str]] = {}
    for uid in sorted(pc):
        spk, chap, _ = uid.split("-")
        if spk not in exclude:
            chapters.setdefault((spk, chap), []).append(uid)
    eligible = [k for k, v in chapters.items() if sum(pc[u]["duration"] for u in v) >= min_minutes * 60]
    if per_speaker:
        rng = random.Random(seed)
        by_spk: dict[str, list] = {}
        for k in eligible:
            by_spk.setdefault(k[0], []).append(k)
        eligible = [k for s in sorted(by_spk) for k in rng.sample(sorted(by_spk[s]), min(per_speaker, len(by_spk[s])))]
    out_dir = CLIPS / split
    out_dir.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(int(gap_s * 16000), np.float32)
    clips = []
    for spk, chap in sorted(eligible):
        meta_path = out_dir / f"{spk}-{chap}.json"
        if meta_path.exists():
            clips.append(json.loads(meta_path.read_text(encoding="utf-8")))
            continue
        pieces, utts, t = [], [], 0.0
        for uid in chapters[(spk, chap)]:
            dur = pc[uid]["duration"]
            if t + dur > max_minutes * 60 and utts:
                break
            audio, sr = sf.read(str(sdir / spk / chap / f"{uid}.flac"), dtype="float32")
            assert sr == 16000
            if utts:
                pieces.append(silence)
                t += gap_s
            utts.append({"id": uid, "start": round(t, 3), "end": round(t + len(audio) / sr, 3), "text": pc[uid]["text"]})
            pieces.append(audio)
            t += len(audio) / sr
        wav = out_dir / f"{spk}-{chap}.wav"
        sf.write(str(wav), np.concatenate(pieces), 16000, subtype="PCM_16")
        meta = {"id": f"fusion-{split}-{spk}-{chap}", "dataset": split, "audio": str(wav), "duration": round(t, 3),
                "has_punct": True, "text": " ".join(u["text"] for u in utts), "utterances": utts}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        clips.append(meta)
    return clips


def build_ted(exclude: set[str]) -> list[dict]:
    import soundfile as sf
    clips = []
    for p in sorted((RAW / "tedlium" / "validation").glob("*.json")):
        if p.stem in exclude:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        wav = p.with_suffix(".wav")
        info = sf.info(str(wav))
        clips.append({"id": f"fusion-ted-{p.stem}", "dataset": "ted", "audio": str(wav),
                      "duration": round(info.frames / info.samplerate, 3), "has_punct": False,
                      "text": " ".join(d["text"].split()), "utterances": None})
    return clips


def all_clips(splits: list[str], per_speaker: int) -> list[dict]:
    libri_ex, ted_ex = benchmark_speakers()
    clips = []
    for split in splits:
        if split == "ted":
            clips += build_ted(ted_ex)
        elif split == "dev-clean":
            clips += build_libri(split, libri_ex, None)
        else:
            clips += build_libri(split, set(), per_speaker)
    return clips


def benchmark_dev() -> list[dict]:
    """Benchmark dev clips with their official references (tuning only, never training)."""
    rows = [json.loads(x) for x in (DATA_DIR / "benchmark" / "dev.jsonl").read_text(encoding="utf-8").splitlines() if x]
    for r in rows:
        r["audio"] = str(ROOT / r["audio"])
    return rows


# ------------------------------------------------------------------ stages
def stage_asr(clips: list[dict], cfg) -> None:
    from subfix.asr import transcribe
    t0 = time.time()
    done = 0.0
    total = sum(c["duration"] for c in clips)
    for k, c in enumerate(clips):
        transcribe(c["audio"], model=cfg.asr.model)
        transcribe(c["audio"], model="tiny.en")
        done += c["duration"]
        if (k + 1) % 10 == 0 or k + 1 == len(clips):
            print(f"[asr] {k + 1}/{len(clips)} clips, {done / 3600:.1f}/{total / 3600:.1f} h of audio, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)


def stage_refs(clips: list[dict], cfg, models) -> None:
    from make_refs import libri_utterances, strict_config, ted_utterances
    from subfix.asr import transcribe
    from subfix.segment import pos_tags, segment
    from subfix.subio import Word, tokenize, write_srt
    from subfix.timing import provisional_times, word_times
    REFS.mkdir(parents=True, exist_ok=True)
    strict = strict_config()
    aligner = models.aligner("ctc", cfg)
    tagger = models.pos_tagger(cfg)
    t0 = time.time()
    for k, c in enumerate(clips):
        path = REFS / f"{c['id']}.srt"
        if path.exists():
            continue
        toks = tokenize(c["text"])
        asr = transcribe(c["audio"], model=cfg.asr.model)
        utts = libri_utterances(c, toks) if c.get("utterances") else ted_utterances(c, toks, asr.words)
        words = [Word(t) for t in toks]
        for u in utts:
            a, b = u["tok"]
            for w, (s, e) in zip(words[a:b], provisional_times(words[a:b], u["end"] - u["start"])):
                w.start, w.end = u["start"] + float(s), u["start"] + float(e)
        timed, _ = word_times(words, c["audio"], cfg, aligner=aligner)
        write_srt(segment(timed, strict, pos_tags(timed, tagger)), path)
        if (k + 1) % 25 == 0:
            print(f"[refs] {k + 1}/{len(clips)} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def candidates(cues, asr, cfg, lm, spell) -> tuple:
    """The real sync -> alignment -> candidates path for one corrupted subtitle."""
    from subfix.align import align_words
    from subfix.fuse import Fuser
    from subfix.subio import Cue, cues_to_words
    from subfix.sync import sync_cues
    synced, res = sync_cues(cues, asr.words, cfg)
    sub = cues_to_words([Cue(c.index, c.start, c.end, c.text) for c in synced])
    ops = align_words(sub, asr.words, cfg, use_time=res.method != "none")
    fz = Fuser(cfg, lm=lm, spell=spell)
    units = fz.units(ops, sub, asr.words)
    return sub, units, [fz.decide(c) for c in fz.featurize(units, sub, asr.words)]


def sequence_row(cues, asr, ref: list[str], cfg, lm, spell, evidence) -> dict:
    """One corrupted subtitle through candidates(), as arrays."""
    from rapidfuzz.distance import Levenshtein
    from subfix.fusenet import unit_table
    from subfix.subio import is_caseless
    sub, units, cands = candidates(cues, asr, cfg, lm, spell)
    table = unit_table(units, cands, sub, asr.words, spell, is_caseless(sub), evidence)
    gains = edit_gains(cands, sub, asr.words, ref)
    n = len(units)
    gain = np.zeros(n, np.int8)
    rule = np.full(n, -1, np.int8)
    scored = np.zeros(n, bool)
    for u in np.flatnonzero(table.cand >= 0):
        c = cands[table.cand[u]]
        gain[u] = int(np.clip(gains[table.cand[u]], -5, 5))
        rule[u] = c.accept
        scored[u] = c.kind in ("sub", "ins", "del") and c.rule not in ("event", "filler")
    flat = [x for w in sub for x in norm_words(w.text)]
    return {"kinds": table.kinds, "s": table.s_forms, "a": table.a_forms, "num": table.num, "gain": gain, "rule": rule,
            "scored": scored, "err": Levenshtein.distance(flat, ref), "n_ref": len(ref)}


def save_rows(path: Path, rows: list[dict], seeds: list[int], dataset: str) -> None:
    forms: dict[str, int] = {"": 0}
    sid = np.array([forms.setdefault(x, len(forms)) for r in rows for x in r["s"]], np.int32)
    aid = np.array([forms.setdefault(x, len(forms)) for r in rows for x in r["a"]], np.int32)
    np.savez(path, forms=np.array(list(forms), dtype=str), s=sid, a=aid,
             lengths=np.array([len(r["kinds"]) for r in rows], np.int32), seeds=np.array(seeds, np.int32),
             kinds=np.concatenate([r["kinds"] for r in rows]), num=np.concatenate([r["num"] for r in rows]),
             gain=np.concatenate([r["gain"] for r in rows]), rule=np.concatenate([r["rule"] for r in rows]),
             scored=np.concatenate([r["scored"] for r in rows]), err=np.array([r["err"] for r in rows], np.int32),
             n_ref=np.array([r["n_ref"] for r in rows], np.int32), dataset=np.array(dataset))


def stage_features(clips: list[dict], dev: list[dict], cfg, models, holdout: str | None = None) -> None:
    from corrupt import corrupt_clip
    from subfix.asr import transcribe
    from subfix.fusenet import CtcEvidence
    from subfix.lm import LMScorer
    from subfix.preprocess import load_audio
    from subfix.subio import read_srt, tokenize
    train_group = f"train_no_{holdout}" if holdout else "train"
    (FEATURES / train_group).mkdir(parents=True, exist_ok=True)
    (FEATURES / "dev").mkdir(parents=True, exist_ok=True)
    lm, spell = models.lm(cfg), models.spell(cfg)
    heldout_lm = LMScorer(str(HELDOUT_LM)) if any(c["dataset"] == HELDOUT for c in clips) else None
    aligner = models.aligner("ctc", cfg)
    rates = {holdout: 0.0} if holdout else None
    jobs = [(train_group, c, REFS / f"{c['id']}.srt", TRAIN_SEEDS) for c in clips]
    jobs += [("dev", c, ROOT / c["ref_srt"], DEV_SEEDS) for c in dev]
    t0 = time.time()
    for k, (group, c, ref_path, seeds) in enumerate(jobs):
        path = FEATURES / group / f"{c['id']}.npz"
        if path.exists() or not ref_path.exists():
            continue
        asr = transcribe(c["audio"], model=cfg.asr.model)
        tiny = [w.text for w in transcribe(c["audio"], model="tiny.en").words]
        ref_cues = read_srt(ref_path)
        ref = [x for cue in ref_cues for t in tokenize(cue.text.replace("\n", " ")) for x in norm_words(t)]
        evidence = CtcEvidence(aligner, load_audio(c["audio"]))
        clip_lm = heldout_lm if c["dataset"] == HELDOUT else lm
        rows, used = [], []
        if group == "dev":
            rows.append(sequence_row(read_srt(ROOT / c["corrupted_srt"]), asr, ref, cfg, lm, spell, evidence))
            used.append(0)
        for seed, scale in seeds:
            cor = corrupt_clip(ref_cues, c["id"], c.get("has_punct", False), tiny, seed=seed, rate_scale=scale,
                               rates=rates if group != "dev" else None)
            rows.append(sequence_row(cor.cues, asr, ref, cfg, clip_lm, spell, evidence))
            used.append(seed)
        save_rows(path, rows, used, c["dataset"])
        if (k + 1) % 10 == 0 or k + 1 == len(jobs):
            print(f"[features] {k + 1}/{len(jobs)} clips ({(time.time() - t0) / 60:.1f} min)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["clips", "asr", "refs", "lm", "features"],
                    choices=["clips", "asr", "refs", "lm", "features"])
    ap.add_argument("--splits", nargs="+", default=["dev-clean", "ted", "train-clean-100"],
                    help="dev-clean and ted run first: their CTC evidence comes from audio the aligner never saw")
    ap.add_argument("--per-speaker", type=int, default=1, help="train-clean-100 chapters per speaker")
    ap.add_argument("--holdout", default=None, choices=["entity", "asr", "homophone", "spelling", "drop", "dup"],
                    help="leave this error type out of the training corruptions (features/train_no_<type>), "
                         "to test the network on an error type it never saw")
    args = ap.parse_args()
    from subfix.config import Config
    from subfix.models import Models
    cfg = Config.load(overrides={"fusion.method": "rule"})
    models = Models(cfg)
    clips = all_clips(args.splits, args.per_speaker)
    print(f"[fusion data] {len(clips)} clips, {sum(c['duration'] for c in clips) / 3600:.1f} h of audio", flush=True)
    dev = benchmark_dev()
    if "asr" in args.stages:
        stage_asr(clips + dev, cfg)
    if "refs" in args.stages:
        stage_refs(clips, cfg, models)
    if "lm" in args.stages and not HELDOUT_LM.exists():
        HELDOUT_LM.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(ROOT / "experiments" / "train_lm.py"), "--exclude", HELDOUT,
                        "--out", str(HELDOUT_LM)], check=True)
    if "features" in args.stages:
        stage_features(clips, dev, cfg, models, args.holdout)


if __name__ == "__main__":
    main()

"""Download evaluation data and build long-form clips — imp.md 5.1.

* LibriSpeech dev-clean / test-clean / test-other (OpenSLR 12, CC BY 4.0)
* LibriSpeech-PC manifests (OpenSLR 145, CC BY 4.0): punctuated + cased reference text
* distil-whisper/tedlium-long-form (TED-LIUM 3 talks, CC BY-NC-ND 3.0): lowercase, no punctuation

LibriSpeech clips: consecutive utterances of one chapter concatenated (0.4 s silence between
them) up to `max_minutes`; utterance offsets are kept so MFA can align per utterance.

Usage:  python tools/fetch_data.py --libri dev-clean test-clean test-other --ted validation test
        python tools/fetch_data.py --clips --libri-n 8 --ted-n validation=4 test=11
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subfix.paths import DATA_DIR  # noqa: E402

RAW = DATA_DIR / "raw"
CLIPS = DATA_DIR / "clips"
SR = 16000
OPENSLR_MIRRORS = ["https://openslr.trmal.net", "https://openslr.elda.org", "https://www.openslr.org"]
LIBRI_SIZES = {"dev-clean": 337926286, "test-clean": 346663984, "test-other": 328757843,
               "train-clean-100": 6387309499}   # train-clean-100: our CTC aligner (experiments/train_aligner.py)


# ------------------------------------------------------------------ downloads
def download(urls: list[str], dest: Path, size: int | None = None, tries: int = 50) -> Path:
    """Resumable download (HTTP Range) with retries across mirrors."""
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        url = urls[attempt % len(urls)]
        have = dest.stat().st_size if dest.exists() else 0
        if size and have >= size:
            return dest
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:      # already complete
                    return dest
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                total = size or int(r.headers.get("Content-Length", 0)) + (have if mode == "ab" else 0)
                with open(dest, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if not size or dest.stat().st_size >= size:
                return dest
        except Exception as exc:  # network resets are common here
            print(f"  download retry {attempt + 1}: {exc}", flush=True)
            time.sleep(min(30, 2 + attempt))
    raise RuntimeError(f"could not download {dest.name}")


def fetch_librispeech(splits: list[str]) -> None:
    base = RAW / "librispeech"
    for split in splits:
        out_dir = base / "LibriSpeech" / split
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[libri] {split}: already extracted")
            continue
        arch = base / f"{split}.tar.gz"
        urls = [f"{m}/resources/12/{split}.tar.gz" for m in OPENSLR_MIRRORS]
        print(f"[libri] downloading {split}")
        download(urls, arch, LIBRI_SIZES.get(split))
        print(f"[libri] extracting {split}")
        with tarfile.open(arch) as tf:
            tf.extractall(base, filter="data")
    pc = base / "pc"
    if not (pc / "test-clean.json").exists():
        arch = base / "librispeech_pc_manifests.tar.gz"
        download([f"{m}/resources/145/manifests.tar.gz" for m in OPENSLR_MIRRORS], arch, 25892530)
        pc.mkdir(parents=True, exist_ok=True)
        with tarfile.open(arch) as tf:
            tf.extractall(pc, filter="data")


def fetch_tedlium(splits: list[str]) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    repo = "distil-whisper/tedlium-long-form"
    out = RAW / "tedlium"
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    for split in splits:
        sdir = out / split
        if sdir.exists() and list(sdir.glob("*.json")):
            print(f"[ted] {split}: already extracted")
            continue
        sdir.mkdir(parents=True, exist_ok=True)
        shards = sorted(f for f in files if f.endswith(".parquet") and f"/{split}-" in f"/{Path(f).name}")
        if not shards:
            shards = sorted(f for f in files if f.endswith(".parquet") and split in f)
        for shard in shards:
            for attempt in range(20):
                try:
                    local = hf_hub_download(repo, shard, repo_type="dataset")
                    break
                except Exception as exc:
                    print(f"  hf retry {attempt + 1}: {exc}", flush=True)
                    time.sleep(5)
            else:
                raise RuntimeError(f"could not fetch {shard}")
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=1):       # one talk at a time (low memory)
                for row in batch.to_pylist():
                    spk = row.get("speaker_id") or f"talk{len(list(sdir.glob('*.json')))}"
                    audio = row["audio"]
                    data = audio.get("bytes") if isinstance(audio, dict) else audio
                    wav_path = sdir / f"{spk}.wav"
                    _write_audio_bytes(data, wav_path)
                    (sdir / f"{spk}.json").write_text(json.dumps({"speaker": spk, "text": row["text"]}),
                                                     encoding="utf-8")
                    print(f"[ted] {split}/{spk}", flush=True)


def _write_audio_bytes(data: bytes, dest: Path) -> None:
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR)
        audio = resample_poly(audio, SR // g, sr // g).astype(np.float32)
    sf.write(str(dest), audio, SR, subtype="PCM_16")


# ------------------------------------------------------------------ clip building
def _libri_transcripts(split_dir: Path) -> dict[str, str]:
    out = {}
    for f in split_dir.rglob("*.trans.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            uid, _, txt = line.partition(" ")
            out[uid] = txt.strip()
    return out


def _pc_manifest(split: str) -> dict[str, dict]:
    path = RAW / "librispeech" / "pc" / f"{split}.json"
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            uid = Path(d["audio_filepath"]).stem
            out[uid] = d
    return out


def build_libri_clips(split: str, n_clips: int, max_minutes: float = 5.0, min_minutes: float = 3.0,
                      gap_s: float = 0.4, seed: int = 13) -> list[dict]:
    import soundfile as sf
    sdir = RAW / "librispeech" / "LibriSpeech" / split
    pc = _pc_manifest(split)
    raw = _libri_transcripts(sdir)
    chapters: dict[tuple[str, str], list[str]] = {}
    for uid in sorted(pc):
        spk, chap, _ = uid.split("-")
        chapters.setdefault((spk, chap), []).append(uid)
    eligible = [k for k, v in chapters.items() if sum(pc[u]["duration"] for u in v) >= min_minutes * 60]
    rng = random.Random(seed)
    # one chapter per speaker first, for speaker diversity
    by_spk: dict[str, list] = {}
    for k in eligible:
        by_spk.setdefault(k[0], []).append(k)
    spk_order = sorted(by_spk)
    rng.shuffle(spk_order)
    chosen = [sorted(by_spk[s])[0] for s in spk_order][:n_clips]
    out_dir = CLIPS / f"libri-{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for spk, chap in sorted(chosen):
        cid = f"libri-{split}-{spk}-{chap}"
        meta_path = out_dir / f"{spk}-{chap}.json"
        if meta_path.exists():
            clips.append(json.loads(meta_path.read_text(encoding="utf-8")))
            continue
        pieces, utts, t = [], [], 0.0
        silence = np.zeros(int(gap_s * SR), dtype=np.float32)
        for uid in chapters[(spk, chap)]:
            dur = pc[uid]["duration"]
            if t + dur > max_minutes * 60 and utts:
                break
            audio, sr = sf.read(str(sdir / spk / chap / f"{uid}.flac"), dtype="float32")
            assert sr == SR
            if utts:
                pieces.append(silence)
                t += gap_s
            utts.append({"id": uid, "start": round(t, 3), "end": round(t + len(audio) / SR, 3),
                         "text": pc[uid]["text"], "raw": raw.get(uid, "")})
            pieces.append(audio)
            t += len(audio) / SR
        wav = out_dir / f"{spk}-{chap}.wav"
        sf.write(str(wav), np.concatenate(pieces), SR, subtype="PCM_16")
        meta = {"id": cid, "dataset": f"libri-{split}", "audio": str(wav.relative_to(ROOT)).replace("\\", "/"),
                "duration": round(t, 3), "has_punct": True,
                "text": " ".join(u["text"] for u in utts), "utterances": utts}
        meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        clips.append(meta)
        print(f"[clips] {cid}: {len(utts)} utts, {t / 60:.1f} min")
    return clips


def build_ted_clips(split: str, n: int) -> list[dict]:
    import soundfile as sf
    sdir = RAW / "tedlium" / split
    out_dir = CLIPS / f"ted-{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for meta_in in sorted(sdir.glob("*.json"))[:n]:
        d = json.loads(meta_in.read_text(encoding="utf-8"))
        spk = d["speaker"]
        wav_in = sdir / f"{spk}.wav"
        info = sf.info(str(wav_in))
        cid = f"ted-{split}-{spk}"
        meta = {"id": cid, "dataset": f"ted-{split}", "audio": str(wav_in.relative_to(ROOT)).replace("\\", "/"),
                "duration": round(info.frames / info.samplerate, 3), "has_punct": False,
                "text": " ".join(d["text"].split()), "utterances": None}
        (out_dir / f"{spk}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        clips.append(meta)
    return clips


def build_own_clips() -> list[dict]:
    """Your own recordings: data/own/<name>.(wav|flac|mp3|m4a|mp4) + <name>.txt (verbatim transcript with
    punctuation and casing). Treated as an extra test set ("own"); references come from MFA like TED."""
    import soundfile as sf
    sys.path.insert(0, str(ROOT / "src"))
    from subfix.preprocess import to_wav
    own = DATA_DIR / "own"
    out_dir = CLIPS / "own"
    clips = []
    for txt in sorted(own.glob("*.txt")):
        media = next((m for ext in (".wav", ".flac", ".mp3", ".m4a", ".mp4") for m in [txt.with_suffix(ext)]
                      if m.exists()), None)
        if media is None or txt.name.lower() == "readme.txt":
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        wav = Path(to_wav(media, out_dir))
        info = sf.info(str(wav))
        meta = {"id": f"own-{txt.stem}", "dataset": "own", "audio": str(wav.relative_to(ROOT)).replace("\\", "/"),
                "duration": round(info.frames / info.samplerate, 3), "has_punct": True,
                "text": " ".join(txt.read_text(encoding="utf-8").split()), "utterances": None}
        (out_dir / f"{txt.stem}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        clips.append(meta)
        print(f"[own] {meta['id']}: {meta['duration'] / 60:.1f} min")
    return clips


def all_clips() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CLIPS.glob("*/*.json"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri", nargs="*", default=[], help="LibriSpeech splits to download")
    ap.add_argument("--ted", nargs="*", default=[], help="TED long-form splits (validation, test)")
    ap.add_argument("--clips", action="store_true", help="build long-form clips")
    ap.add_argument("--libri-n", type=int, default=8)
    ap.add_argument("--libri-minutes", type=float, default=5.0)
    ap.add_argument("--ted-n", nargs="*", default=["validation=4", "test=11"])
    ap.add_argument("--own", action="store_true", help="register data/own/*.wav + *.txt as clips")
    args = ap.parse_args()
    if args.own:
        build_own_clips()
    if args.libri:
        fetch_librispeech(args.libri)
    if args.ted:
        fetch_tedlium(args.ted)
    if args.clips:
        for split in ("dev-clean", "test-clean", "test-other"):
            if (RAW / "librispeech" / "LibriSpeech" / split).exists():
                build_libri_clips(split, args.libri_n, args.libri_minutes)
        for spec in args.ted_n:
            split, _, n = spec.partition("=")
            if (RAW / "tedlium" / split).exists():
                build_ted_clips(split, int(n or 100))


if __name__ == "__main__":
    main()

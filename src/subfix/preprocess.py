"""Audio preprocessing: any media -> 16 kHz mono PCM WAV (ffmpeg, PyAV fallback)."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from .paths import cache_dir
from .utils.cache import file_hash

SR = 16000


@lru_cache(maxsize=1)
def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [os.path.join(local, "Microsoft", "WinGet", "Links", "ffmpeg.exe")]
    candidates += glob.glob(os.path.join(local, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*",
                                         "*", "bin", "ffmpeg.exe"))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def ffmpeg_dir() -> str | None:
    exe = find_ffmpeg()
    return os.path.dirname(exe) if exe else None


def _is_target_wav(path: Path, sr: int) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.samplerate == sr and info.channels == 1 and info.subtype == "PCM_16"
    except Exception:
        return False


def to_wav(media_path: str | Path, out_dir: str | Path | None = None, sr: int = SR) -> str:
    """Convert media to 16 kHz mono WAV; returns the cached WAV path."""
    src = Path(media_path)
    if not src.exists():
        raise FileNotFoundError(src)
    if _is_target_wav(src, sr):
        return str(src)
    out_root = Path(out_dir) if out_dir else cache_dir("wav")
    out_root.mkdir(parents=True, exist_ok=True)
    out = out_root / f"{src.stem}-{file_hash(src)}.wav"
    if out.exists() and out.stat().st_size > 44:
        return str(out)
    exe = find_ffmpeg()
    if exe:
        cmd = [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
               "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and out.exists():
            return str(out)
    audio = _decode_pyav(src, sr)
    import soundfile as sf
    sf.write(str(out), audio, sr, subtype="PCM_16")
    return str(out)


def _decode_pyav(path: Path, sr: int) -> np.ndarray:
    import av
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError(f"{path.name} has no audio track - upload a video/audio file with sound")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray().reshape(-1))
        for rf in resampler.resample(None):
            chunks.append(rf.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return (np.concatenate(chunks).astype(np.float32) / 32768.0)


def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Float32 mono waveform at `sr` (converts first if needed)."""
    import soundfile as sf
    p = Path(path)
    if not _is_target_wav(p, sr):
        p = Path(to_wav(p, sr=sr))
    audio, file_sr = sf.read(str(p), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:  # pragma: no cover - to_wav guarantees sr
        raise ValueError(f"unexpected sample rate {file_sr}")
    return audio


def audio_duration(path: str | Path) -> float:
    import soundfile as sf
    try:
        info = sf.info(str(path))
        return info.frames / float(info.samplerate)
    except Exception:
        return len(load_audio(path)) / float(SR)

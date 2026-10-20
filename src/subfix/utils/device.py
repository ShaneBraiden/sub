"""Device selection and Windows CUDA DLL discovery for CTranslate2 (faster-whisper)."""
from __future__ import annotations

import gc
import glob
import os
import sys
from functools import lru_cache

_DLLS_DONE = False


def setup_cuda_dlls() -> None:
    """Expose cuBLAS/cuDNN DLLs shipped with torch / nvidia-* wheels to CTranslate2.

    CTranslate2 loads cublas64_12.dll and cudnn*64_9.dll lazily; on Windows they are not on
    PATH unless a CUDA toolkit is installed. Importing torch first loads its own copies, and
    the nvidia pip wheels provide a second source.
    """
    global _DLLS_DONE
    if _DLLS_DONE or sys.platform != "win32":
        return
    _DLLS_DONE = True
    dirs: list[str] = []
    try:
        import torch  # noqa: F401  (loads torch/lib DLLs into the process)
        dirs.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception:
        pass
    try:
        import nvidia  # namespace package from nvidia-cublas-cu12 / nvidia-cudnn-cu12
        for base in list(getattr(nvidia, "__path__", [])):
            dirs.extend(glob.glob(os.path.join(base, "*", "bin")))
    except Exception:
        pass
    for d in dirs:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(pref: str = "auto") -> str:
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if cuda_available() else "cpu"
    return "cuda" if cuda_available() else "cpu"


def torch_dtype(device: str):
    import torch
    return torch.float16 if device == "cuda" else torch.float32


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

"""Repository-relative paths (the package is installed in editable mode from the repo)."""
from __future__ import annotations

import os
from pathlib import Path


def _find_root() -> Path:
    env = os.environ.get("SUBFIX_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "default.yaml").exists():
            return parent
    return here.parents[2]


ROOT = _find_root()
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
LEXICON_DIR = DATA_DIR / "lexicons"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def cache_dir(sub: str = "") -> Path:
    base = Path(os.environ.get("SUBFIX_CACHE", ROOT / "cache"))
    path = base / sub if sub else base
    path.mkdir(parents=True, exist_ok=True)
    return path

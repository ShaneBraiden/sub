"""HTTP backend for the subtitle corrector (FastAPI): a job queue around subfix.pipeline.run,
a REST + Server-Sent Events API, and static hosting of the React frontend (frontend/dist).

python -m app.backend            # http://127.0.0.1:8000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "1.0.0"

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT):        # works without the editable install too
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

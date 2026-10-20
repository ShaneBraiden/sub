"""Server settings: defaults < environment (SUBFIX_SERVER_*) < CLI flags."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from subfix.paths import ROOT, cache_dir

MEDIA_EXTS = frozenset({
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma",
    ".mp4", ".m4v", ".mkv", ".mov", ".avi", ".webm", ".mpg", ".mpeg", ".ts", ".wmv",
})
SUB_EXTS = frozenset({".srt", ".vtt"})


def _env(name: str, default: str) -> str:
    return os.environ.get(f"SUBFIX_SERVER_{name}", default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: cache_dir("server"))
    frontend_dir: Path = ROOT / "frontend" / "dist"
    max_upload_mb: int = 2048
    retention_days: float = 30.0          # finished jobs older than this are deleted at start-up (0 = keep)
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")   # Vite dev server

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        s = cls(
            data_dir=Path(_env("DATA_DIR", str(cache_dir("server")))),
            frontend_dir=Path(_env("FRONTEND_DIR", str(ROOT / "frontend" / "dist"))),
            max_upload_mb=int(_env("MAX_UPLOAD_MB", "2048")),
            retention_days=float(_env("RETENTION_DAYS", "30")),
            cors_origins=tuple(o.strip() for o in _env("CORS_ORIGINS", ",".join(cls.cors_origins)).split(",")
                               if o.strip()),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(s, k, v)
        s.data_dir = Path(s.data_dir)
        s.frontend_dir = Path(s.frontend_dir)
        s.data_dir.mkdir(parents=True, exist_ok=True)
        return s

    @property
    def jobs_dir(self) -> Path:
        p = self.data_dir / "jobs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jobs.sqlite3"

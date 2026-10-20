"""Job history in SQLite (stdlib), safe to use from the worker thread and the request handlers."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_COLUMNS = {
    "id": "TEXT PRIMARY KEY",
    "created": "REAL NOT NULL",
    "started": "REAL",
    "finished": "REAL",
    "status": "TEXT NOT NULL",
    "stage": "TEXT",
    "progress": "REAL NOT NULL DEFAULT 0",
    "media_name": "TEXT NOT NULL",
    "media_file": "TEXT NOT NULL",
    "subs_name": "TEXT",
    "subs_file": "TEXT",
    "mode": "TEXT NOT NULL",
    "options": "TEXT NOT NULL",       # JSON
    "error": "TEXT",
    "summary": "TEXT",                # JSON, set when the job is done
    "edited": "REAL",                 # time of the last manual cue edit
}
_JSON = ("options", "summary")


class JobStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        cols = ", ".join(f"{k} {v}" for k, v in _COLUMNS.items())
        self._db.execute(f"CREATE TABLE IF NOT EXISTS jobs ({cols})")
        have = {r["name"] for r in self._db.execute("PRAGMA table_info(jobs)")}
        for k, v in _COLUMNS.items():          # forward-compatible schema upgrades
            if k not in have:
                self._db.execute(f"ALTER TABLE jobs ADD COLUMN {k} {v.replace('PRIMARY KEY', '')}")
        self._db.execute("CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created)")

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        for k in _JSON:
            d[k] = json.loads(d[k]) if d.get(k) else ({} if k == "options" else None)
        return d

    def create(self, **fields: Any) -> dict[str, Any]:
        fields.setdefault("created", time.time())
        fields.setdefault("status", "queued")
        for k in _JSON:
            if k in fields and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k])
        keys = list(fields)
        with self._lock:
            self._db.execute(f"INSERT INTO jobs ({', '.join(keys)}) VALUES ({', '.join('?' * len(keys))})",
                             [fields[k] for k in keys])
        return self.get(fields["id"])

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row(row)

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        for k in _JSON:
            if k in fields and fields[k] is not None and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k])
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE jobs SET {sets} WHERE id = ?", [*fields.values(), job_id])

    def list(self, limit: int = 50, offset: int = 0, status: str | None = None) -> tuple[list[dict], int]:
        where, args = ("WHERE status = ?", [status]) if status else ("", [])
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM jobs {where}", args).fetchone()[0]
            rows = self._db.execute(f"SELECT * FROM jobs {where} ORDER BY created DESC LIMIT ? OFFSET ?",
                                    [*args, limit, offset]).fetchall()
        return [self._row(r) for r in rows], int(total)

    def with_status(self, *statuses: str) -> list[dict[str, Any]]:
        marks = ", ".join("?" * len(statuses))
        with self._lock:
            rows = self._db.execute(f"SELECT * FROM jobs WHERE status IN ({marks}) ORDER BY created",
                                    statuses).fetchall()
        return [self._row(r) for r in rows]

    def finished_before(self, t: float) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs WHERE status IN ('done', 'failed', 'cancelled') "
                                    "AND COALESCE(finished, created) < ?", (t,)).fetchall()
        return [self._row(r) for r in rows]

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def close(self) -> None:
        with self._lock:
            self._db.close()

"""Job queue: one worker thread runs jobs in order (a 6 GB GPU fits one pipeline at a time) and
publishes live progress that the API serves as JSON snapshots and Server-Sent Events."""
from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .runner import STAGE_LABELS, Cancelled
from .schemas import TERMINAL
from .settings import Settings
from .store import JobStore

log = logging.getLogger(__name__)

RESULT_FILE = "result.json"
EDITED_FILE = "edited_cues.json"
LOG_FILE = "run.log"


class JobManager:
    def __init__(self, store: JobStore, settings: Settings, runner):
        self.store = store
        self.settings = settings
        self.runner = runner
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._live: dict[str, dict[str, Any]] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self.current: str | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._cleanup_old()
        for job in self.store.with_status("queued", "running"):    # left over from the last server run
            note = "Re-queued after a server restart" if job["status"] == "running" else None
            self.store.update(job["id"], status="queued", stage=note, progress=0.0, started=None)
            self._enqueue(job["id"])
        self._thread = threading.Thread(target=self._loop, name="subfix-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """A job still running at shutdown stays 'running' in the store and is re-queued next start."""
        self._stopping.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout)

    def _cleanup_old(self) -> None:
        days = self.settings.retention_days
        if days <= 0:
            return
        for job in self.store.finished_before(time.time() - days * 86400):
            self._remove_files(job["id"])
            self.store.delete(job["id"])
            log.info("deleted job %s (older than %g days)", job["id"], days)

    # ------------------------------------------------------------------ paths
    def folder(self, job_id: str) -> Path:
        return self.settings.jobs_dir / job_id

    def _remove_files(self, job_id: str) -> None:
        shutil.rmtree(self.folder(job_id), ignore_errors=True)

    # ------------------------------------------------------------------ public API
    def submit(self, job_id: str) -> None:
        self._enqueue(job_id)

    def _enqueue(self, job_id: str) -> None:
        with self._lock:
            self._live[job_id] = {"stages": [], "version": 0}
            self._cancel[job_id] = threading.Event()
        self._queue.put(job_id)

    def queued_ids(self) -> list[str]:
        return [j["id"] for j in self.store.with_status("queued")]

    def cancel(self, job_id: str) -> bool:
        """Queued jobs are cancelled at once; a running job stops at its next stage boundary."""
        job = self.store.get(job_id)
        if job is None or job["status"] in TERMINAL:
            return False
        if job["status"] == "queued":
            self.store.update(job_id, status="cancelled", stage="Cancelled", finished=time.time())
            self._bump(job_id)
            return True
        self._cancel.setdefault(job_id, threading.Event()).set()
        with self._lock:
            live = self._live.get(job_id)
            if live is not None:
                live["cancelling"] = True
        self._bump(job_id)
        return True

    def delete(self, job_id: str) -> None:
        self._remove_files(job_id)
        self.store.delete(job_id)
        with self._lock:
            self._live.pop(job_id, None)
            self._cancel.pop(job_id, None)

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        job = self.store.get(job_id)
        if job is None:
            return None
        return self._decorate(job)

    def snapshots(self, jobs: list[dict]) -> list[dict]:
        queued = self.queued_ids()
        return [self._decorate(j, queued) for j in jobs]

    def _decorate(self, job: dict, queued: list[str] | None = None) -> dict[str, Any]:
        out = {k: job[k] for k in ("id", "status", "stage", "progress", "created", "started", "finished",
                                   "media_name", "subs_name", "mode", "options", "error", "summary", "edited")}
        with self._lock:
            live = self._live.get(job["id"])
            out["stages"] = [dict(s) for s in live["stages"]] if live else []
            out["stage_key"] = live.get("stage_key") if live else None
            out["cancelling"] = bool(live and live.get("cancelling")) and job["status"] == "running"
            out["version"] = live["version"] if live else 0
        end = job["finished"] or time.time()
        out["elapsed"] = round(end - job["started"], 1) if job["started"] else None
        if job["status"] == "queued":
            queued = queued if queued is not None else self.queued_ids()
            pos = queued.index(job["id"]) if job["id"] in queued else 0
            out["queue_position"] = pos + (1 if self.current else 0)
        else:
            out["queue_position"] = None
        return out

    def _bump(self, job_id: str) -> None:
        with self._lock:
            live = self._live.get(job_id)
            if live is not None:
                live["version"] += 1

    # ------------------------------------------------------------------ results
    def result(self, job_id: str) -> dict | None:
        path = self.folder(job_id) / RESULT_FILE
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        edited = self.folder(job_id) / EDITED_FILE
        data["edited_cues"] = json.loads(edited.read_text(encoding="utf-8")) if edited.exists() else None
        return data

    def save_result(self, job_id: str, result: dict) -> None:
        (self.folder(job_id) / RESULT_FILE).write_text(json.dumps(result, separators=(",", ":")),
                                                       encoding="utf-8")

    # ------------------------------------------------------------------ worker
    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None or self._stopping.is_set():
                return
            job = self.store.get(job_id)
            if job is None or job["status"] != "queued":
                continue
            self.current = job_id
            try:
                self._run(job)
            finally:
                self.current = None
                for other in self.queued_ids():   # queue positions changed
                    self._bump(other)

    def _set_stage(self, job_id: str, key: str, frac: float) -> None:
        now = time.time()
        label = STAGE_LABELS.get(key, key)
        with self._lock:
            live = self._live.setdefault(job_id, {"stages": [], "version": 0})
            stages = live["stages"]
            if stages and stages[-1]["key"] == key:
                return
            if stages and stages[-1].get("seconds") is None:
                stages[-1]["seconds"] = round(now - stages[-1]["started"], 2)
            stages.append({"key": key, "label": label, "started": now, "seconds": None})
            live["stage_key"] = key
            live["version"] += 1
        self.store.update(job_id, stage=label, progress=float(frac))

    def _close_stages(self, job_id: str) -> None:
        with self._lock:
            live = self._live.get(job_id)
            if live and live["stages"] and live["stages"][-1].get("seconds") is None:
                last = live["stages"][-1]
                last["seconds"] = round(time.time() - last["started"], 2)

    def _run(self, job: dict) -> None:
        job_id = job["id"]
        folder = self.folder(job_id)
        cancel = self._cancel.setdefault(job_id, threading.Event())
        self.store.update(job_id, status="running", started=time.time(), stage="Starting", progress=0.0,
                          error=None)
        self._bump(job_id)

        def progress(key: str, frac: float) -> None:
            if cancel.is_set():
                raise Cancelled()
            self._set_stage(job_id, key, frac)

        handler = logging.FileHandler(folder / LOG_FILE, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        handler.setLevel(logging.INFO)
        watched = [logging.getLogger("subfix"), logging.getLogger("app.backend")]
        for lg in watched:
            lg.addHandler(handler)
        t0 = time.time()
        log.info("job %s: %s%s", job_id, job["media_name"], f" + {job['subs_name']}" if job["subs_name"] else "")
        try:
            subs = folder / job["subs_file"] if job["subs_file"] else None
            result = self.runner.run(job, folder / job["media_file"], subs, folder, progress)
            self._close_stages(job_id)
            self.save_result(job_id, result)
            summary = {
                "audio_duration": result.get("audio_duration"),
                "rtf": result.get("rtf"),
                "cues": len(result.get("cues", [])),
                "counts": result.get("counts", {}),
                "sync_note": (result.get("sync") or {}).get("note"),
                "chapters": len(result.get("chapters") or []),
            }
            self.store.update(job_id, status="done", stage="Done", progress=1.0, finished=time.time(),
                              mode=result.get("mode", job["mode"]), summary=summary)
            log.info("job %s: done in %.1f s", job_id, time.time() - t0)
        except Cancelled:
            self._close_stages(job_id)
            self.store.update(job_id, status="cancelled", stage="Cancelled", finished=time.time())
            log.info("job %s: cancelled", job_id)
        except Exception as exc:
            self._close_stages(job_id)
            log.error("job %s failed:\n%s", job_id, traceback.format_exc())
            self.store.update(job_id, status="failed", stage="Failed", finished=time.time(),
                              error=f"{type(exc).__name__}: {exc}")
        finally:
            for lg in watched:
                lg.removeHandler(handler)
            handler.close()
            with self._lock:
                live = self._live.get(job_id)
                if live is not None:
                    live["cancelling"] = False
            self._bump(job_id)

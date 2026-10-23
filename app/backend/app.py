"""FastAPI application: REST + Server-Sent Events API under /api, and the built React frontend.

  GET    /api/system                 device, loaded models, queue
  GET    /api/options                option defaults (from configs/), ranges, pipeline stages
  POST   /api/jobs                   multipart upload: media, optional subs, options (JSON)
  GET    /api/jobs                   history (newest first)
  GET    /api/jobs/{id}              status snapshot
  GET    /api/jobs/{id}/events       live status as Server-Sent Events
  POST   /api/jobs/{id}/cancel       cancel (a running job stops at its next stage)
  POST   /api/jobs/{id}/rerun        same files, new options -> new job
  DELETE /api/jobs/{id}              delete the job and its files
  GET    /api/jobs/{id}/result       cues, diff rows, edits, metrics
  PUT    /api/jobs/{id}/cues         save manually edited cues (-> *.edited.srt / .vtt)
  DELETE /api/jobs/{id}/cues         drop the manual edits
  GET    /api/jobs/{id}/files/{kind} downloads: srt, vtt, json, html, edited_srt, edited_vtt,
                                     rttm + speakers (diarized jobs), chapters_vtt, chapters_txt,
                                     chapters_json, ffmetadata (media with 2+ chapters)
  GET    /api/jobs/{id}/media        the uploaded media (supports Range requests)
  GET    /api/jobs/{id}/audio        the 16 kHz WAV the pipeline used
  GET    /api/jobs/{id}/waveform     peak envelope for the timeline
  GET    /api/jobs/{id}/log          the run log
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import ValidationError

from . import __version__
from .jobs import EDITED_FILE, LOG_FILE, JobManager
from .runner import STAGES, PipelineRunner, cue_json, waveform_peaks
from .schemas import TERMINAL, CuesIn, JobOptions, option_fields
from .settings import MEDIA_EXTS, SUB_EXTS, Settings
from .store import JobStore

log = logging.getLogger(__name__)

DOWNLOAD_TYPES = {"srt": "application/x-subrip", "vtt": "text/vtt", "json": "application/json",
                  "html": "text/html", "rttm": "text/plain", "speakers": "application/json",
                  "txt": "text/plain", "ffmetadata": "text/plain"}
_UNSAFE = re.compile(r"[^\w.\- ]+")


def safe_name(name: str | None, fallback: str = "upload") -> str:
    base = _UNSAFE.sub("_", Path(name or fallback).name).strip(" .")
    return base[:120] or fallback


def create_app(settings: Settings | None = None, runner=None) -> FastAPI:
    settings = settings or Settings.from_env()
    runner = runner or PipelineRunner()
    for name in ("app.backend", "subfix"):      # INFO records feed each job's run.log
        if logging.getLogger(name).level == logging.NOTSET:
            logging.getLogger(name).setLevel(logging.INFO)
    store = JobStore(settings.db_path)
    manager = JobManager(store, settings, runner)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.start()
        yield
        manager.stop()
        store.close()

    app = FastAPI(title="Subtitle correction API", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.state.settings, app.state.manager, app.state.runner = settings, manager, runner
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"],
                           allow_headers=["*"])

    api = APIRouter(prefix="/api")

    def get_job(job_id: str) -> dict:
        snap = manager.snapshot(job_id)
        if snap is None:
            raise HTTPException(404, "No such job (it may have been deleted).")
        return snap

    def job_file(job_id: str, name: str | None) -> Path:
        if not name:
            raise HTTPException(404, "File not found.")
        path = manager.folder(job_id) / name
        if not path.is_file():
            raise HTTPException(404, "File not found.")
        return path

    # ------------------------------------------------------------------ info
    @api.get("/system")
    def system():
        queued = manager.queued_ids()
        return {"version": __version__, "device": runner.device_info(), "models_loaded": runner.loaded_models(),
                "running": manager.current, "queued": len(queued)}

    @api.get("/options")
    def options():
        diarization = getattr(runner, "diarization", None)
        return {"defaults": runner.defaults(), "fields": option_fields(),
                "diarization": diarization() if diarization else {"model": False},
                "stages": [{"key": k, "label": label} for k, label in STAGES],
                "media_exts": sorted(MEDIA_EXTS), "subs_exts": sorted(SUB_EXTS),
                "max_upload_mb": settings.max_upload_mb}

    # ------------------------------------------------------------------ jobs
    @api.post("/jobs", status_code=201)
    def create_job(request: Request, media: UploadFile = File(...), subs: UploadFile | None = File(None),
                   options: str = Form("{}")):
        length = int(request.headers.get("content-length") or 0)
        if length > settings.max_upload_mb * 2**20:
            raise HTTPException(413, f"Upload is larger than {settings.max_upload_mb} MB.")
        try:
            opts = JobOptions(**json.loads(options or "{}"))
        except (ValueError, ValidationError) as exc:
            raise HTTPException(422, f"Invalid options: {exc}") from exc
        media_ext = Path(media.filename or "").suffix.lower()
        if media_ext not in MEDIA_EXTS:
            raise HTTPException(400, f"Unsupported media type '{media_ext or '?'}'. "
                                     f"Use one of: {', '.join(sorted(MEDIA_EXTS))}.")
        subs_ext = Path(subs.filename or "").suffix.lower() if subs is not None and subs.filename else None
        if subs is not None and subs.filename and subs_ext not in SUB_EXTS:
            raise HTTPException(400, "Subtitles must be an .srt or .vtt file.")
        if subs_ext is None and opts.segmentation == "keep":
            raise HTTPException(422, "'Keep the original cues' needs a subtitle file.")

        job_id = uuid.uuid4().hex[:12]
        folder = manager.folder(job_id)
        folder.mkdir(parents=True)
        try:
            media_file = "media" + media_ext
            with (folder / media_file).open("wb") as f:
                shutil.copyfileobj(media.file, f, 1 << 20)
            if (folder / media_file).stat().st_size == 0:
                raise HTTPException(400, "The media file is empty.")
            subs_file = None
            if subs_ext:
                subs_file = "subs" + subs_ext
                with (folder / subs_file).open("wb") as f:
                    shutil.copyfileobj(subs.file, f, 1 << 20)
                _check_subs(folder / subs_file)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        job = manager.store.create(
            id=job_id, media_name=safe_name(media.filename, "media" + media_ext), media_file=media_file,
            subs_name=safe_name(subs.filename) if subs_file else None, subs_file=subs_file,
            mode="A" if subs_file else "B", options=opts.model_dump(exclude_none=True),
            stage="Waiting in the queue")
        manager.submit(job_id)
        log.info("job %s created: %s%s", job_id, job["media_name"],
                 f" + {job['subs_name']}" if job["subs_name"] else "")
        return manager.snapshot(job_id)

    @api.get("/jobs")
    def list_jobs(limit: int = 50, offset: int = 0, status: str | None = None):
        jobs, total = manager.store.list(max(1, min(limit, 500)), max(0, offset), status)
        return {"jobs": manager.snapshots(jobs), "total": total}

    @api.get("/jobs/{job_id}")
    def job_status(job_id: str):
        return get_job(job_id)

    @api.get("/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        get_job(job_id)

        async def stream():
            last = None
            beat = time.monotonic()
            while True:
                if await request.is_disconnected():
                    return
                snap = manager.snapshot(job_id)
                if snap is None:
                    yield "event: deleted\ndata: {}\n\n"
                    return
                key = (snap["status"], snap["version"], snap["progress"], snap["queue_position"], snap["edited"])
                if key != last:
                    last = key
                    beat = time.monotonic()
                    yield f"data: {json.dumps(snap)}\n\n"
                if snap["status"] in TERMINAL:
                    return
                if time.monotonic() - beat > 15:
                    beat = time.monotonic()
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.4)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @api.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        snap = get_job(job_id)
        if snap["status"] in TERMINAL:
            raise HTTPException(409, f"The job is already {snap['status']}.")
        manager.cancel(job_id)
        return manager.snapshot(job_id)

    @api.post("/jobs/{job_id}/rerun", status_code=201)
    def rerun_job(job_id: str, options: JobOptions):
        old = get_job(job_id)
        src = manager.store.get(job_id)
        if options.segmentation == "keep" and not src["subs_file"]:
            raise HTTPException(422, "'Keep the original cues' needs a subtitle file.")
        new_id = uuid.uuid4().hex[:12]
        folder = manager.folder(new_id)
        folder.mkdir(parents=True)
        for name in (src["media_file"], src["subs_file"]):
            if name:
                _link_or_copy(job_file(job_id, name), folder / name)
        manager.store.create(id=new_id, media_name=old["media_name"], media_file=src["media_file"],
                             subs_name=old["subs_name"], subs_file=src["subs_file"], mode=old["mode"],
                             options=options.model_dump(exclude_none=True), stage="Waiting in the queue")
        manager.submit(new_id)
        return manager.snapshot(new_id)

    @api.delete("/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str):
        snap = get_job(job_id)
        if snap["status"] == "running":
            raise HTTPException(409, "The job is running. Cancel it first.")
        if snap["status"] == "queued":
            manager.cancel(job_id)
        manager.delete(job_id)

    # ------------------------------------------------------------------ results
    @api.get("/jobs/{job_id}/result")
    def job_result(job_id: str):
        snap = get_job(job_id)
        if snap["status"] != "done":
            raise HTTPException(409, f"The job is {snap['status']}; results exist once it is done.")
        data = manager.result(job_id)
        if data is None:
            raise HTTPException(404, "The result file is missing.")
        data.pop("wav", None)
        return JSONResponse(data)

    @api.put("/jobs/{job_id}/cues")
    def save_cues(job_id: str, body: CuesIn):
        snap = get_job(job_id)
        if snap["status"] != "done":
            raise HTTPException(409, "Only finished jobs can be edited.")
        from subfix.diarize import parse_labels
        from subfix.subio import Cue, write_srt, write_vtt

        result = manager.result(job_id) or {}
        # diarized job: speaker labels typed in the text decide each line's speaker (so a wrong
        # label can be fixed by editing it), and the edited .vtt gets matching voice tags
        known = [r["speaker"] for r in (result.get("speakers") or {}).get("speakers", [])]
        label_fmt = ((result.get("config") or {}).get("diarize") or {}).get("label_format", "{speaker}: ")
        prev = None
        cues = []
        for k, c in enumerate(sorted(body.cues, key=lambda c: (c.start, c.end))):
            lines = [ln.strip() for ln in c.text.replace("\r", "").split("\n") if ln.strip()]
            if not lines:
                raise HTTPException(422, f"Cue {k + 1} is empty.")
            if c.end <= c.start:
                raise HTTPException(422, f"Cue {k + 1} ends before it starts.")
            cue = Cue(len(cues) + 1, c.start, c.end, "\n".join(lines))
            if known:
                cue.speakers, cue.labels = parse_labels(lines, known, prev, label_fmt)
                prev = cue.speakers[-1]
            cues.append(cue)
        if not cues:
            raise HTTPException(422, "No cues to save.")
        folder = manager.folder(job_id)
        stem = Path(result.get("files", {}).get("srt", "subtitles.corrected.srt")).stem
        write_srt(cues, folder / f"{stem}.edited.srt")
        write_vtt(cues, folder / f"{stem}.edited.vtt")
        (folder / EDITED_FILE).write_text(json.dumps([cue_json(c) for c in cues]), encoding="utf-8")
        manager.store.update(job_id, edited=time.time())
        manager._bump(job_id)
        return {"cues": len(cues), "edited": True}

    @api.delete("/jobs/{job_id}/cues", status_code=204)
    def drop_edits(job_id: str):
        get_job(job_id)
        folder = manager.folder(job_id)
        for p in [folder / EDITED_FILE, *folder.glob("*.edited.srt"), *folder.glob("*.edited.vtt")]:
            p.unlink(missing_ok=True)
        manager.store.update(job_id, edited=None)
        manager._bump(job_id)

    @api.get("/jobs/{job_id}/files/{kind}")
    def download(job_id: str, kind: str, inline: bool = False):
        get_job(job_id)
        result = manager.result(job_id) or {}
        files = dict(result.get("files", {}))
        folder = manager.folder(job_id)
        for ext in ("srt", "vtt"):
            found = sorted(folder.glob(f"*.edited.{ext}"))
            if found:
                files[f"edited_{ext}"] = found[0].name
        if kind not in files:
            raise HTTPException(404, f"No '{kind}' file for this job.")
        path = job_file(job_id, files[kind])
        ext = kind.split("_")[-1]
        return FileResponse(path, media_type=DOWNLOAD_TYPES.get(ext), filename=path.name,
                            content_disposition_type="inline" if inline or ext == "html" else "attachment")

    @api.get("/jobs/{job_id}/media")
    def media(job_id: str):
        get_job(job_id)
        src = manager.store.get(job_id)
        path = job_file(job_id, src["media_file"])
        mime = mimetypes.guess_type(src["media_name"])[0] or mimetypes.guess_type(path.name)[0]
        return FileResponse(path, media_type=mime or "application/octet-stream")

    @api.get("/jobs/{job_id}/audio")
    def audio(job_id: str):
        get_job(job_id)
        wav = (manager.result(job_id) or {}).get("wav")
        if not wav or not Path(wav).is_file():
            raise HTTPException(404, "The converted audio is not available (cache cleared?).")
        return FileResponse(wav, media_type="audio/wav")

    @api.get("/jobs/{job_id}/waveform")
    def waveform(job_id: str, bins: int = 2400):
        get_job(job_id)
        bins = max(100, min(bins, 20000))
        cache = manager.folder(job_id) / f"waveform-{bins}.json"
        if cache.exists():
            return JSONResponse(json.loads(cache.read_text(encoding="utf-8")))
        wav = (manager.result(job_id) or {}).get("wav")
        if not wav or not Path(wav).is_file():
            raise HTTPException(404, "The converted audio is not available.")
        data = waveform_peaks(wav, bins)
        cache.write_text(json.dumps(data), encoding="utf-8")
        return data

    @api.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
    def run_log(job_id: str):
        get_job(job_id)
        path = manager.folder(job_id) / LOG_FILE
        if not path.exists():
            return ""
        data = path.read_bytes()
        return data[-200_000:].decode("utf-8", errors="replace")

    app.include_router(api)
    _mount_frontend(app, settings)
    return app


def _check_subs(path: Path) -> None:
    from subfix.subio import read_subs
    try:
        cues = read_subs(path)
    except Exception as exc:
        raise HTTPException(400, f"Could not read the subtitle file: {exc}") from exc
    if not cues:
        raise HTTPException(400, "The subtitle file has no cues.")


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


_NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>Subtitle correction</title>
<body style="font:16px system-ui;max-width:640px;margin:64px auto;padding:0 16px">
<h1>The frontend is not built yet</h1>
<p>Build it once, then reload this page:</p>
<pre style="background:#f3f3f3;padding:12px;border-radius:8px">cd frontend
npm install
npm run build</pre>
<p>Or run the dev server (<code>npm run dev</code>) and open <a href="http://localhost:5173">localhost:5173</a>.
The API docs are at <a href="/api/docs">/api/docs</a>.</p></body>"""


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Serve frontend/dist with a single-page-app fallback to index.html."""
    dist = settings.frontend_dir.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "Unknown API route.")
        index = dist / "index.html"
        if not index.exists():
            return HTMLResponse(_NOT_BUILT)
        target = (dist / path).resolve()
        if path and target.is_file() and dist in target.parents:
            headers = {"Cache-Control": "public, max-age=31536000, immutable"} if "/assets/" in target.as_posix() else {}
            return FileResponse(target, headers=headers)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

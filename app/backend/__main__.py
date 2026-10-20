"""python -m app.backend [--host 127.0.0.1] [--port 8000] [--no-browser] [--reload]"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

from .settings import Settings


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m app.backend", description="Subtitle correction server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", help="jobs + history (default: cache/server)")
    ap.add_argument("--max-upload-mb", type=int)
    ap.add_argument("--retention-days", type=float, help="delete finished jobs older than this at start-up (0 = keep)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    ap.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for name in ("app.backend", "subfix"):
        logging.getLogger(name).setLevel(logging.INFO)

    # the settings are read again inside create_app (also in --reload workers), so pass CLI values via env
    for flag, env in (("data_dir", "DATA_DIR"), ("max_upload_mb", "MAX_UPLOAD_MB"),
                      ("retention_days", "RETENTION_DAYS")):
        value = getattr(args, flag)
        if value is not None:
            os.environ[f"SUBFIX_SERVER_{env}"] = str(value)
    settings = Settings.from_env()

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"Subtitle correction: {url}   API docs: {url}/api/docs   (Ctrl+C to stop)", flush=True)
    if not (settings.frontend_dir / "index.html").exists():
        print("  note: frontend/dist is missing - run `npm install` and `npm run build` in frontend/", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, [url]).start()
    if args.reload:
        root = Path(__file__).resolve().parents[2]
        uvicorn.run("app.backend.app:create_app", factory=True, host=args.host, port=args.port, reload=True,
                    reload_dirs=[str(root / "app" / "backend"), str(root / "src")], log_level="warning")
    else:
        from .app import create_app
        uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

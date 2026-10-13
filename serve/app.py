"""FastAPI demo service (CPU). Run: ``uvicorn serve.app:app --port 8000`` or the Dockerfile.

Endpoints (the product API: clips, uploads, tracking jobs, benchmarks, is in ``serve/api.py``)
* ``GET  /api/health``   status, version, configs found, detector models, clips, job counts.
* ``GET  /api/configs``  tracker configs that may be named in ``/api/track``.
* ``POST /api/track``    multipart: ``cache`` (a Stage A ``.npz`` detection cache, see
  ``tracker/io.py``), ``config`` (name from ``/api/configs``), optional ``gate`` (``x1,y1,x2,y2``
  in pixels; default: horizontal line at 60 % of the frame height, the ``h60`` rule),
  optional ``start_time`` (ISO 8601 wall time of frame 1; default ``1970-01-01T00:00:00+00:00``
  = replay time). Returns gate events (``serve/events.py``), a viewer clip and latency.
* ``GET  /api/workstation``, ``/api/results/*``, ``/api/viewer-clips``, ``/api/config/{name}``,
  ``/api/outputs``: read-only data for the research workstation (``serve/dashboard_api.py``).
* ``GET  /``             the web app (``web/dist``) when built, else the static viewer
  (``viewer/dist``); the viewer is also at ``/viewer/`` when both exist.

``/api/track`` is an offline replay: it takes the detector output an edge box would produce (a
Stage A cache) and stores nothing. The video jobs of ``serve/api.py`` run a CPU detector.
"""

from __future__ import annotations

import platform
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from eval.export_viewer import build_clip, pending_entries, results_to_rows
from eval.gate_events import Gate
from serve import errors
from serve.api import Service, router
from serve.dashboard_api import router as workstation_router
from serve.detector import MODELS
from serve.events import EventParams, gate_events
from serve.settings import Settings
from tracker.config import load_config
from tracker.io import CacheFormatError, load_cache
from tracker.runner import run_sequence

# shared with serve/dashboard_api.py (re-exported here so existing imports keep working)
from serve.catalog import (  # noqa: E402
    CONFIG_DIRS,
    CONFIG_ROOT,
    MAX_UPLOAD_MB,
    ROOT,
    VERSION,
    VIEWER_DIST,
    available_configs,
)

__all__ = ["CONFIG_DIRS", "CONFIG_ROOT", "ROOT", "app", "available_configs", "create_app"]


def parse_gate(text: str | None, width: int, height: int) -> Gate:
    if not text:
        y = 0.6 * height
        return Gate("h60", (0.0, y), (float(width), y))
    try:
        x1, y1, x2, y2 = (float(v) for v in text.split(","))
    except ValueError as err:
        raise HTTPException(422, "gate must be 'x1,y1,x2,y2'") from err
    if (x1, y1) == (x2, y2):
        raise HTTPException(422, "gate has zero length")
    return Gate("gate", (x1, y1), (x2, y2))


def parse_start(text: str | None) -> datetime:
    try:
        t = datetime.fromisoformat(text or "1970-01-01T00:00:00+00:00")
    except ValueError as err:
        raise HTTPException(422, "start_time must be ISO 8601") from err
    if t.tzinfo is None:
        raise HTTPException(422, "start_time needs a UTC offset")
    return t


def _save_upload(upload: UploadFile, dst: Path) -> None:
    limit, n = int(MAX_UPLOAD_MB * 2**20), 0
    with dst.open("wb") as fh:
        while chunk := upload.file.read(2**20):
            n += len(chunk)
            if n > limit:
                raise HTTPException(413, f"upload larger than {MAX_UPLOAD_MB:g} MB")
            fh.write(chunk)


class SPAStaticFiles(StaticFiles):
    """Static files with a single-page-app fallback: unknown paths serve ``index.html``."""

    async def get_response(self, path: str, scope: Any) -> Any:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as err:
            if err.status_code != 404 or path.startswith("api"):
                raise
            return FileResponse(Path(str(self.directory)) / "index.html")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(a: FastAPI) -> AsyncIterator[None]:
        yield
        a.state.service.manager.shutdown()

    desc = (
        "Identity-preserving gate tracking (Zed PS-2): CPU demo service. "
        "Benchmark numbers come from docs/results."
    )
    application = FastAPI(title="zed-ps2-tracker demo", version=VERSION, lifespan=lifespan,
                          description=desc)  # fmt: skip
    errors.install(application)
    if settings.cors_origins:
        application.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                                   allow_methods=["*"], allow_headers=["*"])  # fmt: skip
    application.state.service = Service(settings)
    application.state.settings = settings
    application.add_api_route("/api/health", health, methods=["GET"])
    application.add_api_route("/api/configs", configs, methods=["GET"])
    application.add_api_route("/api/track", track, methods=["POST"])
    application.include_router(router)
    application.include_router(workstation_router)

    web = settings.web_dist
    viewer = (VIEWER_DIST / "index.html").exists()
    if (web / "index.html").exists():
        if viewer:
            application.mount("/viewer", StaticFiles(directory=VIEWER_DIST, html=True), "viewer")
        application.mount("/", SPAStaticFiles(directory=web, html=True), name="web")
    elif viewer:
        application.mount("/", StaticFiles(directory=VIEWER_DIST, html=True), name="viewer")
    return application


def health(request: Request) -> dict[str, Any]:
    service: Service = request.app.state.service
    s = service.settings
    jobs = service.store.list()
    return {"status": "ok", "version": VERSION, "configs": len(available_configs()),
            "viewer": (VIEWER_DIST / "index.html").exists(),
            "web": (s.web_dist / "index.html").exists(),
            "detectors": {n: (s.model_dir / f"{n}.onnx").exists() for n in MODELS},
            "default_detector": s.detector,
            "clips": len(service.clips()),
            "jobs": {st: sum(j["status"] == st for j in jobs)
                     for st in ("queued", "running", "succeeded", "failed", "cancelled")},
            "limits": {"max_video_mb": s.max_video_mb,
                       "max_video_seconds": s.max_video_seconds,
                       "max_job_frames": s.max_job_frames,
                       "max_cache_mb": MAX_UPLOAD_MB}}  # fmt: skip


def configs() -> list[str]:
    return list(available_configs())


def track(
    cache: Annotated[UploadFile, File(description="Stage A .npz detection cache")],
    config: Annotated[str, Form()] = "baselines/bytetrack",
    gate: Annotated[str | None, Form()] = None,
    start_time: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    cfgs = available_configs()
    if config not in cfgs:
        raise HTTPException(422, f"unknown config {config!r}; see /api/configs")
    start = parse_start(start_time)
    with tempfile.TemporaryDirectory(prefix="zed-serve-") as tmp:
        path = Path(tmp) / "cache.npz"
        _save_upload(cache, path)
        try:
            seq = load_cache(path)
        except (CacheFormatError, ValueError, OSError, KeyError) as err:
            raise HTTPException(422, f"not a valid detection cache: {err}") from err
    m = seq.meta
    g = parse_gate(gate, m.im_width, m.im_height)
    try:
        run = run_sequence(seq, load_config(cfgs[config]))
    except ValueError as err:  # e.g. a config that needs embeddings the cache lacks
        raise HTTPException(422, f"{config} cannot run on this cache: {err}") from err
    rows = results_to_rows(run.results)
    events = gate_events(run.results, m.frame_rate, g, start, EventParams())
    clip = build_clip(m.sequence, config.replace("/", "-"), m.frame_rate, m.seq_length,
                      (m.im_width, m.im_height), rows, np.zeros((0, 7)), None, {},
                      pending_entries(run.results), False)  # fmt: skip
    clip["gate"] = {"a": list(g.a), "b": list(g.b)}
    clip["events"] = sorted(
        clip["events"]
        + [{"f": e["frame"], "type": "cross", "id": e["track_id"], "dir": e["direction"],
            "correct": None} for e in events if e["type"] == "crossing"]
        + [{"f": e["frame"], "type": "tailgate", "ids": [e["lead_track_id"], e["track_id"]]}
           for e in events if e["type"] == "tailgate"],
        key=lambda e: e["f"],
    )  # fmt: skip
    lat = np.asarray(run.latency_ms, float)
    timing = {"frames": len(lat), "machine": platform.machine(),
              "p50_ms": round(float(np.percentile(lat, 50)), 3) if len(lat) else None,
              "p95_ms": round(float(np.percentile(lat, 95)), 3) if len(lat) else None}  # fmt: skip
    return JSONResponse({"sequence": m.sequence, "config": config, "events": events,
                         "latency": timing, "clip": clip})  # fmt: skip


app = create_app()

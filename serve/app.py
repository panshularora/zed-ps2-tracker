"""FastAPI demo service (CPU). Run: ``uvicorn serve.app:app --port 8000`` or the Dockerfile.

Endpoints
* ``GET  /api/health``   status, version, configs found.
* ``GET  /api/configs``  tracker configs that may be named in ``/api/track``.
* ``POST /api/track``    multipart: ``cache`` (a Stage A ``.npz`` detection cache, see
  ``tracker/io.py``), ``config`` (name from ``/api/configs``), optional ``gate`` (``x1,y1,x2,y2``
  in pixels; default: horizontal line at 60 % of the frame height, the ``h60`` rule),
  optional ``start_time`` (ISO 8601 wall time of frame 1; default ``1970-01-01T00:00:00+00:00``
  = replay time). Returns gate events (``serve/events.py``), a viewer clip and latency.
* ``GET  /api/dashboard`` clips, published results, and whether the workstation UI is bundled.
* ``GET  /api/results/*`` ablation, gate-event and go/no-go JSON from ``docs/results/``.
* ``GET  /api/clips``    viewer clip list; ``GET /api/clips/{file}`` the gzipped clip.
* ``GET  /``             the built dashboard (``dashboard/dist``) when present, else the viewer.

This is an offline replay demo: frames are not decoded here (no detector on CPU); the input is
the detector output the edge box would produce. Nothing is stored; the upload is deleted after
the request.
"""

from __future__ import annotations

import platform
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from eval.export_viewer import build_clip, pending_entries, results_to_rows
from eval.gate_events import Gate
from serve.catalog import (
    DASHBOARD_DIST,
    MAX_UPLOAD_MB,
    VERSION,
    VIEWER_DIST,
    available_configs,
)
from serve.dashboard_api import router as dashboard_router
from serve.events import EventParams, gate_events
from tracker.config import load_config
from tracker.io import CacheFormatError, load_cache
from tracker.runner import run_sequence


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


app = FastAPI(title="zed-ps2-tracker demo", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": VERSION, "configs": len(available_configs()),
            "viewer": (VIEWER_DIST / "index.html").exists(),
            "dashboard": (DASHBOARD_DIST / "index.html").exists()}  # fmt: skip


@app.get("/api/configs")
def configs() -> list[str]:
    return list(available_configs())


@app.post("/api/track")
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


app.include_router(dashboard_router)

if (VIEWER_DIST / "index.html").exists():
    app.mount("/viewer", StaticFiles(directory=VIEWER_DIST, html=True), name="viewer")
if (DASHBOARD_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=DASHBOARD_DIST, html=True), name="dashboard")
elif (VIEWER_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=VIEWER_DIST, html=True), name="legacy-viewer")

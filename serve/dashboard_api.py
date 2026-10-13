"""Read-only dashboard API: published results, viewer clips, and tracker YAML.

These endpoints feed the MOT research workstation (``dashboard/``). They never run the tracker;
``POST /api/track`` stays the only mutating replay path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from serve.catalog import ROOT, VERSION, VIEWER_DIST, available_configs
from tracker.config import config_to_dict, load_config

router = APIRouter()


def results_root() -> Path:
    return Path(os.environ.get("ZED_RESULTS_ROOT", str(ROOT / "docs" / "results")))


def clip_root() -> Path:
    return Path(os.environ.get("ZED_CLIP_ROOT", str(ROOT / "viewer" / "data")))


def outputs_root() -> Path:
    return Path(os.environ.get("ZED_OUTPUTS_ROOT", str(ROOT / "outputs")))


def dashboard_dist() -> Path:
    return Path(os.environ.get("ZED_DASHBOARD_DIST", str(ROOT / "dashboard" / "dist")))


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise HTTPException(404, f"missing {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise HTTPException(500, f"invalid json in {path.name}") from err


def _safe_file(root: Path, name: str, *, suffix: str) -> Path:
    if "/" in name or "\\" in name or name.startswith(".") or ".." in name:
        raise HTTPException(404, "not found")
    if not name.endswith(suffix):
        raise HTTPException(404, "not found")
    root = root.resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(404, "not found")
    return path


@router.get("/api/dashboard")
def dashboard_meta() -> dict[str, Any]:
    clips = clip_root()
    n_clips = len(list(clips.glob("*.json.gz"))) if clips.is_dir() else 0
    return {
        "status": "ok",
        "version": VERSION,
        "configs": len(available_configs()),
        "clips": n_clips,
        "viewer": (VIEWER_DIST / "index.html").exists(),
        "dashboard": (dashboard_dist() / "index.html").exists(),
        "results": {
            "ablation": (results_root() / "ablation" / "ablation.json").is_file(),
            "gate_events": (results_root() / "gate_events" / "gate_events.json").is_file(),
            "go_no_go": (results_root() / "go_no_go" / "rows_v2.json").is_file(),
        },
    }


@router.get("/api/results/ablation")
def ablation() -> JSONResponse:
    return JSONResponse(_read_json(results_root() / "ablation" / "ablation.json"))


@router.get("/api/results/gate-events")
def gate_event_metrics() -> JSONResponse:
    return JSONResponse(_read_json(results_root() / "gate_events" / "gate_events.json"))


@router.get("/api/results/go-no-go")
def go_no_go() -> JSONResponse:
    return JSONResponse(_read_json(results_root() / "go_no_go" / "rows_v2.json"))


@router.get("/api/config/{name:path}")
def config_yaml(name: str) -> dict[str, Any]:
    cfgs = available_configs()
    if name not in cfgs:
        raise HTTPException(404, f"unknown config {name!r}")
    try:
        rel = str(cfgs[name].relative_to(ROOT))
    except ValueError:
        rel = name
    return {"name": name, "path": rel, "config": config_to_dict(load_config(cfgs[name]))}


@router.get("/api/clips")
def list_clips() -> dict[str, Any]:
    root = clip_root()
    items: list[dict[str, str]] = []
    if root.is_dir():
        for p in sorted(root.glob("*.json.gz")):
            stem = p.name[: -len(".json.gz")]
            clip, _, tracker = stem.partition(".")
            if not tracker:
                continue
            items.append({"file": p.name, "clip": clip, "tracker": tracker})
    return {"clips": items}


@router.get("/api/clips/{name}")
def get_clip(name: str) -> FileResponse:
    path = _safe_file(clip_root(), name, suffix=".json.gz")
    return FileResponse(path, media_type="application/gzip", filename=name)


@router.get("/api/outputs")
def list_outputs() -> dict[str, Any]:
    root = outputs_root()
    runs: list[dict[str, Any]] = []
    if root.is_dir():
        for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            metrics_root = split_dir / "metrics"
            if not metrics_root.is_dir():
                continue
            for tracker_dir in sorted(p for p in metrics_root.iterdir() if p.is_dir()):
                metrics_file = tracker_dir / "metrics.json"
                if not metrics_file.is_file():
                    continue
                data = _read_json(metrics_file)
                sequences = [k for k in data if k != "COMBINED"] if isinstance(data, dict) else []
                combined = data.get("COMBINED") if isinstance(data, dict) else None
                runs.append(
                    {
                        "split": split_dir.name,
                        "tracker": tracker_dir.name,
                        "sequences": sequences,
                        "combined": combined,
                    }
                )
    return {"runs": runs}

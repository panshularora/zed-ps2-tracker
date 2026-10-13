"""Shared paths and config names for the demo service."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = Path(os.environ.get("ZED_CONFIG_ROOT", ROOT / "configs"))
VIEWER_DIST = Path(os.environ.get("ZED_VIEWER_DIST", ROOT / "viewer" / "dist"))
DASHBOARD_DIST = Path(os.environ.get("ZED_DASHBOARD_DIST", ROOT / "dashboard" / "dist"))
MAX_UPLOAD_MB = float(os.environ.get("ZED_MAX_UPLOAD_MB", "200"))
CONFIG_DIRS = ("baselines", "ours")
VERSION = "0.1.0"


def available_configs(root: Path = CONFIG_ROOT) -> dict[str, Path]:
    """``{"baselines/bytetrack": path, ...}``; only these names are accepted (no paths)."""
    out = {}
    for d in CONFIG_DIRS:
        for p in sorted((root / d).glob("*.yaml")):
            out[f"{d}/{p.stem}"] = p
    return out

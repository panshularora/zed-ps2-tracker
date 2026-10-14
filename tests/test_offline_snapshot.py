"""The web app's offline snapshot (web/public/offline) must match the live read-only routes."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from serve.app import create_app

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "web" / "public" / "offline"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    os.environ.setdefault("ZED_DATA_DIR", str(tmp_path_factory.mktemp("data")))
    return TestClient(create_app())


@pytest.mark.parametrize(
    "route", ["/api/benchmarks", "/api/viewer-clips", "/api/results/ablation", "/api/results/gate-events", "/api/results/go-no-go", "/api/configs"]
)
def test_snapshot_matches_route(client, route):
    name = route.removeprefix("/api/").replace("/", "__") + ".json"
    snap = json.loads((SNAP / name).read_text())
    assert client.get(route).json() == snap, f"stale {name}: run python scripts/web/export_offline.py"

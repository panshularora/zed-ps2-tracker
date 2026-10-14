"""Snapshot the read-only GET API into web/public/offline/ so the web app works without the backend.

    python scripts/web/export_offline.py

Every file is produced by the real FastAPI routes (TestClient) over committed repo data only:
docs/results, viewer/data, configs, web/public/demo. The web app's fetch fallback
(web/src/lib/offline.ts) reads these when /api is unreachable. Viewer clips (.json.gz) are copied
at build time by the Vite plugin, not duplicated here.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "public" / "offline"

ENDPOINTS = [
    "/api/health",
    "/api/benchmarks",
    "/api/clips",
    "/api/workstation",
    "/api/viewer-clips",
    "/api/results/ablation",
    "/api/results/gate-events",
    "/api/results/go-no-go",
    "/api/configs",
]


def slug(path: str) -> str:
    return path.removeprefix("/api/").replace("/", "__") + ".json"


def main() -> int:
    tmp = tempfile.mkdtemp()
    os.environ["ZED_DATA_DIR"] = tmp  # no data/: behave like a fresh clone
    os.environ["ZED_OUTPUTS_ROOT"] = str(Path(tmp) / "no-outputs")
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient

    from serve.app import create_app

    client = TestClient(create_app())
    OUT.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def dump(path: str) -> object | None:
        r = client.get(path)
        if r.status_code != 200:
            print(f"skip {path}: {r.status_code}", file=sys.stderr)
            return None
        body = r.json()
        name = slug(path)
        (OUT / name).write_text(json.dumps(body, indent=None, separators=(",", ":")) + "\n")
        written[path] = name
        return body

    for ep in ENDPOINTS:
        dump(ep)
    cfgs = dump("/api/configs")
    names = cfgs if isinstance(cfgs, list) else (cfgs or {}).get("configs", []) if isinstance(cfgs, dict) else []
    for n in names:
        dump(f"/api/config/{n if isinstance(n, str) else n.get('name')}")
    for p, body in [("/api/outputs", {"runs": []}), ("/api/jobs", [])]:
        (OUT / slug(p)).write_text(json.dumps(body) + "\n")
        written[p] = slug(p)
    (OUT / "manifest.json").write_text(json.dumps(written, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(written)} files to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

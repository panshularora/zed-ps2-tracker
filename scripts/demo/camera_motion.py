"""Background (camera) shift per tracked frame of a Scenarios clip, for the gate rule.

Phone clips pan: in ``rabat_stadium`` the background moves ~1,500 px sideways, so raw image motion
says "people walk sideways" when they walk up the walkway. ``scripts/demo/guide.py`` subtracts this
shift before it finds the dominant walking direction. Phase correlation of consecutive frames
(grey, 1/4 scale, Hanning window) on the **unblurred** clip in ``data/demo/raw``; the cumulative
shift (px, full resolution) is stored as ``camera`` in ``web/public/scenarios/<clip>.json``.

    .venv/bin/python -m scripts.demo.camera_motion [--only rabat_stadium ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCALE = 0.25


def camera_shift(path: Path, stride: int, n: int) -> list[list[float]]:
    """Cumulative background shift (dx, dy) in px at each of the first ``n`` tracked frames."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    prev = None
    win = None
    cum = np.zeros(2)
    out: list[list[float]] = []
    i = 0
    while len(out) < n:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            g = cv2.resize(fr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if win is None:
                win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
            if prev is not None:
                (dx, dy), _ = cv2.phaseCorrelate(prev, g, win)
                cum = cum + np.array([dx, dy]) / SCALE
            out.append([round(float(cum[0]), 1), round(float(cum[1]), 1)])
            prev = g
        i += 1
    cap.release()
    return out


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(ROOT))
    from serve.clips import load_manifest

    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=ROOT / "data/demo/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "web/public/scenarios")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    man = {c.id: c for c in load_manifest()}
    for dst in sorted(a.out.glob("*.json")):
        if dst.name == "index.json" or (a.only and dst.stem not in a.only):
            continue
        data = json.loads(dst.read_text())
        m = data["meta"]
        data["camera"] = camera_shift(a.clips / man[m["id"]].file, m["stride"], m["frames_tracked"])
        dst.write_text(json.dumps(data, separators=(",", ":")))
        print(m["id"], len(data["camera"]), data["camera"][-1], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

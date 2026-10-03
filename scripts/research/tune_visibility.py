"""Grid-search the visibility switches of ByteTrackLike on MOT17 **train_half** (public FRCNN).

Each run gets its own output folder (no race on ``OUT/gt``). Selection is by HOTA on train_half
only; the frozen values go to ``configs/ours/visibility.yaml`` and are then run once on val_half.

    python scripts/research/tune_visibility.py --out /tmp/vis_tune --jobs 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = ["visibility.estimator=depth_order", "visibility.score_fusion=product",
        "visibility.occluder_min_score=0.1"]  # fmt: skip
GRID = {
    "visibility.soften_weight": [0.0, 0.25, 0.5, 1.0],
    "visibility.occluded_buffer_scale": [1.0, 2.0, 3.0],
    "visibility.occluded_thresh": [0.5, 0.7],
}
# Stage 2 (after stage 1 showed softening does not help and the best buffer values sat on the
# grid edge): beta = 0, wider buffer/threshold grid. occluded_thresh = 1.01 applies the longer
# buffer to every track; together with the plain ``track_buffer`` controls it tests whether
# visibility matters at all or a longer buffer alone gives the gain.
GRID2 = {
    "visibility.soften_weight": [0.0],
    "visibility.occluded_buffer_scale": [3.0, 4.0, 6.0, 10.0],
    "visibility.occluded_thresh": [0.7, 0.8, 0.9, 1.01],
}
CONTROLS = [("buffer60", ["lifecycle.track_buffer=60"]), ("buffer90", ["lifecycle.track_buffer=90"]),
            ("buffer120", ["lifecycle.track_buffer=120"]),
            ("buffer180", ["lifecycle.track_buffer=180"])]  # fmt: skip


def run(out: Path, name: str, overrides: list[str], split: str) -> dict[str, float]:
    cmd = [sys.executable, "-m", "eval.run", "--dataset", "MOT17", "--root", "data/MOT17",
           "--split", split, "--dets", "public", "--config", "configs/baselines/bytetrack.yaml",
           "--name", name, "--out", str(out / name), "--no-stress"]  # fmt: skip
    for o in overrides:
        cmd += ["-o", o]
    subprocess.run(cmd, check=True, capture_output=True)
    m = json.loads((out / name / "metrics" / name / "metrics.json").read_text())["COMBINED"]
    return {k: float(m[k]) for k in ("HOTA", "IDF1", "MOTA", "IDSW", "AssA", "DetA")}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--split", default="train_half")
    p.add_argument("--stage", type=int, default=1, choices=(1, 2))
    a = p.parse_args()
    grid = GRID if a.stage == 1 else GRID2
    keys = list(grid)
    combos = []
    for vals in itertools.product(*grid.values()):
        d = dict(zip(keys, vals, strict=True))
        if d["visibility.occluded_buffer_scale"] == 1.0 and d["visibility.occluded_thresh"] != 0.5:
            continue  # thresh is irrelevant without the buffer scale
        combos.append(d)
    jobs = [("baseline", [])] + [
        (f"r{i:02d}", BASE + [f"{k}={v}" for k, v in d.items()]) for i, d in enumerate(combos)
    ]
    if a.stage == 2:
        jobs += CONTROLS
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda j: run(a.out, j[0], j[1], a.split), jobs))
    rows = [{"name": n, "overrides": o, **r} for (n, o), r in zip(jobs, res, strict=True)]
    rows.sort(key=lambda r: -r["HOTA"])
    for r in rows:
        print(f"{r['name']:9s} HOTA {r['HOTA']:.2f} IDF1 {r['IDF1']:.2f} MOTA {r['MOTA']:.2f} "
              f"IDSW {r['IDSW']:.0f}  {' '.join(o.split('.', 1)[-1] for o in r['overrides'][-3:])}")  # fmt: skip
    (a.out / f"tune_stage{a.stage}.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

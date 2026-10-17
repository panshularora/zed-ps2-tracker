"""Offline operator demo on an upload job: which possible missed crossings are real?

Ground truth is used here only, offline, never in the UI. Steps:
1. True crossings: the clip's MOT ground truth (pedestrians, visible) on the job's own gate,
   scaled to the ground-truth resolution, hysteresis scaled the same way.
2. Plain crossings: the job's ``crossing`` events (its IN/OUT count).
3. Missed: true crossings not matched one-to-one by a plain crossing (same direction, within
   ``delta`` frames).
4. Real suggestions: ``recovered_crossing`` suggestions matched one-to-one to a missed true
   crossing (same direction, within ``delta`` frames). Only these are the ones to confirm.

    python -m eval.operator_demo --job DIR --gt gt.txt --gt-size 1920 1080
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.gate_events import Event, Gate, crossings, gt_pedestrians, match_events


def _io(ev: list[Event]) -> dict[str, int]:
    return {
        "in": sum(e.direction == "in" for e in ev),
        "out": sum(e.direction == "out" for e in ev),
    }


def _key(e: dict[str, Any]) -> str:
    return f"{e['frame']}:{e['from_track_id']}:{e['track_id']}"


def analyse(
    result: dict[str, Any], gt_rows: np.ndarray, gt_w: int, gt_h: int, delta: int
) -> dict[str, Any]:
    v, g = result["video"], result["gate"]
    sx, sy = gt_w / v["width"], gt_h / v["height"]
    gate = Gate("g", (g["a"][0] * sx, g["a"][1] * sy), (g["b"][0] * sx, g["b"][1] * sy))
    h = float(g["hysteresis_px"]) * sy
    truth = crossings(gt_pedestrians(gt_rows, "MOT17"), gate, h)
    gev = [e for e in result["events"] if e.get("kind") == "gate"]
    plain = [
        Event(int(e["frame"]), int(e["track_id"]), "g", e["direction"])
        for e in gev
        if e["type"] == "crossing"
    ]
    sugs = [e for e in gev if e["type"] == "recovered_crossing"]
    sug_ev = [Event(int(e["frame"]), int(e["track_id"]), "g", e["direction"]) for e in sugs]
    pm = match_events(truth, plain, delta)
    hit = {i for i, _ in pm}
    missed = [e for i, e in enumerate(truth) if i not in hit]
    sm = match_events(missed, sug_ev, delta)
    real = sorted(j for _, j in sm)

    def order(j: int) -> tuple[float, int]:
        sc = sugs[j].get("pair_score")
        return (-(sc if sc is not None else -1.0), int(sugs[j]["frame"]))

    ranked = sorted(range(len(sugs)), key=order)
    rank_of = {j: r + 1 for r, j in enumerate(ranked)}
    return {
        "truth": _io(truth),
        "plain": _io(plain),
        "plain_matched_to_truth": f"{len(pm)} of {len(plain)}",
        "missed_by_plain": [{"frame": e.frame, "direction": e.direction} for e in missed],
        "suggestions": [
            {
                "key": _key(e),
                "frame": e["frame"],
                "from": e["from_track_id"],
                "to": e["track_id"],
                "direction": e["direction"],
                "pair_score": e.get("pair_score"),
                "gap_s": e.get("gap_s"),
                "distance_rel": e.get("distance_rel"),
                "score_rank": rank_of[j],
                "real": j in real,
            }
            for j, e in enumerate(sugs)
        ],
        "real_keys": [_key(sugs[j]) for j in real],
        "real_score_ranks": sorted(rank_of[j] for j in real),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.operator_demo")
    ap.add_argument("--job", type=Path, required=True, help="job directory with result.json")
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--gt-size", type=int, nargs=2, default=(1920, 1080))
    ap.add_argument("--delta", type=int, default=15)
    a = ap.parse_args(argv)
    result = json.loads((a.job / "result.json").read_text())
    gt = np.loadtxt(a.gt, delimiter=",", ndmin=2)
    r = analyse(result, gt, a.gt_size[0], a.gt_size[1], a.delta)
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

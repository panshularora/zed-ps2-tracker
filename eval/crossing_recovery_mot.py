"""Held-out check of crossing recovery on MOT17 val_half (never used to pick its limits).

Frozen gates ``configs/gates/mot17_val_half.yaml`` (h60, v50), MOT17-04 and MOT17-09, tracker
outputs from ``eval.run``. For each gate: true IN/OUT from ground truth, the tracker's plain
IN/OUT, and IN/OUT after adding recovered crossings, plus how many events match a true crossing
(same gate and direction within ``delta_frames``).

    python -m eval.crossing_recovery_mot --root /workspace/zed-e2e/mot17
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.crossing_recovery import RecoveryParams, recover_crossings
from eval.gate_events import Event, _fps, crossings, gt_pedestrians, load_gates, match_events

GATES = Path("configs/gates/mot17_val_half.yaml")
SEQS = ("MOT17-04-FRCNN", "MOT17-09-FRCNN")


def _io(ev: list[Event]) -> dict[str, int]:
    return {
        "in": sum(e.direction == "in" for e in ev),
        "out": sum(e.direction == "out" for e in ev),
    }


def check(root: Path, trackers: list[str], p: RecoveryParams) -> dict[str, Any]:
    gates, params = load_gates(GATES)
    h, delta = float(params["hysteresis_px"]), int(params["delta_frames"])
    out: dict[str, Any] = {"params": p.__dict__, "gates_file": str(GATES), "sequences": {}}
    for seq in SEQS:
        gt = gt_pedestrians(
            np.loadtxt(root / "gt" / seq / "gt" / "gt.txt", delimiter=",", ndmin=2), "MOT17"
        )
        fps = _fps(root / "gt", seq)
        res: dict[str, Any] = {}
        for g in gates[seq]:
            ge = crossings(gt, g, h)
            row: dict[str, Any] = {"truth": _io(ge)}
            for t in trackers:
                tr = np.loadtxt(
                    root / "trackers" / t / "data" / f"{seq}.txt", delimiter=",", ndmin=2
                )
                te = crossings(tr, g, h)
                rec = recover_crossings(tr, g, h, fps, p)
                re = [Event(e.frame, e.track_id, e.gate, e.direction) for e in rec]
                after = sorted(te + re, key=lambda e: e.frame)
                row[t] = {
                    "before": _io(te),
                    "after": _io(after),
                    "recovered": len(rec),
                    "matched_before": len(match_events(ge, te, delta)),
                    "matched_after": len(match_events(ge, after, delta)),
                    "abs_err_before": abs(len(te) - len(ge)),
                    "abs_err_after": abs(len(after) - len(ge)),
                }
            res[g.name] = row
        out["sequences"][seq] = res
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.crossing_recovery_mot")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--trackers", nargs="+", default=["ours_vis", "bytetrack"])
    ap.add_argument("--out", type=Path, default=Path("docs/results/stress/crossing_recovery.json"))
    a = ap.parse_args(argv)
    r = check(a.root, a.trackers, RecoveryParams())
    doc = json.loads(a.out.read_text()) if a.out.exists() else {}
    doc["held_out_mot17"] = r
    a.out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

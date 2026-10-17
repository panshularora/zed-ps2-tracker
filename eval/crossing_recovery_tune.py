"""Pick the crossing-recovery limits on simulator tuning seeds, then report held-out checks.

    python -m eval.crossing_recovery_tune tune     # seeds 0-9 only -> chosen limits
    python -m eval.crossing_recovery_tune report   # seeds 1000-1019 grid + scenarios, with them

Objective on the tuning seeds (both trackers, every cell): sum of |gate count error| after recovery
plus the number of FALSE recoveries (the ended and the new number belong to different people). A
false recovery is charged even when it happens to fix the total, so the limits cannot buy count
accuracy with wrong pairings. Ties go to the smaller (more conservative) limits.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from typing import Any

import numpy as np

from eval import stress_run as sr
from eval.crossing_recovery import RecoveryParams, recover_crossings
from eval.gate_events import Gate
from eval.stress_metrics import GATE_HYST, TrackRows, gate_counts, match_frames
from eval.stress_sim import FPS, GATE_Y, Scenario, W, render

GATE = Gate("gate", (0.0, GATE_Y), (float(W), GATE_Y))
GAPS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
DISTS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
NEARS = (0.5, 1.0, 1.5, 2.0, 3.0)
TUNE_SEEDS = list(range(10))
TUNE_CELLS = [
    Scenario(people=n, hide_s=h) for n in (5, 10, 20, 40) for h in (0.5, 1.0, 2.0, 3.0, 5.0)
]
TUNE_CELLS += [
    replace(sc, label=k)
    for k, sc in sr.SCENARIOS.items()
    if k in ("lookalike_cross", "group", "stop_and_go", "step_back", "linger", "dropout_15f")
]
OUT = sr.OUT / "crossing_recovery.json"


def mot_rows(rows: TrackRows) -> np.ndarray:
    out = [[f, t, *b] for f, tr in enumerate(rows, start=1) for t, b in tr]
    return np.asarray(out, float).reshape(-1, 6)


def _one(args: tuple[Scenario, int, dict[str, str]]) -> dict[str, Any]:
    sc, seed, trackers = args
    frames = render(sc, seed)
    res: dict[str, Any] = {"cell": sc.key(), "seed": seed}
    for name, path in trackers.items():
        rows, _ = sr.run_tracker(sr.config(path), frames)
        m = match_frames(frames, rows)
        votes: dict[int, Counter[int]] = {}
        for mm in m:
            for p, t in mm.items():
                votes.setdefault(t, Counter())[p] += 1
        owner = {t: c.most_common(1)[0][0] for t, c in votes.items()}
        true_n, trk_n = gate_counts(frames, rows)
        res[name] = {"rows": mot_rows(rows), "owner": owner, "true": true_n, "plain": trk_n}
    return res


def evaluate(runs: list[dict[str, Any]], names: list[str], p: RecoveryParams) -> dict[str, Any]:
    tot: dict[str, Any] = {}
    for name in names:
        err_b = err_a = false = rec = 0
        for r in runs:
            x = r[name]
            ev = recover_crossings(x["rows"], GATE, GATE_HYST, FPS, p)
            wrong = sum(
                x["owner"].get(e.track_id, -1) != x["owner"].get(e.from_track_id, -2) for e in ev
            )
            err_b += abs(x["plain"] - x["true"])
            err_a += abs(x["plain"] + len(ev) - x["true"])
            false += wrong
            rec += len(ev)
        tot[name] = {
            "abs_err_before": err_b,
            "abs_err_after": err_a,
            "recovered": rec,
            "false": false,
        }
    return tot


def collect(
    cells: list[Scenario], seeds: list[int], trackers: dict[str, str], workers: int
) -> list[dict[str, Any]]:
    jobs = [(c, s, trackers) for c in cells for s in seeds]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_one, jobs, chunksize=4))


def tune(workers: int) -> dict[str, Any]:
    t0 = time.time()
    runs = collect(TUNE_CELLS, TUNE_SEEDS, sr.TRACKERS, workers)
    names = list(sr.TRACKERS)
    table = []
    for g, d, n in itertools.product(GAPS, DISTS, NEARS):
        p = RecoveryParams(g, d, n)
        e = evaluate(runs, names, p)
        obj = sum(e[k]["abs_err_after"] + e[k]["false"] for k in names)
        table.append({"params": asdict(p), "objective": obj, **e})
    table.sort(
        key=lambda r: (
            r["objective"],
            r["params"]["max_gap_s"],
            r["params"]["max_dist_rel"],
            r["params"]["near_rel"],
        )
    )
    base = evaluate(runs, names, RecoveryParams(0.0, 0.0, 0.0))
    return {
        "seeds": TUNE_SEEDS,
        "cells": [c.key() for c in TUNE_CELLS],
        "objective": "sum over both trackers of |gate error| after recovery + false recoveries",
        "no_recovery": base,
        "chosen": table[0],
        "top10": table[:10],
        "elapsed_s": round(time.time() - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.crossing_recovery_tune")
    ap.add_argument("what", choices=["tune", "report"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    doc = json.loads(OUT.read_text()) if OUT.exists() else {}
    if a.what == "tune":
        doc["tuning"] = tune(a.workers)
    else:
        p = RecoveryParams(**doc["tuning"]["chosen"]["params"])
        seeds = list(range(1000, 1020))
        grid = [Scenario(people=n, hide_s=h) for n in sr.GRID_PEOPLE for h in sr.GRID_HIDE]
        scen = [replace(sc, label=k) for k, sc in sr.SCENARIOS.items()]
        rep: dict[str, Any] = {"seeds": seeds, "params": asdict(p), "cells": {}}
        for group, cells in (("grid", grid), ("scenarios", scen)):
            runs = collect(cells, seeds, sr.TRACKERS, a.workers)
            rep[group] = evaluate(runs, list(sr.TRACKERS), p)
            for c in cells:
                rs = [r for r in runs if r["cell"] == c.key()]
                rep["cells"][c.key()] = evaluate(rs, list(sr.TRACKERS), p)
        doc["held_out_sim"] = rep
    OUT.write_text(json.dumps(doc, indent=1))
    keys = ("grid", "scenarios", "params")
    summary = {
        k: v.get("chosen") if k == "tuning" else {g: v.get(g) for g in keys}
        for k, v in doc.items()
        if k in ("tuning", "held_out_sim")
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

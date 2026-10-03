"""Go/no-go sanity check for the delayed decision: does look-ahead K reduce identity switches?

Conditions (val_half, IoU/motion only, ``configs/ours/delayed.yaml`` with K overridden):

* corrupted ground truth (``eval/corrupt.py``), MOT17 and MOT20, fixed seeds;
* MOT17 public detections FRCNN, SDP and DPM (DPM scores logistic-mapped, see ``eval/run.py``).

For K in {0, 3, 5, 8} it reports TrackEval HOTA / IDF1 / IDSW over all frames, compute latency
(p95 / max of ``update()``, from run.json) and the **IDSW inside flagged windows**. A flagged window
is ±8 frames around a frame where the ambiguity detector fired. Two window sets are used: the K=0
run's (the same frames for every K) and each run's own. Window IDSW uses our own CLEAR-style count
(gt pedestrians, IoU ≥ 0.5 Hungarian per frame, a switch when a gt id's matched tracker id
changes), so its all-frames total is reported next to TrackEval's as a check.

    python -m eval.go_no_go --out /tmp/gng --jobs 7 --report docs/results/go_no_go.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.cost import iou_matrix
from tracker.types import tlwh_to_xyxy

KS = (0, 3, 5, 8)
WINDOW = 8
CONFIG = "configs/ours/delayed.yaml"


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    dataset: str
    root: str
    dets: str
    variant: str = "FRCNN"
    score_map: str = "none"
    seed: int | None = None


def conditions(corrupt_dir: Path, seeds: tuple[int, ...], mot20: bool) -> list[Condition]:
    out = [Condition(f"c17s{s}", f"MOT17 corrupted gt, seed {s}", "MOT17", "data/MOT17",
                     str(corrupt_dir / f"MOT17_s{s}"), seed=s) for s in seeds]  # fmt: skip
    if mot20:
        out.append(Condition("c20s0", "MOT20 corrupted gt, seed 0", "MOT20", "data/MOT20",
                             str(corrupt_dir / "MOT20_s0"), seed=0))  # fmt: skip
    out += [
        Condition("pFRCNN", "MOT17 public FRCNN", "MOT17", "data/MOT17", "public", "FRCNN"),
        Condition("pSDP", "MOT17 public SDP", "MOT17", "data/MOT17", "public", "SDP"),
        Condition("pDPM", "MOT17 public DPM (logistic scores)", "MOT17", "data/MOT17", "public",
                  "DPM", "logistic"),
    ]  # fmt: skip
    return out


def run_one(c: Condition, k: int, out: Path) -> Path:
    name = f"k{k}"
    od = out / c.key / name
    if (od / "metrics" / name / "metrics.json").exists():
        return od
    cmd = [sys.executable, "-m", "eval.run", "--dataset", c.dataset, "--root", c.root,
           "--split", "val_half", "--dets", c.dets, "--config", CONFIG,
           "-o", f"delayed.max_delay_frames={k}", "--name", name, "--out", str(od), "--no-stress",
           "--det-variant", c.variant, "--score-map", c.score_map]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True)
    return od


# ------------------------------------------------------------------------------ window IDSW
def _load(path: Path) -> np.ndarray:
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    return a if a.size else np.zeros((0, 10))


def idsw_frames(gt: np.ndarray, trk: np.ndarray, dataset: str) -> np.ndarray:
    """Frame numbers of every CLEAR-style identity switch (one entry per switch)."""
    ped = gt[:, 7] == 1
    if dataset.upper() == "MOT17":
        ped &= gt[:, 6] == 1
    gt = gt[ped]
    last: dict[int, int] = {}
    sw = []
    for f in np.unique(gt[:, 0]).astype(int):
        g = gt[gt[:, 0] == f]
        t = trk[trk[:, 0] == f]
        if len(t) == 0:
            continue
        iou = iou_matrix(tlwh_to_xyxy(g[:, 2:6]), tlwh_to_xyxy(t[:, 2:6]))
        r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c, strict=True):
            if iou[i, j] < 0.5:
                continue
            gid, tid = int(g[i, 1]), int(t[j, 1])
            if gid in last and last[gid] != tid:
                sw.append(f)
            last[gid] = tid
    return np.asarray(sw, int)


def in_windows(frames: np.ndarray, triggers: list[int]) -> int:
    if len(frames) == 0 or not triggers:
        return 0
    trig = np.asarray(sorted(triggers))
    idx = np.searchsorted(trig, frames)
    lo = np.abs(frames - trig[np.clip(idx - 1, 0, len(trig) - 1)])
    hi = np.abs(frames - trig[np.clip(idx, 0, len(trig) - 1)])
    return int((np.minimum(lo, hi) <= WINDOW).sum())


def analyse(c: Condition, out: Path) -> list[dict[str, Any]]:
    runs = {k: out / c.key / f"k{k}" for k in KS}
    info = {k: json.loads((runs[k] / "trackers" / f"k{k}" / "run.json").read_text()) for k in KS}
    k0_trig = {s: v["tracker_info"]["delayed"]["trigger_frames"]
               for s, v in info[0]["latency"]["per_sequence"].items()}  # fmt: skip
    rows = []
    for k in KS:
        m = json.loads((runs[k] / "metrics" / f"k{k}" / "metrics.json").read_text())["COMBINED"]
        per = info[k]["latency"]["per_sequence"]
        own_n = k0_n = tot = n_trig = changed = 0
        for seq, v in per.items():
            gt = _load(runs[k] / "gt" / seq / "gt" / "gt.txt")
            trk = _load(runs[k] / "trackers" / f"k{k}" / "data" / f"{seq}.txt")
            sw = idsw_frames(gt, trk, c.dataset)
            d = v["tracker_info"]["delayed"]
            tot += len(sw)
            k0_n += in_windows(sw, k0_trig[seq])
            own_n += in_windows(sw, d["trigger_frames"])
            n_trig += len(d["trigger_frames"])
            changed += d["changed"]
        lat = info[k]["latency"]["combined"]
        fps = float(np.mean([v["tracker_info"]["delayed"]["K_ms"] for v in per.values()]))
        rows.append({
            "condition": c.label, "K": k, "K_ms_mean": fps, "HOTA": m["HOTA"], "IDF1": m["IDF1"],
            "MOTA": m["MOTA"], "IDSW_trackeval": m["IDSW"], "IDSW_ours_all": tot,
            "IDSW_in_K0_windows": k0_n, "IDSW_in_own_windows": own_n, "flagged_frames": n_trig,
            "changed_decisions": changed, "p95_ms": lat["p95_ms"], "max_ms": lat["max_ms"],
        })  # fmt: skip
    return rows


def markdown(rows: list[dict[str, Any]]) -> str:
    head = ("| Condition | K (frames / ms) | HOTA | IDF1 | IDSW (TrackEval) | IDSW in K=0 windows | "
            "IDSW in own windows | flagged frames | changed | p95 / max ms |")  # fmt: skip
    lines = [head, "|---" * 10 + "|"]
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['K']} / {r['K_ms_mean']:.0f} | {r['HOTA']:.1f} | "
            f"{r['IDF1']:.1f} | {r['IDSW_trackeval']:.0f} | {r['IDSW_in_K0_windows']} | "
            f"{r['IDSW_in_own_windows']} | {r['flagged_frames']} | {r['changed_decisions']} | "
            f"{r['p95_ms']:.1f} / {r['max_ms']:.0f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.go_no_go", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corrupt-dir", type=Path, default=Path("/tmp/corrupt"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--no-mot20", action="store_true")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    from eval.corrupt import write_corrupted

    conds = conditions(a.corrupt_dir, tuple(a.seeds), not a.no_mot20)
    for c in conds:
        if c.seed is not None and not Path(c.dets).exists():
            write_corrupted(Path(c.root), c.dataset, c.seed, Path(c.dets))
    jobs = [(c, k) for c in conds for k in KS]
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda j: run_one(j[0], j[1], a.out), jobs))
    rows = [r for c in conds for r in analyse(c, a.out)]
    md = markdown(rows)
    print(md)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

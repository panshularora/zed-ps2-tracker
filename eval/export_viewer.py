"""Export tracker runs for the static viewer: one ``<clip>.<tracker>.json.gz`` per clip per tracker.

Format (viewer spec in ``06_phase2_specs.md``; ``viewer/src/data.ts`` validates it)::

    {"clip", "tracker", "fps", "frames", "width", "height", "gate": {"a": [x, y], "b": [x, y]},
     "tracks": {"<id>": [[frame, x, y, w, h, conf], ...]},       # pixels, frames 1-based
     "pending": [{"f0", "f1", "candidates": [ids], "chosen": id}],
     "events": [{"f", "type": "id_switch", "gt", "from", "to"},
                {"f", "type": "cross", "id", "dir": "in"|"out", "correct": bool},
                {"f", "type": "tailgate", "ids": [a, b]}]}

The tracker is re-run here on the split (public detections), so the delayed tracker's held
decisions are available.
- **pending**: one entry per contested detection, i.e. a detection that is a candidate for two or
  more tracks in an ambiguity cluster. ``f0`` is the frame of the decision and ``f1 = f0 + K``. For
  ``candidates`` and ``chosen`` (the track that got the detection, -1 if none), see
  ``tracker.delayed``.
- **id_switch**: CLEAR-style switches against gt (pedestrians, IoU ≥ 0.5, per-frame Hungarian).
- **cross / tailgate**: ``eval.gate_events`` on the clip's first gate. A crossing is ``correct``
  when it matches a gt crossing (same direction, within δ) and its id maps to that gt id under the
  IDF1 bijection.

    python -m eval.export_viewer --seq MOT17-04-FRCNN --gates configs/gates/mot17_val_half.yaml \
        --tracker bytetrack=configs/baselines/bytetrack.yaml --include-gt --out viewer/data
"""

from __future__ import annotations

import argparse
import gzip
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from eval.gate_events import (
    crossings,
    gt_pedestrians,
    id_bijection,
    load_gates,
    match_events,
    tailgate_pairs,
)
from eval.run import load_inputs
from eval.splits import crop_cache, write_gt_split
from tracker.config import load_config
from tracker.cost import iou_matrix
from tracker.runner import run_sequence
from tracker.types import FrameResult, tlwh_to_xyxy


def id_switch_events(gt: np.ndarray, tr: np.ndarray) -> list[dict[str, Any]]:
    last: dict[int, int] = {}
    out = []
    for f in np.unique(gt[:, 0]).astype(int):
        g, t = gt[gt[:, 0] == f], tr[tr[:, 0] == f]
        if len(t) == 0:
            continue
        iou = iou_matrix(tlwh_to_xyxy(g[:, 2:6]), tlwh_to_xyxy(t[:, 2:6]))
        r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c, strict=True):
            if iou[i, j] < 0.5:
                continue
            gid, tid = int(g[i, 1]), int(t[j, 1])
            if gid in last and last[gid] != tid:
                out.append({"f": int(f), "type": "id_switch", "gt": gid, "from": last[gid],
                            "to": tid})  # fmt: skip
            last[gid] = tid
    return out


def pending_entries(results: list[FrameResult]) -> list[dict[str, Any]]:
    out = []
    for r in results:
        by_box: dict[tuple[float, ...], dict[str, Any]] = {}
        for p in r.pending:
            for k, box in enumerate(p.candidate_boxes_tlwh):
                e = by_box.setdefault(tuple(box), {"f0": p.opened_frame, "f1": p.deadline_frame,
                                                   "candidates": set(), "chosen": -1})  # fmt: skip
                e["candidates"].add(p.track_id)
                if p.chosen_index == k:
                    e["chosen"] = p.track_id
        for e in by_box.values():
            if len(e["candidates"]) >= 2:
                out.append({**e, "candidates": sorted(e["candidates"])})
    return out


def results_to_rows(results: list[FrameResult]) -> np.ndarray:
    rows = [[r.frame_id, t.track_id, *t.tlwh, t.score] for r in results for t in r.tracks]
    return np.array(rows, float).reshape(-1, 7)


def build_clip(clip: str, tracker: str, fps: float, frames: int, size: tuple[int, int],
               rows: np.ndarray, gt: np.ndarray, gate: Any, params: dict[str, Any],
               pending: list[dict[str, Any]], is_gt: bool) -> dict[str, Any]:  # fmt: skip
    tracks: dict[str, list[list[float]]] = {}
    for r in rows:
        tracks.setdefault(str(int(r[1])), []).append(
            [int(r[0]), *(round(float(v), 1) for v in r[2:6]), round(float(r[6]), 3)]
        )
    events: list[dict[str, Any]] = [] if is_gt else id_switch_events(gt, rows)
    if gate is not None:
        h, delta = float(params["hysteresis_px"]), int(params["delta_frames"])
        ge, te = crossings(gt, gate, h), crossings(rows, gate, h)
        bij = {int(t): int(t) for t in np.unique(rows[:, 1])} if is_gt else id_bijection(gt, rows)
        ok = {
            j for i, j in match_events(ge, te, delta) if bij.get(te[j].track_id) == ge[i].track_id
        }
        events += [{"f": e.frame, "type": "cross", "id": e.track_id, "dir": e.direction,
                    "correct": j in ok} for j, e in enumerate(te)]  # fmt: skip
        window = round(float(params["tailgate_seconds"]) * fps)
        events += [{"f": e.frame, "type": "tailgate", "ids": [prev, e.track_id]}
                   for e, prev in tailgate_pairs(te, window)]  # fmt: skip
    events.sort(key=lambda e: e["f"])
    return {
        "clip": clip, "tracker": tracker, "fps": fps, "frames": frames,
        "width": size[0], "height": size[1],
        "gate": None if gate is None else {"a": list(gate.a), "b": list(gate.b)},
        "tracks": tracks, "pending": pending, "events": events,
    }  # fmt: skip


def write_clip(data: dict[str, Any], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{data['clip']}.{data['tracker']}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(data, fh, separators=(",", ":"))
    return path


def export(seq: str, trackers: dict[str, str], out: Path, gates_file: Path | None,
           root: Path = Path("data/MOT17"), split: str = "val_half", dataset: str = "MOT17",
           include_gt: bool = True) -> list[Path]:  # fmt: skip
    seq_dir = root / "train" / seq
    cache = crop_cache(load_inputs(seq_dir, dataset, "public"), split)
    with tempfile.TemporaryDirectory() as tmp:
        write_gt_split(seq_dir, Path(tmp), split)
        gt = gt_pedestrians(np.loadtxt(Path(tmp) / seq / "gt" / "gt.txt", delimiter=",",
                                       ndmin=2), dataset)  # fmt: skip
    gate, params = None, {}
    if gates_file is not None:
        gates, params = load_gates(gates_file)
        gate = gates[seq][0] if seq in gates else None
    m = cache.meta
    clip = seq.rsplit("-", 1)[0] if dataset == "MOT17" else seq
    common = (clip, m.frame_rate, m.seq_length, (m.im_width, m.im_height))
    paths = []
    if include_gt:
        rows = np.c_[gt[:, :6], np.ones(len(gt))]
        paths.append(write_clip(build_clip(common[0], "gt", *common[1:], rows, gt, gate, params,
                                           [], True), out))  # fmt: skip
    for name, cfg_path in trackers.items():
        res = run_sequence(cache, load_config(Path(cfg_path)))
        rows = results_to_rows(res.results)
        paths.append(write_clip(build_clip(common[0], name, *common[1:], rows, gt, gate, params,
                                           pending_entries(res.results), False), out))  # fmt: skip
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.export_viewer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--seq", nargs="+", required=True)
    p.add_argument("--tracker", action="append", default=[], help="name=config.yaml")
    p.add_argument("--gates", type=Path, default=None)
    p.add_argument("--split", default="val_half")
    p.add_argument("--include-gt", action="store_true")
    p.add_argument("--out", type=Path, default=Path("viewer/data"))
    a = p.parse_args(argv)
    trackers = dict(t.split("=", 1) for t in a.tracker)
    for s in a.seq:
        for path in export(s, trackers, a.out, a.gates, split=a.split, include_gt=a.include_gt):
            print(path, path.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

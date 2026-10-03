"""Gate events: line crossings, their attribution to identities, and tailgating.

These are **our proposed metrics** for an access-control setting. They build on the line-crossing
counting setup of Ma & Chan, "Crossing the Line: Crowd Counting by Integer Programming with Local
Features" (CVPR 2013), and add identity attribution and tailgating on top.

Definitions (all in image pixels and frames):

* A **gate** is a segment ``a → b``. Its normal ``n`` is ``b − a`` rotated by +90°. A point ``p``
  has signed distance ``s = (p − a) · n̂``, and it counts only while its projection falls inside
  the segment.
* Each track (gt or tracker) is reduced to its **bottom-centre** point per frame, the foot
  position on the ground.
* **Hysteresis** ``h``: a track's side becomes ``+`` when ``s > h`` and ``−`` when ``s < −h``.
  A side change ``− → +`` is an ``in`` crossing and ``+ → −`` an ``out`` crossing, stamped with
  the frame where the new side is first reached. Frames missing from the track keep the last side.
* **Event matching**: gt and tracker events of the same gate and direction are matched one to one
  by the Hungarian method on ``|Δf|``, allowing only pairs with ``|Δf| ≤ δ``. This gives event
  precision, recall and F1, and the **crossing-count error** ``|N_trk − N_gt| / N_gt``.
* **Attribution**: a matched pair is attributed correctly when the tracker id maps to the gt id
  under the IDF1 bijection (a global one-to-one id mapping that maximises the number of frames
  matched at IoU ≥ 0.5, as TrackEval's Identity metric does). Attribution error = 1 − correct /
  matched.
* **Tailgating**: two different ids crossing the same gate in the same direction within ``Δt``.
  Each tailgate is stamped with the second crossing's frame. gt and tracker tailgates are matched
  with the same ``δ`` rule. This gives tailgating precision and recall.

    python -m eval.gate_events --gates configs/gates/mot17_val_half.yaml \
        --gt-dir OUT/gt --trackers-dir OUT/trackers --names bytetrack botsort
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from tracker.cost import iou_matrix
from tracker.types import tlwh_to_xyxy


@dataclass(frozen=True)
class Gate:
    name: str
    a: tuple[float, float]
    b: tuple[float, float]


@dataclass(frozen=True)
class Event:
    frame: int
    track_id: int
    gate: str
    direction: str  # "in" | "out"


def load_gates(path: Path) -> tuple[dict[str, list[Gate]], dict[str, Any]]:
    """``{sequence: [Gate, ...]}`` and the protocol parameters (h, delta, tailgate seconds)."""
    raw = yaml.safe_load(path.read_text())
    gates = {seq: [Gate(g["name"], tuple(g["a"]), tuple(g["b"])) for g in gs]  # type: ignore[misc]
             for seq, gs in raw["sequences"].items()}  # fmt: skip
    return gates, raw["params"]


def bottom_centre(tlwh: np.ndarray) -> np.ndarray:
    return np.c_[tlwh[:, 0] + tlwh[:, 2] / 2, tlwh[:, 1] + tlwh[:, 3]]


def crossings(rows: np.ndarray, gate: Gate, h: float) -> list[Event]:
    """Crossing events of MOT rows ``frame,id,x,y,w,h,...`` for one gate."""
    a, b = np.asarray(gate.a, float), np.asarray(gate.b, float)
    d = b - a
    length = float(np.hypot(*d))
    if length == 0:
        raise ValueError(f"gate {gate.name} has zero length")
    u = d / length
    nrm = np.array([-u[1], u[0]])
    out: list[Event] = []
    for tid in np.unique(rows[:, 1]).astype(int):
        r = rows[rows[:, 1] == tid]
        r = r[np.argsort(r[:, 0], kind="stable")]
        p = bottom_centre(r[:, 2:6]) - a
        along = p @ u
        s = p @ nrm
        side = 0
        for f, al, si in zip(r[:, 0].astype(int), along, s, strict=True):
            if al < 0 or al > length:
                continue
            new = 1 if si > h else (-1 if si < -h else side)
            if side != 0 and new != side:
                out.append(Event(int(f), int(tid), gate.name, "in" if new > 0 else "out"))
            side = new
    return sorted(out, key=lambda e: (e.frame, e.track_id))


def match_events(gt: list[Event], tr: list[Event], delta: int) -> list[tuple[int, int]]:
    """One-to-one matches (indices) of same-gate, same-direction events within ``delta`` frames."""
    if not gt or not tr:
        return []
    cost = np.full((len(gt), len(tr)), 1e9)
    for i, g in enumerate(gt):
        for j, t in enumerate(tr):
            if g.gate == t.gate and g.direction == t.direction and abs(g.frame - t.frame) <= delta:
                cost[i, j] = abs(g.frame - t.frame)
    r, c = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(r, c, strict=True) if cost[i, j] < 1e9]


def tailgates(events: list[Event], window: int) -> list[Event]:
    """Second crossing of each pair of different ids, same gate and direction, within window."""
    out = []
    for i, e in enumerate(events):
        for prev in events[:i]:
            if (prev.gate == e.gate and prev.direction == e.direction
                    and prev.track_id != e.track_id and 0 <= e.frame - prev.frame <= window):  # fmt: skip
                out.append(e)
                break
    return out


def id_bijection(gt: np.ndarray, tr: np.ndarray) -> dict[int, int]:
    """Tracker id → gt id under the IDF1 global assignment (IoU ≥ 0.5 co-occurrence counts)."""
    counts: dict[tuple[int, int], int] = {}
    for f in np.unique(gt[:, 0]):
        g = gt[gt[:, 0] == f]
        t = tr[tr[:, 0] == f]
        if len(t) == 0:
            continue
        iou = iou_matrix(tlwh_to_xyxy(g[:, 2:6]), tlwh_to_xyxy(t[:, 2:6]))
        for i, j in zip(*np.nonzero(iou >= 0.5), strict=True):
            k = (int(g[i, 1]), int(t[j, 1]))
            counts[k] = counts.get(k, 0) + 1
    if not counts:
        return {}
    gids = sorted({k[0] for k in counts})
    tids = sorted({k[1] for k in counts})
    gi = {g: i for i, g in enumerate(gids)}
    ti = {t: i for i, t in enumerate(tids)}
    m = np.zeros((len(gids), len(tids)))
    for (g, t), n in counts.items():
        m[gi[g], ti[t]] = n
    r, c = linear_sum_assignment(-m)
    return {tids[j]: gids[i] for i, j in zip(r, c, strict=True) if m[i, j] > 0}


def gt_pedestrians(gt: np.ndarray, benchmark: str) -> np.ndarray:
    keep = gt[:, 7] == 1
    if benchmark.upper() == "MOT17":
        keep &= gt[:, 6] == 1
    return gt[keep]


def evaluate_sequence(gt: np.ndarray, tr: np.ndarray, gates: list[Gate], params: dict[str, Any],
                      fps: float) -> dict[str, Any]:  # fmt: skip
    h, delta = float(params["hysteresis_px"]), int(params["delta_frames"])
    window = int(round(float(params["tailgate_seconds"]) * fps))
    ge = sorted((e for g in gates for e in crossings(gt, g, h)), key=lambda e: e.frame)
    te = sorted((e for g in gates for e in crossings(tr, g, h)), key=lambda e: e.frame)
    m = match_events(ge, te, delta)
    bij = id_bijection(gt, tr)
    correct = sum(bij.get(te[j].track_id) == ge[i].track_id for i, j in m)
    gtg, ttg = tailgates(ge, window), tailgates(te, window)
    mt = match_events(gtg, ttg, delta)
    return {"gt_events": ge, "trk_events": te, "matches": m, "correct": correct,
            "gt_tailgates": gtg, "trk_tailgates": ttg, "tailgate_matches": mt,
            "bijection": bij}  # fmt: skip


def summarise(per_seq: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n_gt = sum(len(v["gt_events"]) for v in per_seq.values())
    n_tr = sum(len(v["trk_events"]) for v in per_seq.values())
    n_m = sum(len(v["matches"]) for v in per_seq.values())
    n_c = sum(v["correct"] for v in per_seq.values())
    tg_gt = sum(len(v["gt_tailgates"]) for v in per_seq.values())
    tg_tr = sum(len(v["trk_tailgates"]) for v in per_seq.values())
    tg_m = sum(len(v["tailgate_matches"]) for v in per_seq.values())
    abs_err = sum(abs(len(v["trk_events"]) - len(v["gt_events"])) for v in per_seq.values())

    def div(a: float, b: float) -> float | None:
        return a / b if b else None

    p, r = div(n_m, n_tr), div(n_m, n_gt)
    return {
        "gt_crossings": n_gt, "trk_crossings": n_tr, "matched": n_m,
        "count_error": div(abs(n_tr - n_gt), n_gt),
        "count_error_per_seq_abs_sum": div(abs_err, n_gt),
        "precision": p, "recall": r,
        "f1": div(2 * p * r, p + r) if p is not None and r is not None and p + r > 0 else None,
        "attribution_error": None if n_m == 0 else 1 - n_c / n_m,
        "gt_tailgates": tg_gt, "trk_tailgates": tg_tr, "tailgate_matched": tg_m,
        "tailgate_precision": div(tg_m, tg_tr), "tailgate_recall": div(tg_m, tg_gt),
    }  # fmt: skip


def run(gates_file: Path, gt_dir: Path, trk_file_for: Any, benchmark: str = "MOT17",
        fps_for: Any = None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:  # fmt: skip
    gates, params = load_gates(gates_file)
    per: dict[str, dict[str, Any]] = {}
    for seq, gs in gates.items():
        gt = gt_pedestrians(np.loadtxt(gt_dir / seq / "gt" / "gt.txt", delimiter=",", ndmin=2),
                            benchmark)  # fmt: skip
        tf = Path(trk_file_for(seq))
        tr = np.loadtxt(tf, delimiter=",", ndmin=2) if tf.stat().st_size else np.zeros((0, 10))
        if tf.name == "gt.txt":  # gt submitted as a tracker: same pedestrian filter
            tr = gt_pedestrians(tr, benchmark)
        fps = fps_for(seq) if fps_for else 30.0
        per[seq] = evaluate_sequence(gt, tr, gs, params, fps)
    return summarise(per), per


def _fps(gt_dir: Path, seq: str) -> float:
    import configparser

    cp = configparser.ConfigParser()
    cp.read(gt_dir / seq / "seqinfo.ini")
    return float(cp["Sequence"]["frameRate"])


def _fmt(x: Any, pct: bool = False) -> str:
    if x is None:
        return "–"
    return f"{100 * x:.1f}" if pct else f"{x}"


def markdown(rows: dict[str, dict[str, Any]]) -> str:
    lines = ["| Tracker | gt / trk crossings | count error % | event P / R % | "
             "attribution error % | tailgates gt / trk | tailgate P / R % |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, s in rows.items():
        lines.append(
            f"| {name} | {s['gt_crossings']} / {s['trk_crossings']} | "
            f"{_fmt(s['count_error'], True)} | {_fmt(s['precision'], True)} / "
            f"{_fmt(s['recall'], True)} | {_fmt(s['attribution_error'], True)} | "
            f"{s['gt_tailgates']} / {s['trk_tailgates']} | "
            f"{_fmt(s['tailgate_precision'], True)} / {_fmt(s['tailgate_recall'], True)} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.gate_events", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--gates", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True, help="eval.run OUT/gt")
    p.add_argument("--runs", type=Path, nargs="+", required=True,
                   help="eval.run OUT/trackers/<name> folders")  # fmt: skip
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--include-gt", action="store_true", help="add gt-as-tracker (sanity: 0 error)")
    p.add_argument("--benchmark", default="MOT17")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    labels = a.labels or [r.name for r in a.runs]
    rows: dict[str, dict[str, Any]] = {}
    runs = list(zip(labels, a.runs, strict=True))
    if a.include_gt:
        rows["gt as tracker (check)"], _ = run(
            a.gates, a.gt_dir, lambda s: a.gt_dir / s / "gt" / "gt.txt", a.benchmark,
            lambda s: _fps(a.gt_dir, s))  # fmt: skip
    for label, rdir in runs:
        rows[label], _ = run(a.gates, a.gt_dir, lambda s, d=rdir: d / "data" / f"{s}.txt",
                             a.benchmark, lambda s: _fps(a.gt_dir, s))  # fmt: skip
    md = markdown(rows)
    print(md)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "gate_events.json").write_text(json.dumps(rows, indent=1))
        (a.out / "gate_events.md").write_text(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

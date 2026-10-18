"""Guided-replay annotations for a Scenarios clip, computed from the tracked boxes alone.

Adds a ``guide`` block to ``web/public/scenarios/<clip>.json``:

* ``gate``: the counting line, from where people walk (``motion_gate``: perpendicular to the
  dominant walking direction of the tracks, both trackers pooled); no tracker's count is used.
* ``events[run]``: gate crossings (``eval.gate_events.crossings``, 10 px hysteresis; none in the
  first ``SETTLE_S`` of the clip, see below), ``lost`` /
  ``back`` (same ID returns after a gap) and ``renumber`` (one tracker changed the number while the
  other kept it: the precomputed disagreements).
* ``moments``: EVERY moment where at least one tracker changed a person's number (``RULE``), with
  per-run IDs before/after, which tracker(s) changed, and whether the changed number would be
  counted twice at the gate.

No ground truth: everything here describes what each tracker did, never who was right.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from eval.gate_events import Gate, crossings

HYST_PX = 10.0
# No crossing is counted in the first second of a clip: people already in view on frame 1 were
# mid-stride when the video started, so the tracker has not seen which side they came from (on the
# turnstile clip a box shrinking at the frame edge "crossed" on frame 4). Same rule for both runs.
SETTLE_S = 1.0
GAP_S = 0.5  # same-ID gap that counts as "lost then back"

# Gate rule (same for every clip, documented in docs/demo_gates.md): the line is placed from
# where people walk, never from any tracker's count. Per track (both trackers' tracks pooled with
# equal weight, so neither is favoured), take the bottom-centre (foot point) displacement from its
# first to its last frame; tracks shorter than MIN_TRACK_S or moving less than MIN_MOVE_FRAC of the
# frame diagonal are ignored (people standing still). The dominant walking axis is the principal
# axis of those displacements (sum of d d^T, so opposite flows on one axis add up). The gate is the
# line perpendicular to that axis, through the median of the tracks' mid-points along the axis,
# spanning the whole frame. "in" = the direction most of those tracks walk along the axis. A
# panning camera is removed first (background shift per frame, scripts/demo/camera_motion.py).
MIN_TRACK_S = 1.0
MIN_MOVE_FRAC = 0.05


def rows_of(frames: dict[str, list[list[float]]]) -> np.ndarray:
    out = [[int(f), *b] for f, boxes in frames.items() for b in boxes]
    return np.asarray(sorted(out), float).reshape(-1, 6)


def _foot(box: np.ndarray) -> np.ndarray:
    return np.array([box[0] + box[2] / 2.0, box[1] + box[3]])


def walking_motion(
    rows: dict[str, np.ndarray],
    fps_t: float,
    w: int,
    h: int,
    camera: list[list[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(displacements, mid-points) of the foot point of every moving track of every run.

    ``camera[f - 1]`` is the cumulative image shift of the background at tracked frame ``f``
    (``scripts/demo/camera_motion.py``); it is subtracted so a panning phone camera does not look
    like people walking sideways. Mid-points stay in image coordinates (where the line is drawn)."""
    cam = np.asarray(camera, float).reshape(-1, 2) if camera else None

    def shift(f: float) -> np.ndarray:
        if cam is None or not len(cam):
            return np.zeros(2)
        return cam[min(max(int(f) - 1, 0), len(cam) - 1)]

    disp, mid = [], []
    min_f = max(2, round(MIN_TRACK_S * fps_t))
    min_d = MIN_MOVE_FRAC * float(np.hypot(w, h))
    for r in rows.values():
        if not len(r):
            continue
        for tid in np.unique(r[:, 1]):
            t = r[r[:, 1] == tid]
            t = t[np.argsort(t[:, 0], kind="stable")]
            if len(t) < min_f:
                continue
            p0, p1 = _foot(t[0, 2:6]), _foot(t[-1, 2:6])
            d = (p1 - shift(t[-1, 0])) - (p0 - shift(t[0, 0]))
            if float(np.hypot(*d)) < min_d:
                continue
            disp.append(d)
            mid.append((p0 + p1) / 2.0)
    return np.asarray(disp, float).reshape(-1, 2), np.asarray(mid, float).reshape(-1, 2)


def _clip_line(c: float, n: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """End points on the frame border of the line {p : n . p = c}."""
    pts: list[np.ndarray] = []
    if abs(n[1]) > 1e-9:
        for x in (0.0, float(w)):
            y = (c - n[0] * x) / n[1]
            if -1e-6 <= y <= h + 1e-6:
                pts.append(np.array([x, min(max(y, 0.0), h)]))
    if abs(n[0]) > 1e-9:
        for y in (0.0, float(h)):
            x = (c - n[1] * y) / n[0]
            if -1e-6 <= x <= w + 1e-6:
                pts.append(np.array([min(max(x, 0.0), w), y]))
    uniq: list[np.ndarray] = []
    for q in pts:
        if all(np.hypot(*(q - u)) > 1e-3 for u in uniq):
            uniq.append(q)
    if len(uniq) < 2:
        raise ValueError("gate line does not cross the frame")
    best = max(((p, q) for i, p in enumerate(uniq) for q in uniq[i + 1 :]),
               key=lambda pq: float(np.hypot(*(pq[0] - pq[1]))))  # fmt: skip
    return best


def motion_gate(
    rows: dict[str, np.ndarray],
    fps_t: float,
    w: int,
    h: int,
    camera: list[list[float]] | None = None,
) -> dict[str, Any]:
    """The documented gate rule (see the comment above ``MIN_TRACK_S``). Uses track motion only
    (camera pan removed): no crossing count of any tracker is looked at."""
    disp, mid = walking_motion(rows, fps_t, w, h, camera)
    if len(disp) == 0:
        axis = np.array([0.0, 1.0])  # nobody moves: fall back to a horizontal line at mid-height
        c_pt = np.array([w / 2.0, h / 2.0])
    else:
        vals, vecs = np.linalg.eigh(disp.T @ disp)
        axis = vecs[:, int(np.argmax(vals))]
        proj = disp @ axis
        if int(np.sum(proj > 0)) < int(np.sum(proj < 0)):
            axis = -axis
        c_pt = mid[int(np.argsort(mid @ axis)[len(mid) // 2])]
    c = float(axis @ c_pt)
    p, q = _clip_line(c, axis, w, h)
    u_want = np.array([axis[1], -axis[0]])  # left-hand normal of a->b == axis  ("in")
    if float((q - p) @ u_want) < 0:
        p, q = q, p
    ang = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return {
        "a": [round(float(p[0])), round(float(p[1]))],
        "b": [round(float(q[0])), round(float(q[1]))],
        "norm": [
            round(float(p[0]) / w, 3),
            round(float(p[1]) / h, 3),
            round(float(q[0]) / w, 3),
            round(float(q[1]) / h, 3),
        ],
        "walk_deg": round(ang, 1),
        "tracks_used": len(disp),
    }


def gaps(r: np.ndarray, fps_tracked: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tid in np.unique(r[:, 1]).astype(int):
        fr = np.sort(r[r[:, 1] == tid, 0].astype(int))
        for a, b in pairwise(fr):
            gap = (b - a) / fps_tracked
            if gap >= GAP_S:
                out.append({"t": "lost", "f": int(a), "id": int(tid)})
                out.append({"t": "back", "f": int(b), "id": int(tid), "gap_s": round(gap, 2)})
    return out


def _box(r: np.ndarray, f: int, tid: int) -> np.ndarray | None:
    m = r[(r[:, 0] == f) & (r[:, 1] == tid)]
    return m[0, 2:6] if len(m) else None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


def _match(r: np.ndarray, f: int, box: np.ndarray, min_iou: float = 0.3) -> int | None:
    best, bid = min_iou, None
    for row in r[r[:, 0] == f]:
        v = _iou(row[2:6], box)
        if v > best:
            best, bid = v, int(row[1])
    return bid


def annotate(data: dict[str, Any]) -> dict[str, Any]:
    meta = data["meta"]
    w, h = meta["width"], meta["height"]
    fps_t = meta["src_fps"] / meta["stride"]
    rows = {k: rows_of(v["frames"]) for k, v in data["runs"].items()}
    mg = motion_gate(rows, fps_t, w, h, data.get("camera"))
    gate = Gate("walk", (mg["a"][0], mg["a"][1]), (mg["b"][0], mg["b"][1]))
    settle = 1 + SETTLE_S * fps_t
    cross = {
        k: [e for e in crossings(r, gate, HYST_PX) if e.frame >= settle] if len(r) else []
        for k, r in rows.items()
    }
    events: dict[str, list[dict[str, Any]]] = {}
    for k, r in rows.items():
        ev = [{"t": "cross", "f": e.frame, "id": e.track_id, "dir": e.direction} for e in cross[k]]
        ev += gaps(r, fps_t) if len(r) else []
        events[k] = ev
    ren = {k: renumbers(r, fps_t) if len(r) else [] for k, r in rows.items()}
    for k, evs in ren.items():
        events[k] += [{"t": "renumber", "f": e["f"], "from": e["from"], "to": e["to"]} for e in evs]
    for ev in events.values():
        ev.sort(key=lambda e: (e["f"], e["t"], e.get("id", e.get("to", 0))))

    def crossed(k: str, tid: int) -> set[str]:
        return {e.direction for e in cross[k] if e.track_id == tid}

    moments = merge_moments(ren, rows, fps_t)
    for m in moments:
        m["counted_twice"] = {
            k: bool(m["changed"].get(k))
            and bool(
                crossed(k, m["people"][0]["ids"][k]["before"])
                & crossed(k, m["people"][0]["ids"][k]["after"])
            )
            for k in rows
        }
    return {
        "gate": {
            "name": gate.name,
            "a": list(gate.a),
            "b": list(gate.b),
            "how": "perpendicular to the dominant walking direction of the tracks (both trackers"
            " pooled), through the median track mid-point; no tracker count is used",
            "norm": mg["norm"],
            "walk_deg": mg["walk_deg"],
            "tracks_used": mg["tracks_used"],
        },
        "rule": RULE,
        "events": events,
        "moments": moments,
        "kind": meta["kind"],
    }


RULE = (
    "every moment where at least one tracker changed a person's number: a number disappears and a"
    " new number appears in the same place (box overlap >= 0.3) within 1 s"
)


def renumbers(r: np.ndarray, fps_t: float, window_s: float = 1.0) -> list[dict[str, Any]]:
    """One tracker's number changes: id A ends, a new id B starts at the same spot within 1 s."""
    first: dict[int, tuple[int, np.ndarray]] = {}
    last: dict[int, tuple[int, np.ndarray]] = {}
    for row in r[np.argsort(r[:, 0], kind="stable")]:
        tid, f = int(row[1]), int(row[0])
        first.setdefault(tid, (f, row[2:6]))
        last[tid] = (f, row[2:6])
    win = max(1, round(window_s * fps_t))
    used: set[int] = set()
    out: list[dict[str, Any]] = []
    for b, (fb, bb) in sorted(first.items(), key=lambda x: x[1][0]):
        best, pick = 0.3, None
        for a, (fa, ba) in last.items():
            if a == b or a in used or not (fb - win <= fa < fb) or first[a][0] >= fb:
                continue
            v = _iou(ba, bb)
            if v >= best:
                best, pick = v, a
        if pick is not None:
            used.add(pick)
            out.append({"f": fb, "from": pick, "to": b, "box": bb, "last_f": last[pick][0],
                        "last_box": last[pick][1]})  # fmt: skip
    return out


def merge_moments(
    ren: dict[str, list[dict[str, Any]]], rows: dict[str, np.ndarray], fps_t: float
) -> list[dict[str, Any]]:
    """One moment per number change; a change on both trackers at the same spot within 1 s is one
    "both" moment. For the other tracker we record what number it had there before and after."""
    win = max(1, round(fps_t))
    taken: set[int] = set()
    out: list[dict[str, Any]] = []
    for e in ren["baseline"]:
        twin = None
        for j, o in enumerate(ren["ours"]):
            if j not in taken and abs(o["f"] - e["f"]) <= win and _iou(o["box"], e["box"]) >= 0.3:
                twin = j
                break
        if twin is not None:
            taken.add(twin)
        out.append(_moment(e, "baseline", ren["ours"][twin] if twin is not None else None, rows))
    for j, o in enumerate(ren["ours"]):
        if j not in taken:
            out.append(_moment(o, "ours", None, rows))
    return sorted(out, key=lambda m: m["f"])


def _moment(
    e: dict[str, Any], run: str, twin: dict[str, Any] | None, rows: dict[str, np.ndarray]
) -> dict[str, Any]:
    other = "ours" if run == "baseline" else "baseline"
    ids: dict[str, dict[str, int | None]] = {run: {"before": e["from"], "after": e["to"]}}
    if twin is not None:
        ids[other] = {"before": twin["from"], "after": twin["to"]}
    else:
        ids[other] = {
            "before": _match(rows[other], e["last_f"], e["last_box"]),
            "after": _match(rows[other], e["f"], e["box"]),
        }
    f = min(e["f"], twin["f"]) if twin else e["f"]
    box = [round(float(v)) for v in e["box"]]
    return {
        "f": int(f),
        "kind": "occlusion" if e["f"] - e["last_f"] > 1 else "overlap",
        "changed": {run: True, other: twin is not None},
        "people": [{"box": box, "ids": ids}],
        "focus": box,
    }

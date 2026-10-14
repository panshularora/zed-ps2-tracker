"""Guided-replay annotations for a Scenarios clip, computed from the tracked boxes alone.

Adds a ``guide`` block to ``web/public/scenarios/<clip>.json``:

* ``gate``: the counting line. Candidates are fixed fractions of the frame (image geometry only);
  the one people walk across most often (summed over BOTH trackers, so neither is favoured) wins.
* ``events[run]``: gate crossings (``eval.gate_events.crossings``, 10 px hysteresis), ``lost`` /
  ``back`` (same ID returns after a gap) and ``renumber`` (one tracker changed the number while the
  other kept it: the precomputed disagreements).
* ``moments``: up to 3 disagreement moments to pause on, with the people involved (per-run IDs
  before/after) and whether a changed number would be counted twice at the gate.

No ground truth: everything here describes what each tracker did, never who was right.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from eval.gate_events import Gate, crossings

HYST_PX = 10.0
GAP_S = 0.5  # same-ID gap that counts as "lost then back"
MAX_MOMENTS = 3
MIN_SPACING_S = 2.5

# (name, a, b) as fractions of width / height
CANDIDATES: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
    ("h50", (0.0, 0.5), (1.0, 0.5)),
    ("h65", (0.0, 0.65), (1.0, 0.65)),
    ("h80", (0.0, 0.8), (1.0, 0.8)),
    ("v35", (0.35, 0.0), (0.35, 1.0)),
    ("v50", (0.5, 0.0), (0.5, 1.0)),
    ("v65", (0.65, 0.0), (0.65, 1.0)),
]


def rows_of(frames: dict[str, list[list[float]]]) -> np.ndarray:
    out = [[int(f), *b] for f, boxes in frames.items() for b in boxes]
    return np.asarray(sorted(out), float).reshape(-1, 6)


def _gate(name: str, a: tuple[float, float], b: tuple[float, float], w: int, h: int) -> Gate:
    return Gate(name, (round(a[0] * w), round(a[1] * h)), (round(b[0] * w), round(b[1] * h)))


def pick_gate(rows: dict[str, np.ndarray], w: int, h: int) -> Gate:
    best, best_n = None, -1
    for name, a, b in CANDIDATES:
        g = _gate(name, a, b, w, h)
        n = sum(len(crossings(r, g, HYST_PX)) for r in rows.values() if len(r))
        if n > best_n:
            best, best_n = g, n
    assert best is not None
    return best


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


def _near(r: np.ndarray, f: int, box: np.ndarray, skip: int) -> np.ndarray | None:
    """The other box at f overlapping `box` most (the person they were mixed up with)."""
    best, out = 0.02, None
    for row in r[r[:, 0] == f]:
        if int(row[1]) == skip:
            continue
        v = _iou(row[2:6], box)
        if best < v < 0.8:  # >= 0.8 is the same person's own box, not someone else
            best, out = v, row[2:6]
    return out


def annotate(data: dict[str, Any]) -> dict[str, Any]:
    meta = data["meta"]
    w, h = meta["width"], meta["height"]
    fps_t = meta["src_fps"] / meta["stride"]
    rows = {k: rows_of(v["frames"]) for k, v in data["runs"].items()}
    gate = pick_gate(rows, w, h)
    cross = {k: crossings(r, gate, HYST_PX) if len(r) else [] for k, r in rows.items()}
    events: dict[str, list[dict[str, Any]]] = {}
    for k, r in rows.items():
        ev = [{"t": "cross", "f": e.frame, "id": e.track_id, "dir": e.direction} for e in cross[k]]
        ev += gaps(r, fps_t) if len(r) else []
        events[k] = ev
    for d in data["disagreements"]:
        if d["kept"] == "ours":
            events["baseline"].append(
                {
                    "t": "renumber",
                    "f": d["frame"],
                    "from": d["baseline_from"],
                    "to": d["baseline_to"],
                }
            )
        else:
            events["ours"].append(
                {"t": "renumber", "f": d["frame"], "from": d["ours_from"], "to": d["ours_to"]}
            )
    for ev in events.values():
        ev.sort(key=lambda e: (e["f"], e["t"], e.get("id", e.get("to", 0))))

    def crossed(k: str, tid: int) -> list[str]:
        return [e.direction for e in cross[k] if e.track_id == tid]

    # candidate moments: disagreements, merged within 1 s, scored by overlap and gate proximity
    cands: list[tuple[float, dict[str, Any]]] = []
    last_f = -(10**9)
    for d in data["disagreements"]:
        f = d["frame"]
        if (f - last_f) / fps_t < 1.0:
            continue
        last_f = f
        box = np.asarray(d["box"], float)
        changed = "baseline" if d["kept"] == "ours" else "ours"
        kept = "ours" if changed == "baseline" else "baseline"
        frm, to = (
            (d["baseline_from"], d["baseline_to"])
            if changed == "baseline"
            else (d["ours_from"], d["ours_to"])
        )
        kept_id = d["ours_id"] if kept == "ours" else d["baseline_id"]
        people = [
            {
                "box": [round(float(v)) for v in box],
                "ids": {
                    changed: {"before": frm, "after": to},
                    kept: {"before": kept_id, "after": kept_id},
                },
            }
        ]
        other = _near(rows["ours"], f, box, skip=kept_id if kept == "ours" else -1)
        kind = "overlap" if other is not None else "occlusion"
        if other is not None:
            ids: dict[str, dict[str, int | None]] = {}
            for k in rows:
                ids[k] = {
                    "before": _match(rows[k], max(1, f - 1), other),
                    "after": _match(rows[k], f, other),
                }
            people.append({"box": [round(float(v)) for v in other], "ids": ids})
        # counted twice: old and new number both cross the gate in the same direction
        twice = bool(set(crossed(changed, frm)) & set(crossed(changed, to)))
        near_gate = any(
            abs(e.frame - f) / fps_t < 2.0 and e.track_id in (frm, to) for e in cross[changed]
        )
        score = (
            (2.0 if twice else 0.0)
            + (1.0 if near_gate else 0.0)
            + (0.5 if other is not None else 0.0)
        )
        xs = [p["box"][0] for p in people] + [p["box"][0] + p["box"][2] for p in people]
        ys = [p["box"][1] for p in people] + [p["box"][1] + p["box"][3] for p in people]
        cands.append(
            (
                score,
                {
                    "f": f,
                    "kind": kind,
                    "changed": changed,
                    "kept": kept,
                    "people": people,
                    "focus": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                    "counted_twice": twice,
                    "near_gate": near_gate,
                },
            )
        )
    chosen: list[dict[str, Any]] = []
    for _, m in sorted(cands, key=lambda c: (-c[0], c[1]["f"])):
        if len(chosen) >= MAX_MOMENTS:
            break
        if all(abs(m["f"] - c["f"]) / fps_t >= MIN_SPACING_S for c in chosen):
            chosen.append(m)
    chosen.sort(key=lambda m: m["f"])
    return {
        "gate": {
            "name": gate.name,
            "a": list(gate.a),
            "b": list(gate.b),
            "how": "fixed-fraction candidate line most often crossed (both trackers summed)",
        },
        "events": events,
        "moments": chosen,
        "kind": meta["kind"],
    }

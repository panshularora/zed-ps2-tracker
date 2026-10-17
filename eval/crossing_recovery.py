"""Crossing recovery: count a crossing that no single track number made.

When the tracker loses a person while they are hidden *at* the gate line, the old number ends on
one side and a new number begins on the other. Neither number crosses, so plain line counting
(:func:`eval.gate_events.crossings`) misses the crossing. This step, run after the plain count and
separate from it, adds **one** crossing for such a pair:

* track ``a`` **ends** with its last foot point within ``near_rel`` box heights of the line (and
  inside the segment), on side ``A`` (hysteresis side at its end, ``±h``);
* track ``b`` **starts** after ``a`` ended, at most ``max_gap_s`` later, with its first foot point
  within ``near_rel`` box heights of the line, on side ``B = −A`` (hysteresis side at its start);
* the two points are at most ``max_dist_rel`` box heights apart;
* one-to-one: each ending and each start is used at most once (closest in time, then space).

The direction comes from ``A → B`` with the same convention as ``crossings`` (``− → +`` = in). The
event is stamped at ``b``'s first frame. Track numbers are never changed: this is a counting rule,
the IDs on screen stay what the tracker drew. It cannot double count a crossing that either number
made by itself, because ``a`` ends on ``A`` and ``b`` begins on ``B``: the side change happened
between the two, where neither number existed.

The limits (``max_gap_s``, ``max_dist_rel``, ``near_rel``) were picked on simulator tuning seeds
only (``eval.crossing_recovery_tune``, seeds 0-9); MOT17 is a held-out check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval.gate_events import Gate, bottom_centre


@dataclass(frozen=True)
class RecoveryParams:
    # chosen by eval.crossing_recovery_tune on simulator seeds 0-9 only (see
    # docs/results/stress/CROSSING_RECOVERY.md); never fitted on MOT17
    max_gap_s: float = 4.0
    max_dist_rel: float = 2.0
    near_rel: float = 3.0


@dataclass(frozen=True)
class RecoveredCrossing:
    frame: int
    track_id: int  # the new number
    from_track_id: int  # the number that ended
    gate: str
    direction: str  # "in" | "out"
    gap_frames: int
    distance_rel: float


@dataclass(frozen=True)
class _End:
    tid: int
    frame: int
    p: np.ndarray
    h: float
    s: float
    side: int


def _geometry(gate: Gate) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    a, b = np.asarray(gate.a, float), np.asarray(gate.b, float)
    d = b - a
    length = float(np.hypot(*d))
    if length == 0:
        raise ValueError(f"gate {gate.name} has zero length")
    u = d / length
    return a, u, np.array([-u[1], u[0]]), length


def _ends(rows: np.ndarray, gate: Gate, h: float) -> tuple[list[_End], list[_End]]:
    """Last point (side at the end) and first point (side at the start) of every track."""
    a, u, nrm, length = _geometry(gate)
    lasts: list[_End] = []
    firsts: list[_End] = []
    for tid in np.unique(rows[:, 1]).astype(int):
        r = rows[rows[:, 1] == tid]
        r = r[np.argsort(r[:, 0], kind="stable")]
        p = bottom_centre(r[:, 2:6])
        s = (p - a) @ nrm
        al = (p - a) @ u
        sides = np.where(s > h, 1, np.where(s < -h, -1, 0))
        nz = np.flatnonzero(sides)
        if len(nz) == 0:
            continue
        end_side, start_side = int(sides[nz[-1]]), int(sides[nz[0]])
        if 0 <= al[-1] <= length:
            lasts.append(_End(tid, int(r[-1, 0]), p[-1], float(r[-1, 5]), float(s[-1]), end_side))
        if 0 <= al[0] <= length:
            firsts.append(_End(tid, int(r[0, 0]), p[0], float(r[0, 5]), float(s[0]), start_side))
    return lasts, firsts


def recover_crossings(
    rows: np.ndarray, gate: Gate, h: float, fps: float, params: RecoveryParams
) -> list[RecoveredCrossing]:
    """Crossings made by no single number (see module doc). ``rows`` = MOT rows."""
    if len(rows) == 0:
        return []
    lasts, firsts = _ends(rows, gate, h)
    max_gap = round(params.max_gap_s * fps)
    cands: list[tuple[int, float, _End, _End]] = []
    for e in lasts:
        if abs(e.s) > params.near_rel * e.h:
            continue
        for b in firsts:
            if b.tid == e.tid or b.side != -e.side:
                continue
            gap = b.frame - e.frame
            if gap <= 0 or gap > max_gap or abs(b.s) > params.near_rel * b.h:
                continue
            dist = float(np.hypot(*(b.p - e.p))) / max(e.h, b.h, 1.0)
            if dist <= params.max_dist_rel:
                cands.append((gap, dist, e, b))
    cands.sort(key=lambda c: (c[0], c[1], c[2].tid, c[3].tid))
    used_a: set[int] = set()
    used_b: set[int] = set()
    out: list[RecoveredCrossing] = []
    for gap, dist, e, b in cands:
        if e.tid in used_a or b.tid in used_b:
            continue
        used_a.add(e.tid)
        used_b.add(b.tid)
        out.append(
            RecoveredCrossing(  # plain Python numbers: these go straight into job JSON
                int(b.frame),
                int(b.tid),
                int(e.tid),
                gate.name,
                "in" if b.side > 0 else "out",
                int(gap),
                round(float(dist), 3),
            )
        )
    return sorted(out, key=lambda x: (x.frame, x.track_id))

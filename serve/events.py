"""Gate-event contract for ZIG (``docs/integration.md`` section 2).

Three event types, each with ``type``, ``track_id``, ``direction``, ``timestamp`` and
``confidence``:

* ``crossing``: a confirmed track crossed the gate line (hysteresis as in ``eval.gate_events``).
* ``tailgate``: a second track crossed the same gate in the same direction within
  ``tailgate_seconds`` of a first one (both crossings not held).
* ``ambiguous_hold``: a crossing happened while the tracker itself was unsure of that track, i.e.
  over the ``window_s`` before the crossing its estimated visibility dropped below
  ``hold_visibility`` (occlusion) or its mean box score was below ``hold_score``. Meaning:
  *tracker unsure, don't guess*; ZIG should fall back to its own rule (e.g. ask for a tap).
  A held crossing is not reported as a crossing and takes no part in tailgate pairs.

``ambiguous_hold`` uses only the per-frame state the online tracker already has; it does not
use the delayed (look-ahead) decision, which failed its go/no-go.

``confidence`` is a heuristic in [0, 1]: min(mean score, min visibility) over the window. It is
not calibrated. Trackers without a visibility estimate report visibility 1.0, so for them only
the score rule can hold a crossing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from eval.gate_events import Gate, crossings, tailgate_pairs
from tracker.types import FrameResult

EVENT_TYPES = ("crossing", "tailgate", "ambiguous_hold")


@dataclass(frozen=True)
class EventParams:
    hysteresis_px: float = 10.0
    tailgate_seconds: float = 1.0
    window_s: float = 0.5
    hold_visibility: float = 0.5
    hold_score: float = 0.5


def track_rows(results: list[FrameResult]) -> np.ndarray:
    """``frame, id, x, y, w, h, score, visibility`` for every reported box."""
    rows = [[r.frame_id, t.track_id, *t.tlwh, t.score, t.visibility]
            for r in results for t in r.tracks]  # fmt: skip
    return np.array(rows, float).reshape(-1, 8)


def _iso(start: datetime, seconds: float) -> str:
    return (start + timedelta(seconds=seconds)).isoformat(timespec="milliseconds")


def gate_events(results: list[FrameResult], fps: float, gate: Gate, start: datetime,
                params: EventParams | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """Contract events for one gate, sorted by frame. ``start`` = wall time of frame 1."""
    params = params or EventParams()
    rows = track_rows(results)
    if len(rows) == 0:
        return []
    w = max(1, round(params.window_s * fps))
    out: list[dict[str, Any]] = []
    kept = []
    conf: dict[tuple[int, int], float] = {}
    for e in crossings(rows, gate, params.hysteresis_px):
        r = rows[(rows[:, 1] == e.track_id) & (rows[:, 0] > e.frame - w) & (rows[:, 0] <= e.frame)]
        score, vis = float(r[:, 6].mean()), float(r[:, 7].min())
        c = round(max(0.0, min(1.0, score, vis)), 3)
        checks = (("occlusion", vis < params.hold_visibility),
                  ("low_confidence", score < params.hold_score))  # fmt: skip
        reasons = [name for name, bad in checks if bad]
        t_s = (e.frame - 1) / fps
        base = {"track_id": e.track_id, "direction": e.direction, "timestamp": _iso(start, t_s),
                "confidence": c, "frame": e.frame, "gate": e.gate}  # fmt: skip
        if reasons:
            out.append({"type": "ambiguous_hold", **base, "reasons": reasons})
        else:
            out.append({"type": "crossing", **base})
            kept.append(e)
            conf[(e.frame, e.track_id)] = c
    window = round(params.tailgate_seconds * fps)
    for e, prev_id in tailgate_pairs(kept, window):
        prev = max((p for p in kept if p.track_id == prev_id and p.frame <= e.frame
                    and p.gate == e.gate and p.direction == e.direction),
                   key=lambda p: p.frame)  # fmt: skip
        out.append({
            "type": "tailgate", "track_id": e.track_id, "direction": e.direction,
            "timestamp": _iso(start, (e.frame - 1) / fps),
            "confidence": min(conf[(e.frame, e.track_id)], conf[(prev.frame, prev.track_id)]),
            "frame": e.frame, "gate": e.gate, "lead_track_id": prev_id,
            "gap_s": round((e.frame - prev.frame) / fps, 3),
        })  # fmt: skip
    out.sort(key=lambda d: (d["frame"], EVENT_TYPES.index(d["type"])))
    return out

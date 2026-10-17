"""Per-job analysis of tracker output on a clip WITHOUT ground truth.

Everything here is computed from the tracker's own output, so it describes the tracker's
behaviour, not its accuracy. Accuracy (HOTA, IDF1, ID switches against ground truth) exists only
for the benchmark (``docs/results``).

* **Occlusion recovery** (``recovery``): a track that was *not reported* for at least
  ``min_gap`` frames (lost, usually hidden behind someone) or whose estimated visibility fell
  below ``occluded_vis``, and that comes back **with the same ID**. Every such event is an
  ID-preserving recovery: the tracker re-associated the person instead of starting a new ID.
* **Suspected ID switch** (``suspected_id_switch``): a heuristic proxy for an identity break,
  because a real ID switch needs ground truth. It fires when a track ends for good and a *new* ID
  is born within ``max_gap_s`` seconds, close to where the old one was last seen (bottom-centre
  distance ≤ ``max_dist`` × the old box height). It over- and under-counts; it is labelled as a
  heuristic everywhere it is shown.
* **Crossings** for clips without consent (``event_styling: neutral``): plain side changes of the
  gate line (``eval.gate_events.crossings``), no tailgate / hold classification. Consented clips
  get the full ZIG contract (``serve.events``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from eval.gate_events import Gate, crossings
from serve.events import EventParams, gate_events, track_rows
from tracker.types import FrameResult


@dataclass(frozen=True)
class AnalysisParams:
    min_gap: int = 3
    """Frames a track must be missing before its return counts as a recovery."""
    occluded_vis: float = 0.5
    max_gap_s: float = 1.0
    max_dist: float = 0.75
    min_track_frames: int = 3
    """Tracks shorter than this are ignored by the metrics (flicker)."""


def recoveries(
    rows: np.ndarray, fps: float, p: AnalysisParams | None = None
) -> list[dict[str, Any]]:
    """Recovery events from rows ``frame, id, x, y, w, h, score, visibility``."""
    p = p or AnalysisParams()
    out: list[dict[str, Any]] = []
    for tid in np.unique(rows[:, 1]).astype(int):
        r = rows[rows[:, 1] == tid]
        r = r[np.argsort(r[:, 0], kind="stable")]
        f = r[:, 0].astype(int)
        occluded_since: int | None = None
        for i in range(1, len(r)):
            gap = int(f[i] - f[i - 1] - 1)
            if gap >= p.min_gap:
                out.append(
                    {
                        "frame": int(f[i]),
                        "type": "recovery",
                        "track_id": int(tid),
                        "reason": "lost",
                        "gap_frames": gap,
                        "gap_s": round(gap / fps, 3),
                        "since": int(f[i - 1]),
                    }
                )
                occluded_since = None
                continue
            if r[i - 1, 7] < p.occluded_vis and occluded_since is None:
                occluded_since = int(f[i - 1])
            if occluded_since is not None and r[i, 7] >= p.occluded_vis:
                gap = int(f[i] - occluded_since)
                if gap >= p.min_gap:
                    out.append(
                        {
                            "frame": int(f[i]),
                            "type": "recovery",
                            "track_id": int(tid),
                            "reason": "occluded",
                            "gap_frames": gap,
                            "gap_s": round(gap / fps, 3),
                            "since": occluded_since,
                        }
                    )
                occluded_since = None
    return sorted(out, key=lambda e: (e["frame"], e["track_id"]))


def suspected_switches(
    rows: np.ndarray, fps: float, n_frames: int, p: AnalysisParams | None = None
) -> list[dict[str, Any]]:
    """Heuristic identity breaks: a track ends and a new ID is born near it soon after."""
    p = p or AnalysisParams()
    if len(rows) == 0:
        return []
    ids = np.unique(rows[:, 1]).astype(int)
    first: dict[int, np.ndarray] = {}
    last: dict[int, np.ndarray] = {}
    for tid in ids:
        r = rows[rows[:, 1] == tid]
        if len(r) < p.min_track_frames:
            continue
        r = r[np.argsort(r[:, 0], kind="stable")]
        first[int(tid)], last[int(tid)] = r[0], r[-1]
    max_gap = max(1, round(p.max_gap_s * fps))
    out: list[dict[str, Any]] = []
    used: set[int] = set()
    for old, lr in sorted(last.items(), key=lambda kv: kv[1][0]):
        end = int(lr[0])
        if end >= n_frames:  # still alive at the end of the clip
            continue
        bx, by, h = lr[2] + lr[4] / 2, lr[3] + lr[5], max(float(lr[5]), 1.0)
        best: tuple[float, int] | None = None
        for new, fr in first.items():
            start = int(fr[0])
            if new == old or new in used or not (end < start <= end + max_gap):
                continue
            d = float(np.hypot(fr[2] + fr[4] / 2 - bx, fr[3] + fr[5] - by)) / h
            if d <= p.max_dist and (best is None or d < best[0]):
                best = (d, new)
        if best is not None:
            used.add(best[1])
            start = int(first[best[1]][0])
            out.append(
                {
                    "frame": start,
                    "type": "suspected_id_switch",
                    "from_id": old,
                    "to_id": best[1],
                    "gap_frames": start - end,
                    "distance_rel": round(best[0], 3),
                    "heuristic": True,
                }
            )
    return sorted(out, key=lambda e: e["frame"])


def neutral_crossings(
    rows: np.ndarray, fps: float, gate: Gate, start: datetime, hysteresis_px: float = 10.0
) -> list[dict[str, Any]]:
    """Plain crossings (no tailgate / hold classification) for clips without consent."""
    if len(rows) == 0:
        return []
    return [
        {
            "type": "crossing",
            "track_id": e.track_id,
            "direction": e.direction,
            "timestamp": (start + timedelta(seconds=(e.frame - 1) / fps)).isoformat(
                timespec="milliseconds"
            ),
            "frame": e.frame,
            "gate": e.gate,
        }
        for e in crossings(rows, gate, hysteresis_px)
    ]


def gate_output(
    results: list[FrameResult],
    fps: float,
    gate: Gate,
    start: datetime,
    styling: str,
    hysteresis_px: float = 10.0,
) -> list[dict[str, Any]]:
    """``alert`` (consented clips): the full ZIG contract; ``neutral``: crossings only.

    Neutral crossings still say when the tracker was unsure of that track (the same rule that makes
    an ``ambiguous_hold`` on consented footage): ``unsure`` + ``reasons``. That is a statement about
    the tracker, not an alert about the person, so it is shown for every clip.
    """
    params = EventParams(hysteresis_px=hysteresis_px)
    if styling == "alert":
        return gate_events(results, fps, gate, start, params)
    ev = neutral_crossings(track_rows(results), fps, gate, start, hysteresis_px)
    held = {
        (e["frame"], e["track_id"]): e["reasons"]
        for e in gate_events(results, fps, gate, start, params)
        if e["type"] == "ambiguous_hold"
    }
    for e in ev:
        reasons = held.get((e["frame"], e["track_id"]))
        e["unsure"] = reasons is not None
        if reasons is not None:
            e["reasons"] = reasons
    return ev


def metrics(
    rows: np.ndarray,
    fps: float,
    n_frames: int,
    gate_ev: list[dict[str, Any]],
    rec: list[dict[str, Any]],
    sus: list[dict[str, Any]],
    p: AnalysisParams | None = None,
) -> dict[str, Any]:
    p = p or AnalysisParams()
    lengths = []
    for tid in np.unique(rows[:, 1]).astype(int) if len(rows) else []:
        f = rows[rows[:, 1] == tid, 0]
        if len(f) >= p.min_track_frames:
            lengths.append(int(f.max() - f.min() + 1))
    lens = np.asarray(lengths, float)
    counted = [e for e in gate_ev if e["type"] in ("crossing", "ambiguous_hold")]
    return {
        "frames": n_frames,
        "tracks": len(lengths),
        "avg_track_length_frames": round(float(lens.mean()), 1) if len(lens) else None,
        "avg_track_length_s": round(float(lens.mean()) / fps, 2) if len(lens) else None,
        "median_track_length_s": round(float(np.median(lens)) / fps, 2) if len(lens) else None,
        "boxes": len(rows),
        "id_preserving_recoveries": len(rec),
        "recoveries_after_lost": sum(e["reason"] == "lost" for e in rec),
        "recoveries_after_occluded": sum(e["reason"] == "occluded" for e in rec),
        "suspected_id_switches": len(sus),
        "crossings_in": sum(e["direction"] == "in" for e in counted),
        "crossings_out": sum(e["direction"] == "out" for e in counted),
        "tailgates": sum(e["type"] == "tailgate" for e in gate_ev),
        "ambiguous_holds": sum(e["type"] == "ambiguous_hold" for e in gate_ev),
        "unsure_crossings": sum(
            e["type"] == "ambiguous_hold" or bool(e.get("unsure")) for e in gate_ev
        ),
    }

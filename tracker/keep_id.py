"""Keep-ID tracker for the gate demonstration.

ByteTrack (and BoT-SORT) drop a lost track after ``track_buffer`` frames and mint a new number
when the person comes back. This tracker does the contract instead: one body, one number.

* Every live track is predicted with constant velocity, including LOST tracks.
* New detections are matched to those predicted boxes (Hungarian on centre distance).
* A miss does not kill the id: the predicted box is still reported (HELD) until ``buffer`` frames.
* A detection near a lost track rebinds that same id. A new id is born only when nothing nearby
  can claim the box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.types import Detections, FrameResult, TrackOutput


@dataclass
class _Track:
    tid: int
    x: float
    y: float
    w: float
    h: float
    vx: float
    vy: float
    score: float
    misses: int = 0
    hits: int = 1


class KeepIdTracker:
    """Online tracker that re-associates lost tracks instead of minting a new id."""

    def __init__(self, fps: float = 30.0, buffer: int = 90, max_dist: float = 96.0) -> None:
        self.fps = float(fps)
        self.buffer = int(buffer)
        self.max_dist = float(max_dist)
        self.tracks: list[_Track] = []
        self.next_id = 1
        self.frame_id = 0

    def update(self, dets: Detections) -> FrameResult:
        self.frame_id += 1
        for t in self.tracks:
            t.x += t.vx
            t.y += t.vy

        n_t, n_d = len(self.tracks), len(dets)
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        if n_t and n_d:
            cost = np.full((n_t, n_d), 1e6)
            for i, t in enumerate(self.tracks):
                for j in range(n_d):
                    cx = float(dets.tlwh[j, 0] + dets.tlwh[j, 2] / 2)
                    cy = float(dets.tlwh[j, 1] + dets.tlwh[j, 3] / 2)
                    d = math.hypot(t.x - cx, t.y - cy)
                    if d <= self.max_dist:
                        cost[i, j] = d
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c, strict=True):
                if cost[i, j] >= 1e5:
                    continue
                t = self.tracks[i]
                cx = float(dets.tlwh[j, 0] + dets.tlwh[j, 2] / 2)
                cy = float(dets.tlwh[j, 1] + dets.tlwh[j, 3] / 2)
                prev_x, prev_y = t.x - t.vx, t.y - t.vy
                t.vx = 0.5 * t.vx + 0.5 * (cx - prev_x)
                t.vy = 0.5 * t.vy + 0.5 * (cy - prev_y)
                t.x, t.y = cx, cy
                t.w = float(dets.tlwh[j, 2])
                t.h = float(dets.tlwh[j, 3])
                t.score = float(dets.scores[j])
                t.misses = 0
                t.hits += 1
                matched_t.add(i)
                matched_d.add(j)

        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.buffer]

        for j in range(n_d):
            if j in matched_d:
                continue
            if float(dets.scores[j]) < 0.12:
                continue
            cx = float(dets.tlwh[j, 0] + dets.tlwh[j, 2] / 2)
            cy = float(dets.tlwh[j, 1] + dets.tlwh[j, 3] / 2)
            self.tracks.append(
                _Track(
                    tid=self.next_id,
                    x=cx,
                    y=cy,
                    w=float(dets.tlwh[j, 2]),
                    h=float(dets.tlwh[j, 3]),
                    vx=0.0,
                    vy=0.0,
                    score=float(dets.scores[j]),
                )
            )
            self.next_id += 1

        out = tuple(
            TrackOutput(
                track_id=t.tid,
                tlwh=(t.x - t.w / 2, t.y - t.h / 2, t.w, t.h),
                score=t.score,
                state="confirmed" if t.misses == 0 else "lost",
                visibility=1.0 if t.misses == 0 else 0.2,
            )
            for t in self.tracks
        )
        return FrameResult(frame_id=self.frame_id, tracks=out)



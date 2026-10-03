"""Helpers shared by the SORT-family ports (OC-SORT, Deep OC-SORT, Hybrid-SORT).

``iou_batch``, ``speed_direction`` / ``speed_direction_batch``, ``k_previous_obs`` and
``linear_assignment`` are ported from OC-SORT ``trackers/ocsort_tracker/association.py`` and
``ocsort.py`` (commit 8462e7e, MIT, Copyright (c) 2021 Yifu Zhang).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from tracker.config import OutputConfig
from tracker.types import FloatArray, TrackOutput

IntArray = npt.NDArray[np.int64]
PLACEHOLDER = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
"""OC-SORT's "no observation" marker (any row whose last column is < 0)."""


def iou_batch(b1: FloatArray, b2: FloatArray) -> FloatArray:
    """IoU of ``(N, ≥4)`` × ``(M, ≥4)`` ``xyxy`` boxes → ``(N, M)`` (no +1, as SORT)."""
    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)))
    a = np.expand_dims(b1, 1)
    b = np.expand_dims(b2, 0)
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    wh = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = wh / (area_a + area_b - wh)
    return np.nan_to_num(np.asarray(out, np.float64))


def speed_direction(b1: FloatArray, b2: FloatArray) -> FloatArray:
    """Unit ``[dy, dx]`` from the centre of ``b1`` to the centre of ``b2``."""
    cx1, cy1 = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
    cx2, cy2 = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1], np.float64)
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return np.asarray(speed / norm, np.float64)


def speed_direction_batch(dets: FloatArray, tracks: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``(dy, dx)``, each ``(n_tracks, n_dets)``: unit direction track-obs centre → det centre."""
    tr = tracks[..., np.newaxis]
    cx1, cy1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    cx2, cy2 = (tr[:, 0] + tr[:, 2]) / 2.0, (tr[:, 1] + tr[:, 3]) / 2.0
    dx = cx1 - cx2
    dy = cy1 - cy2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    return dy / norm, dx / norm


def angle_cost(
    dets: FloatArray, velocities: FloatArray, previous_obs: FloatArray, inertia: float
) -> FloatArray:
    """OC-SORT's OCM term, ``(n_dets, n_tracks)``, already × inertia and × det score."""
    Y, X = speed_direction_batch(dets, previous_obs)
    iy = np.repeat(velocities[:, 0][:, None], Y.shape[1], axis=1)
    ix = np.repeat(velocities[:, 1][:, None], X.shape[1], axis=1)
    cos = np.clip(ix * X + iy * Y, -1.0, 1.0)
    diff = (np.pi / 2.0 - np.abs(np.arccos(cos))) / np.pi
    valid = np.ones(previous_obs.shape[0])
    valid[np.where(previous_obs[:, 4] < 0)] = 0
    valid_m = np.repeat(valid[:, None], X.shape[1], axis=1)
    cost = ((valid_m * diff) * inertia).T
    scores = np.repeat(dets[:, -1][:, None], previous_obs.shape[0], axis=1)
    return np.asarray(cost * scores, np.float64)


def k_previous_obs(observations: dict[int, FloatArray], cur_age: int, k: int) -> FloatArray:
    """Observation ``k`` steps back (or the nearest later one, or the newest, or placeholder)."""
    if len(observations) == 0:
        return PLACEHOLDER.copy()
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    return observations[max(observations.keys())]


def linear_assignment(cost: FloatArray) -> IntArray:
    """Min-cost assignment with no cost limit, ``(K, 2)`` (row, col) (scipy, as lap's fallback)."""
    if cost.size == 0:
        return np.empty((0, 2), np.int64)
    r, c = linear_sum_assignment(cost)
    return np.stack([r, c], axis=1).astype(np.int64)


def one_to_one_or_assign(iou: FloatArray, thr: float, total: FloatArray) -> IntArray:
    """OC-SORT shortcut: if ``iou > thr`` is already one-to-one use it, else maximise ``total``."""
    if min(iou.shape) == 0:
        return np.empty((0, 2), np.int64)
    a = (iou > thr).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        return np.stack(np.where(a), axis=1).astype(np.int64)
    return linear_assignment(-total)


def split_matches(
    matched: IntArray, iou: FloatArray, thr: float, n_dets: int, n_trks: int
) -> tuple[IntArray, IntArray, IntArray]:
    """Drop matches with IoU < thr; return (matches, unmatched dets, unmatched tracks)."""
    um_d = [d for d in range(n_dets) if d not in matched[:, 0]]
    um_t = [t for t in range(n_trks) if t not in matched[:, 1]]
    keep = []
    for m in matched:
        if iou[m[0], m[1]] < thr:
            um_d.append(int(m[0]))
            um_t.append(int(m[1]))
        else:
            keep.append(m)
    mt = np.stack(keep).astype(np.int64) if keep else np.empty((0, 2), np.int64)
    return mt, np.asarray(um_d, np.int64), np.asarray(um_t, np.int64)


def make_output(
    tid: int, xyxy: FloatArray, score: float, vis: float, oc: OutputConfig
) -> TrackOutput | None:
    """Apply the common MOT output filter (area > min, w/h ≤ max) → :class:`TrackOutput`."""
    x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
    w, h = x2 - x1, y2 - y1
    if w * h <= oc.min_box_area:
        return None
    if oc.max_aspect_ratio > 0 and h > 0 and w / h > oc.max_aspect_ratio:
        return None
    return TrackOutput(tid, (x1, y1, w, h), score, "confirmed", vis)

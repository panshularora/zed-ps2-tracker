"""Hybrid-SORT (IoU/motion variant, no re-ID), ported for CPU replay of cached detections.

Source: Hybrid-SORT ``trackers/hybrid_sort_tracker/hybrid_sort.py`` (``KalmanBoxTracker``,
``Hybrid_Sort.update``), ``association.py`` (``hmiou``, ``associate_4_points_with_score``,
``cost_vel``, ``cal_score_dif_batch(_two_score)``) and ``kalmanfilter_score_new.py``, commit
396f8d3, MIT, Copyright (c) 2021 Yifu Zhang (see ``NOTICE``). Defaults from
``exps/example/mot/yolox_x_ablation_hybrid_sort.py`` and ``utils/args.py``.

What it adds to OC-SORT:

* **TCM** (tracklet confidence modelling): the Kalman state carries the detection score
  (``z = [x, y, s, score, r]``, 9-D state with a score velocity). The first association
  subtracts ``|score_det − score_KF|`` (KF score clipped to ``[det_thresh, 1]``); the BYTE
  round subtracts ``|score_det − score_linear|`` where ``score_linear`` extrapolates the last two
  scores (clipped to ``[0.1, det_thresh]``).
* **ROCM**: the OCM direction term is computed for each of the four box corners (velocities
  summed over the last Δt observation gaps) and the four terms are added.
* **HMIoU**: IoU × height overlap ratio, used for every IoU in association.

Deviations: as for OC-SORT (cached boxes, common output filter, per-instance IDs); the
per-sequence ``track_thresh`` overrides in Hybrid-SORT's evaluator (MOT17-01/06/12/14 tweaks) are
**not** applied (pitfall §6.7); no interpolation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tracker.baselines.common import (
    PLACEHOLDER,
    iou_batch,
    k_previous_obs,
    linear_assignment,
    make_output,
    one_to_one_or_assign,
    split_matches,
)
from tracker.baselines.sort_kf import OruKalmanFilter
from tracker.config import TrackerConfig
from tracker.types import Detections, FloatArray, FrameResult, TrackOutput
from tracker.visibility import VisibilityEstimator, build_visibility

_F9 = np.eye(9)
for _i in range(4):
    _F9[_i, 5 + _i] = 1.0
_H9 = np.eye(5, 9)

# corner index pairs (x index, y index) in xyxy, in Hybrid-SORT's lt, rt, lb, rb order
CORNERS = ((0, 1), (0, 3), (2, 1), (2, 3))


def hmiou(b1: FloatArray, b2: FloatArray) -> FloatArray:
    """Height-modulated IoU ``(N, M)`` (Hybrid-SORT ``association.hmiou``)."""
    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)))
    a = np.expand_dims(b1, 1)
    b = np.expand_dims(b2, 0)
    yy11 = np.maximum(a[..., 1], b[..., 1])
    yy12 = np.minimum(a[..., 3], b[..., 3])
    yy21 = np.minimum(a[..., 1], b[..., 1])
    yy22 = np.maximum(a[..., 3], b[..., 3])
    with np.errstate(divide="ignore", invalid="ignore"):
        o = (yy12 - yy11) / (yy22 - yy21)
    return np.nan_to_num(np.asarray(o * iou_batch(b1, b2), np.float64))


def bbox_to_z5(bbox: FloatArray) -> FloatArray:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array(
        [bbox[0] + w / 2.0, bbox[1] + h / 2.0, w * h, bbox[4], w / float(h + 1e-6)], np.float64
    ).reshape(5, 1)


def x9_to_bbox(x: FloatArray) -> FloatArray:
    w = np.sqrt(x[2] * x[4])
    h = x[2] / w
    return np.array(
        [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0], np.float64
    ).reshape(1, 4)


def corner_direction(b1: FloatArray, b2: FloatArray, corner: int) -> FloatArray:
    xi, yi = CORNERS[corner]
    cx1, cy1 = b1[xi], b1[yi]
    cx2, cy2 = b2[xi], b2[yi]
    speed = np.array([cy2 - cy1, cx2 - cx1], np.float64)
    return np.asarray(speed / (np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6), np.float64)


def corner_cost(
    dets: FloatArray, velocities: FloatArray, previous_obs: FloatArray, inertia: float, corner: int
) -> FloatArray:
    """``cost_vel`` for one corner, ``(n_dets, n_tracks)``."""
    xi, yi = CORNERS[corner]
    tr = previous_obs[..., np.newaxis]
    dx = dets[:, xi] - tr[:, xi]
    dy = dets[:, yi] - tr[:, yi]
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    X, Y = dx / norm, dy / norm
    iy = np.repeat(velocities[:, 0][:, None], Y.shape[1], axis=1)
    ix = np.repeat(velocities[:, 1][:, None], X.shape[1], axis=1)
    cos = np.clip(ix * X + iy * Y, -1.0, 1.0)
    diff = (np.pi / 2.0 - np.abs(np.arccos(cos))) / np.pi
    valid = np.ones(previous_obs.shape[0])
    valid[np.where(previous_obs[:, 4] < 0)] = 0
    cost = ((np.repeat(valid[:, None], X.shape[1], axis=1) * diff) * inertia).T
    return np.asarray(cost * np.repeat(dets[:, -1][:, None], previous_obs.shape[0], axis=1))


class HybridTrack:
    def __init__(self, bbox: FloatArray, tid: int, delta_t: int, oru: bool,
                 det_thresh: float) -> None:  # fmt: skip
        kf = OruKalmanFilter(9, 5, "xyscr")
        kf.F = _F9.copy()
        kf.H = _H9.copy()
        kf.R[2:, 2:] *= 10.0
        kf.P[5:, 5:] *= 1000.0
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[-2, -2] *= 0.01
        kf.Q[5:, 5:] *= 0.01
        kf.x[:5] = bbox_to_z5(bbox)
        kf.oru = oru
        self.kf = kf
        self.det_thresh = det_thresh
        self.time_since_update = 0
        self.id = tid
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_observation: FloatArray = PLACEHOLDER.copy()
        self.observations: dict[int, FloatArray] = {}
        self.velocities: list[FloatArray | None] = [None, None, None, None]
        self.delta_t = delta_t
        self.confidence_pre: float | None = None
        self.confidence = float(bbox[-1])
        self.visibility = 1.0

    def update(self, bbox: FloatArray | None) -> None:
        if bbox is None:
            self.kf.update(None)
            self.confidence_pre = None
            return
        if self.last_observation.sum() >= 0:
            acc: list[FloatArray | None] = [None, None, None, None]
            previous_box = None
            for i in range(self.delta_t):
                if self.age - i - 1 in self.observations:
                    previous_box = self.observations[self.age - i - 1]
                    for c in range(4):
                        d = corner_direction(previous_box, bbox, c)
                        prev = acc[c]
                        acc[c] = d if prev is None else prev + d
            if previous_box is None:
                previous_box = self.last_observation
                self.velocities = [corner_direction(previous_box, bbox, c) for c in range(4)]
            else:
                self.velocities = acc
        self.last_observation = bbox
        self.observations[self.age] = bbox
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox_to_z5(bbox))
        self.confidence_pre = self.confidence
        self.confidence = float(bbox[-1])

    def predict(self) -> tuple[FloatArray, float, float]:
        if self.kf.x[7, 0] + self.kf.x[2, 0] <= 0:
            self.kf.x[7] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        box = x9_to_bbox(self.kf.x[:, 0])
        kalman_score = float(np.clip(self.kf.x[3, 0], self.det_thresh, 1.0))
        if not self.confidence_pre:
            simple = float(np.clip(self.confidence, 0.1, self.det_thresh))
        else:
            simple = float(np.clip(self.confidence - (self.confidence_pre - self.confidence),
                                   0.1, self.det_thresh))  # fmt: skip
        return box, kalman_score, simple

    def get_state(self) -> FloatArray:
        return x9_to_bbox(self.kf.x[:, 0])


class HybridSortTracker:
    """Hybrid-SORT (``name: hybrid_sort``)."""

    def __init__(self, cfg: TrackerConfig, frame_rate: float = 30.0, *,
                 visibility: VisibilityEstimator | None = None) -> None:  # fmt: skip
        self.cfg = cfg
        self.det_thresh = cfg.detection.high_thresh
        self.low_thresh = cfg.detection.low_thresh
        self.max_age = cfg.lifecycle.track_buffer
        self.min_hits = cfg.lifecycle.min_hits
        oc = cfg.ocsort
        self.iou_threshold = oc.iou_threshold
        self.delta_t = oc.delta_t
        self.inertia = oc.inertia
        self.asso = hmiou if oc.asso == "hmiou" else iou_batch
        self.visibility = visibility or build_visibility(cfg.visibility.estimator)
        self.trackers: list[HybridTrack] = []
        self.frame_count = 0
        self._next_id = 1
        self.extras: dict[str, Any] = {}

    def update(self, detections: Detections) -> FrameResult:
        self.frame_count += 1
        oc = self.cfg.ocsort
        xyxy, scores = detections.xyxy, detections.scores
        vis_all = self.visibility(xyxy)
        dets_all = np.concatenate([xyxy, scores[:, None]], axis=1)
        second = (scores > self.low_thresh) & (scores < self.det_thresh)
        remain = scores > self.det_thresh
        dets, dets_second = dets_all[remain], dets_all[second]
        vis, vis_second = vis_all[remain], vis_all[second]

        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        for t in range(len(self.trackers)):
            pos, ks, ss = self.trackers[t].predict()
            trks[t, :] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], ks, ss]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = trks[np.isfinite(trks).all(axis=1)]  # np.ma.compress_rows(masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        vels = [self._velocities(c) for c in range(4)]
        last_boxes = np.array([t.last_observation for t in self.trackers]).reshape(-1, 5)
        k_obs = np.array(
            [k_previous_obs(t.observations, t.age, self.delta_t) for t in self.trackers]
        ).reshape(-1, 5)

        # first round: HMIoU + ROCM − TCM
        iou = self.asso(dets, trks)
        if len(self.trackers) == 0:
            matched = np.empty((0, 2), np.int64)
        else:
            ang = sum(corner_cost(dets, vels[c], k_obs, self.inertia, c) for c in range(4))
            score_dif = np.abs(trks[None, :, 4] - dets[:, None, 4])
            total = iou + ang - score_dif * oc.tcm_first_weight
            matched = one_to_one_or_assign(iou, self.iou_threshold, total)
        matched, um_dets, um_trks = split_matches(
            matched, iou, self.iou_threshold, len(dets), len(self.trackers)
        )
        for d, t in matched:
            self._match(int(t), dets[d], vis[d])

        # BYTE round with TCM on the linear score
        if oc.use_byte and len(dets_second) > 0 and um_trks.shape[0] > 0:
            u_trks = trks[um_trks]
            iou_left = self.asso(dets_second, u_trks)
            if iou_left.max() > self.iou_threshold:
                iou_left = iou_left - np.abs(u_trks[None, :, 5] - dets_second[:, None, 4]) * (
                    oc.tcm_byte_weight
                )
                rem = []
                for m in linear_assignment(-iou_left):
                    d_i, t_i = int(m[0]), int(um_trks[m[1]])
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self._match(t_i, dets_second[d_i], vis_second[d_i])
                    rem.append(t_i)
                um_trks = np.setdiff1d(um_trks, np.asarray(rem, np.int64))

        # OCR on last observations
        if oc.ocr and um_dets.shape[0] > 0 and um_trks.shape[0] > 0:
            iou_left = self.asso(dets[um_dets], last_boxes[um_trks])
            if iou_left.max() > self.iou_threshold:
                rem_d, rem_t = [], []
                for m in linear_assignment(-iou_left):
                    d_i, t_i = int(um_dets[m[0]]), int(um_trks[m[1]])
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self._match(t_i, dets[d_i], vis[d_i])
                    rem_d.append(d_i)
                    rem_t.append(t_i)
                um_dets = np.setdiff1d(um_dets, np.asarray(rem_d, np.int64))
                um_trks = np.setdiff1d(um_trks, np.asarray(rem_t, np.int64))

        for t in um_trks:
            self.trackers[int(t)].update(None)
        for d in um_dets:
            trk = HybridTrack(dets[int(d)].copy(), self._next_id, self.delta_t, oc.oru,
                              self.det_thresh)  # fmt: skip
            self._next_id += 1
            trk.visibility = float(vis[int(d)])
            self.trackers.append(trk)

        out: list[TrackOutput] = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                box, score = trk.get_state()[0], 0.0
            else:
                box, score = trk.last_observation[:4], float(trk.last_observation[4])
            if trk.time_since_update < 1 and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                o = make_output(trk.id, box, score, trk.visibility, self.cfg.output)
                if o is not None:
                    out.append(o)
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return FrameResult(self.frame_count, tuple(sorted(out, key=lambda o: o.track_id)))

    def _velocities(self, corner: int) -> FloatArray:
        rows = []
        for t in self.trackers:
            v = t.velocities[corner]
            rows.append(np.zeros(2) if v is None else v)
        return np.array(rows).reshape(-1, 2)

    def _match(self, t: int, det: FloatArray, vis: float) -> None:
        trk = self.trackers[t]
        trk.update(det.copy())
        trk.visibility = float(vis)

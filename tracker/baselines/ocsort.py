"""OC-SORT and Deep OC-SORT, ported for CPU replay of cached detections.

Sources (MIT; see ``NOTICE``):

* OC-SORT ``trackers/ocsort_tracker/ocsort.py`` (``KalmanBoxTracker``, ``OCSort.update``) and
  ``association.py`` (``associate``), commit 8462e7e, Copyright (c) 2021 Yifu Zhang.
* Deep OC-SORT ``trackers/integrated_ocsort_embedding/ocsort.py`` and ``association.py``
  (dynamic appearance, adaptive weighting, CMC on the SORT state), commit 6bb51d0,
  Copyright (c) 2023 Gerard Maggiolino.

Per frame (``OCSort.update``):

1. keep detections with score > ``det_thresh``; (BYTE only) ``0.1 < score < det_thresh`` →
   second set;
2. (Deep, CMC on) warp every track with the frame's affine; predict every track;
3. first association: maximise ``IoU + OCM angle term (+ appearance)`` over predicted boxes, after
   the "already one-to-one" shortcut; drop pairs with IoU < ``iou_threshold``;
4. (BYTE) unmatched tracks ↔ low detections on IoU;
5. OCR: unmatched tracks' *last observations* ↔ unmatched detections on IoU;
6. unmatched tracks coast (``update(None)``, which freezes the filter for ORU); unmatched
   detections start tracks;
7. report tracks updated this frame with ``hit_streak ≥ min_hits`` (or in the first
   ``min_hits`` frames), box = last observation; drop tracks unseen for > ``max_age``.

Deviations (all deliberate, see ``docs/implementation_notes.md``): detections arrive as cached
boxes (no image rescaling); appearance is the cached embedding (Deep OC-SORT's grid/"split"
cosine is off, as in its best runs with ``--grid_off``); Deep OC-SORT's ``new_kf`` is off
(``--new_kf_off``, the README's runs); when a frame has no embeddings the appearance term is 0
(``--emb_off --aw_off``). Track IDs are per-instance counters, the output filter is the common
one (``output.*``). No offline post-processing (Deep OC-SORT's reported numbers use ``--post``
linear interpolation; we don't).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from tracker.baselines.common import (
    PLACEHOLDER,
    angle_cost,
    iou_batch,
    k_previous_obs,
    linear_assignment,
    make_output,
    one_to_one_or_assign,
    speed_direction,
    split_matches,
)
from tracker.baselines.sort_kf import OruKalmanFilter, bbox_to_z, x_to_bbox
from tracker.cmc import CMC, NoCMC
from tracker.config import TrackerConfig
from tracker.types import Detections, FloatArray, FrameResult, TrackOutput
from tracker.visibility import VisibilityEstimator, build_visibility

_F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1],
               [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 1]], dtype=np.float64)  # fmt: skip
_H = np.eye(4, 7)


class OCTrack:
    """OC-SORT ``KalmanBoxTracker`` (+ Deep OC-SORT's embedding and affine correction)."""

    def __init__(self, bbox: FloatArray, tid: int, delta_t: int, oru: bool,
                 emb: npt.NDArray[np.float32] | None, anchor: str) -> None:  # fmt: skip
        kf = OruKalmanFilter(7, 4, "xysr")
        kf.F = _F.copy()
        kf.H = _H.copy()
        kf.R[2:, 2:] *= 10.0
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        kf.x[:4] = bbox_to_z(bbox)
        kf.oru = oru
        kf.anchor = "last_measurement" if anchor == "last_measurement" else "history"
        self.kf = kf
        self.time_since_update = 0
        self.id = tid
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_observation: FloatArray = PLACEHOLDER.copy()
        self.observations: dict[int, FloatArray] = {}
        self.velocity: FloatArray | None = None
        self.delta_t = delta_t
        self.emb = emb
        self.visibility = 1.0

    def update(self, bbox: FloatArray | None) -> None:
        if bbox is not None:
            if self.last_observation.sum() >= 0:
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                self.velocity = speed_direction(previous_box, bbox)
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(bbox_to_z(bbox))
        else:
            self.kf.update(None)

    def update_emb(self, emb: npt.NDArray[np.float32], alpha: float) -> None:
        if self.emb is None:
            self.emb = emb.copy()
            return
        e = alpha * self.emb + (1 - alpha) * emb
        self.emb = (e / np.linalg.norm(e)).astype(np.float32)

    def predict(self) -> FloatArray:
        if self.kf.x[6, 0] + self.kf.x[2, 0] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_bbox(self.kf.x[:, 0])

    def get_state(self) -> FloatArray:
        return x_to_bbox(self.kf.x[:, 0])

    def apply_affine_correction(self, affine: FloatArray) -> None:
        """Deep OC-SORT ``KalmanBoxTracker.apply_affine_correction`` (in-place, as upstream:
        ``last_observation`` and ``observations[age]`` may be the same array)."""
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)
        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)
        self.kf.apply_affine_correction(m, t)


def aw_weights(emb_cost: FloatArray, w_assoc_emb: float, max_diff: float) -> FloatArray:
    """Deep OC-SORT ``compute_aw_new_metric``: boost rows/cols with a clear top-1."""
    w = np.full_like(emb_cost, w_assoc_emb)
    bonus = np.zeros_like(emb_cost)
    if emb_cost.shape[1] >= 2:
        for i in range(emb_cost.shape[0]):
            inds = np.argsort(-emb_cost[i])
            bonus[i] += min(emb_cost[i, inds[0]] - emb_cost[i, inds[1]], max_diff) / 2
    if emb_cost.shape[0] >= 2:
        for j in range(emb_cost.shape[1]):
            inds = np.argsort(-emb_cost[:, j])
            bonus[:, j] += min(emb_cost[inds[0], j] - emb_cost[inds[1], j], max_diff) / 2
    return np.asarray(w + bonus, np.float64)


class OCSortTracker:
    """OC-SORT (``name: ocsort``) or Deep OC-SORT (``name: deep_ocsort``)."""

    def __init__(self, cfg: TrackerConfig, frame_rate: float = 30.0, *, cmc: CMC | None = None,
                 visibility: VisibilityEstimator | None = None) -> None:  # fmt: skip
        self.cfg = cfg
        oc = cfg.ocsort
        self.deep = cfg.name == "deep_ocsort"
        self.det_thresh = cfg.detection.high_thresh
        self.low_thresh = cfg.detection.low_thresh
        self.max_age = cfg.lifecycle.track_buffer
        self.min_hits = cfg.lifecycle.min_hits
        self.iou_threshold = oc.iou_threshold
        self.delta_t = oc.delta_t
        self.inertia = oc.inertia
        self.cmc: CMC = cmc or NoCMC()
        self.visibility = visibility or build_visibility(cfg.visibility)
        self.trackers: list[OCTrack] = []
        self.frame_count = 0
        self._next_id = 1
        self.extras: dict[str, Any] = {}

    def _new_track(self, det: FloatArray, emb: npt.NDArray[np.float32] | None) -> OCTrack:
        t = OCTrack(det, self._next_id, self.delta_t, self.cfg.ocsort.oru, emb,
                    "last_measurement" if self.deep else "history")  # fmt: skip
        self._next_id += 1
        return t

    def update(self, detections: Detections) -> FrameResult:
        self.frame_count += 1
        oc = self.cfg.ocsort
        xyxy = detections.xyxy
        scores = detections.scores
        vis_all = self.visibility(xyxy, scores)
        dets_all = np.concatenate([xyxy, scores[:, None]], axis=1)
        remain = scores > self.det_thresh
        second = (scores > self.low_thresh) & (scores < self.det_thresh)
        dets, dets_second = dets_all[remain], dets_all[second]
        vis, vis_second = vis_all[remain], vis_all[second]
        embs_all = detections.embeddings
        embs = None if embs_all is None or not self.deep else embs_all[remain]
        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = oc.alpha_fixed_emb
        dets_alpha = af + (1 - af) * (1 - trust)

        if self.deep:
            aff = self.cmc.affine(self.frame_count)
            if aff is not None:
                for trk in self.trackers:
                    trk.apply_affine_correction(aff)

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t in range(len(self.trackers)):
            pos = self.trackers[t].predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = trks[np.isfinite(trks).all(axis=1)]  # np.ma.compress_rows(masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [t.velocity if t.velocity is not None else np.zeros(2) for t in self.trackers]
        ).reshape(-1, 2)
        last_boxes = np.array([t.last_observation for t in self.trackers]).reshape(-1, 5)
        k_obs = np.array(
            [k_previous_obs(t.observations, t.age, self.delta_t) for t in self.trackers]
        ).reshape(-1, 5)

        # 1st round
        iou = iou_batch(dets, trks)
        if len(self.trackers) == 0:
            matched = np.empty((0, 2), np.int64)
        else:
            total = iou + angle_cost(dets, velocities, k_obs, self.inertia)
            if embs is not None and len(dets) and all(t.emb is not None for t in self.trackers):
                trk_embs = np.stack([t.emb for t in self.trackers if t.emb is not None])
                emb_cost = (embs @ trk_embs.T).astype(np.float64)
                w = (aw_weights(emb_cost, oc.w_assoc_emb, oc.aw_param)
                     if oc.adaptive_weighting else oc.w_assoc_emb)  # fmt: skip
                total = total + emb_cost * w
            matched = one_to_one_or_assign(iou, self.iou_threshold, total)
        matched, um_dets, um_trks = split_matches(
            matched, iou, self.iou_threshold, len(dets), len(self.trackers)
        )
        for d, t in matched:
            self._match(t, dets[d], vis[d], None if embs is None else embs[d], dets_alpha[d])

        # BYTE (OC-SORT only, off by default)
        if oc.use_byte and len(dets_second) > 0 and um_trks.shape[0] > 0:
            u_trks = trks[um_trks]
            iou_left = iou_batch(dets_second, u_trks)
            if iou_left.max() > self.iou_threshold:
                rem = []
                for m in linear_assignment(-iou_left):
                    d_i, t_i = int(m[0]), int(um_trks[m[1]])
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self._match(t_i, dets_second[d_i], vis_second[d_i], None, 1.0)
                    rem.append(t_i)
                um_trks = np.setdiff1d(um_trks, np.asarray(rem, np.int64))

        # OCR
        if oc.ocr and um_dets.shape[0] > 0 and um_trks.shape[0] > 0:
            left_dets = dets[um_dets]
            left_trks = last_boxes[um_trks]
            iou_left = iou_batch(left_dets, left_trks)
            if iou_left.max() > self.iou_threshold:
                rem_d, rem_t = [], []
                for m in linear_assignment(-iou_left):
                    d_i, t_i = int(um_dets[m[0]]), int(um_trks[m[1]])
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self._match(t_i, dets[d_i], vis[d_i], None if embs is None else embs[d_i],
                                dets_alpha[d_i])  # fmt: skip
                    rem_d.append(d_i)
                    rem_t.append(t_i)
                um_dets = np.setdiff1d(um_dets, np.asarray(rem_d, np.int64))
                um_trks = np.setdiff1d(um_trks, np.asarray(rem_t, np.int64))

        for t in um_trks:
            self.trackers[int(t)].update(None)
        for d in um_dets:
            trk = self._new_track(dets[int(d)].copy(),
                                  None if embs is None else embs[int(d)].copy())  # fmt: skip
            trk.visibility = float(vis[int(d)])
            self.trackers.append(trk)

        out: list[TrackOutput] = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                box = trk.get_state()[0]
                score = 0.0
            else:
                box = trk.last_observation[:4]
                score = float(trk.last_observation[4])
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

    def _match(self, t: int, det: FloatArray, vis: float,
               emb: npt.NDArray[np.float32] | None, alpha: float) -> None:  # fmt: skip
        trk = self.trackers[int(t)]
        trk.update(det.copy())
        trk.visibility = float(vis)
        if emb is not None:
            trk.update_emb(emb, float(alpha))

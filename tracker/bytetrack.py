"""``ByteTrackLike``: ByteTrack's two-stage association on our own components.

Per frame (Zhang et al., 2021, Alg. 1, with BoT-SORT's ``xywh`` Kalman filter):

1. Split detections: ``D_high`` (score > τ_high) and ``D_low`` (τ_low < score ≤ τ_high).
2. Predict every track with the Kalman filter.
3. First association: confirmed + lost tracks ↔ ``D_high`` on IoU cost (optionally fused with
   detection score and, if enabled, with re-ID appearance).
4. Second association: still-unmatched *confirmed* tracks ↔ ``D_low`` on IoU only. Low-score
   boxes are often occluded people, which is ByteTrack's key idea.
5. Unmatched confirmed tracks become lost.
6. Tentative tracks ↔ leftover ``D_high``; unmatched tentative tracks are removed.
7. Leftover ``D_high`` above τ_new start new tentative tracks.
8. Lost tracks older than the buffer are removed; near-duplicate confirmed/lost pairs are pruned.
9. Report confirmed tracks updated this frame.

With ``cmc`` set, every predicted track is warped by the frame's camera-motion affine (BoT-SORT).
``name: botsort`` is this class with BoT-SORT's settings (``configs/baselines/botsort.yaml``).

Known, intentional differences from the reference implementation (documented in
``docs/decisions.md``): ``xywh`` instead of ``xyah`` Kalman state, every track (including
tentative ones) is Kalman-predicted each frame, and IDs are per-tracker-instance counters.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from tracker.assignment import Assigner, HungarianAssigner, Matching
from tracker.cmc import CMC, NoCMC
from tracker.config import TrackerConfig
from tracker.cost import (
    cosine_cost,
    fuse_iou_appearance,
    fuse_score,
    iou_cost,
    visibility_softened_iou_cost,
)
from tracker.gallery import EmaAppearance
from tracker.kalman import KalmanFilterXYWH
from tracker.lifecycle import LifecycleConfig, Track, TrackState
from tracker.types import Detections, FloatArray, FrameResult, TrackOutput, tlwh_to_xywh
from tracker.visibility import VisibilityEstimator, build_visibility


def apply_gmc(
    means: FloatArray, covs: FloatArray, warp: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """BoT-SORT ``STrack.multi_gmc`` on ``(N, 8)`` xywh means and ``(N, 8, 8)`` covariances.

    ``R8 = kron(I4, A[:, :2])`` rotates/scales every (x, y)-pair of the state, then the
    translation is added to the position. Ported from BoT-SORT ``tracker/bot_sort.py``
    (commit 2519854, MIT, Copyright (c) 2022 Nir Aharon).
    """
    R = warp[:2, :2]
    R8 = np.kron(np.eye(4), R)
    t = warp[:2, 2]
    means = means @ R8.T
    means[:, :2] += t
    covs = R8 @ covs @ R8.T
    return means, covs


Stage1Hook = Callable[[list[Track], FloatArray], FloatArray]
"""Rewrites the first-association cost matrix (rows = tracks, cols = high detections)."""


@dataclass(frozen=True)
class Stage1Record:
    """What the first association saw this frame (read by :mod:`tracker.delayed`)."""

    track_ids: tuple[int, ...]
    lost: tuple[bool, ...]
    """Per row: the track was lost (not confirmed) before this frame."""
    cost: FloatArray
    """Cost matrix before any hook."""
    dets_tlwh: FloatArray
    matches: npt.NDArray[np.int64]
    thresh: float
    dets_vis: FloatArray | None = None
    """Estimated visibility of each high-score detection (column), from the estimator."""


class ByteTrackLike:
    """Online tracker. Call :meth:`update` once per frame, in order."""

    def __init__(
        self,
        config: TrackerConfig | None = None,
        frame_rate: float = 30.0,
        *,
        assigner: Assigner | None = None,
        visibility: VisibilityEstimator | None = None,
        cmc: CMC | None = None,
        allow_delayed: bool = False,
    ) -> None:
        """``allow_delayed`` is set by :class:`tracker.delayed.DelayedTracker`, which wraps this
        class. Built on its own with ``delayed.enabled`` it raises instead of silently running
        the immediate baseline."""
        self.cfg = config or TrackerConfig()
        if self.cfg.delayed.enabled and not allow_delayed:
            raise ValueError(
                "delayed.enabled: build through tracker.runner.build_tracker (DelayedTracker)"
            )
        lc = self.cfg.lifecycle
        self.lifecycle = LifecycleConfig(
            min_hits=lc.min_hits,
            tentative_max_misses=lc.tentative_max_misses,
            max_time_lost=int(
                (frame_rate / 30.0 if lc.scale_buffer_by_fps else 1.0) * lc.track_buffer
            ),
            confirm_on_first_frame=lc.confirm_on_first_frame,
        )
        self._lc_occluded = dataclasses.replace(
            self.lifecycle,
            max_time_lost=round(
                self.lifecycle.max_time_lost * self.cfg.visibility.occluded_buffer_scale
            ),
        )
        self.kf = KalmanFilterXYWH(self.cfg.kalman)
        self.assign: Assigner = assigner or HungarianAssigner(self.cfg.association.backend)
        self.visibility: VisibilityEstimator = visibility or build_visibility(self.cfg.visibility)
        self.cmc: CMC = cmc or NoCMC()
        self.tracks: list[Track] = []
        self.frame_id = 0
        self._next_id = 1
        self.stage1_hook: Stage1Hook | None = None
        self.record = False
        """When True, fill :attr:`last_stage1` and :attr:`match_costs` every frame."""
        self.last_stage1: Stage1Record | None = None
        self.match_costs: dict[int, float] = {}
        """track_id → association cost of its match this frame (any stage)."""
        self.trace_enabled = False
        """When True, fill :attr:`last_trace` with this frame's association decisions (read-only
        explainability: candidates, IoU / fused costs, thresholds, expiries and the buffer used)."""
        self.last_trace: dict[str, Any] | None = None

    def clone(self) -> ByteTrackLike:
        """Independent copy of the tracking state; stateless helpers (KF, CMC, estimator,
        assigner, config) are shared."""
        new = copy.copy(self)
        new.tracks = copy.deepcopy(self.tracks)
        new.match_costs = {}
        new.stage1_hook = None
        return new

    # ------------------------------------------------------------------ helpers
    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    @staticmethod
    def _boxes(tracks: list[Track]) -> FloatArray:
        if not tracks:
            return np.zeros((0, 4))
        return np.stack([t.xyxy for t in tracks])

    def _cost(
        self,
        tracks: list[Track],
        dets: Detections,
        *,
        use_score: bool,
        use_appearance: bool,
        vis: FloatArray | None = None,
        soften: bool = False,
    ) -> FloatArray:
        plus_one = self.cfg.association.iou_plus_one
        c = iou_cost(self._boxes(tracks), dets.xyxy, plus_one)
        iou_only = c
        beta = self.cfg.visibility.soften_weight
        if soften and beta > 0 and vis is not None:
            c = visibility_softened_iou_cost(self._boxes(tracks), dets.xyxy, vis, beta, plus_one)
        if use_score and self.cfg.association.fuse_score:
            c = fuse_score(c, dets.scores)
        if use_appearance and tracks and len(dets) and dets.embeddings is not None:
            mem = [t.appearance for t in tracks]
            if all(m is not None for m in mem):
                track_emb = np.stack([m.embedding for m in mem if m is not None])
                app = cosine_cost(track_emb, dets.embeddings)
                ac = self.cfg.appearance
                if ac.cost_min_visibility > 0 and vis is not None:
                    app = app.copy()
                    app[:, vis < ac.cost_min_visibility] = 2.0  # max cosine distance: ignored
                c = fuse_iou_appearance(
                    iou_only,
                    app,
                    proximity_thresh=ac.proximity_thresh,
                    appearance_thresh=ac.appearance_thresh,
                    motion_cost=c,
                )
        return c

    def _apply_matches(
        self,
        tracks: list[Track],
        dets: Detections,
        vis: FloatArray,
        m: Matching,
        cost: FloatArray | None = None,
    ) -> None:
        use_app = self.cfg.appearance.enabled and dets.embeddings is not None
        for r, c in m.matches:
            t = tracks[int(r)]
            if self.record and cost is not None:
                self.match_costs[t.track_id] = float(cost[r, c])
            t.mark_matched(self.kf, dets.tlwh[c], float(dets.scores[c]), self.frame_id,
                           self.lifecycle)  # fmt: skip
            t.visibility = float(vis[c])
            if use_app and dets.embeddings is not None:
                if t.appearance is None:
                    t.appearance = self._new_memory(dets.embeddings[c])
                else:
                    t.appearance.update(dets.embeddings[c], t.visibility)

    def _new_memory(self, emb: npt.NDArray[np.float32]) -> EmaAppearance:
        ac = self.cfg.appearance
        return EmaAppearance(emb.astype(np.float32), ac.alpha, ac.min_visibility)

    # ------------------------------------------------------------------ main loop
    def update(self, dets: Detections) -> FrameResult:
        self.frame_id += 1
        cfg = self.cfg
        use_app = cfg.appearance.enabled and dets.embeddings is not None

        vis_all = self.visibility(dets.xyxy, dets.scores)
        s = dets.scores
        hi_mask = s > cfg.detection.high_thresh
        lo_mask = (s > cfg.detection.low_thresh) & (s < cfg.detection.high_thresh)
        d_high, v_high = dets.subset(hi_mask), vis_all[hi_mask]
        d_low, v_low = dets.subset(lo_mask), vis_all[lo_mask]

        # 2. predict all live tracks (lost tracks: freeze size velocity, as BoT-SORT does)
        live = [t for t in self.tracks if t.state is not TrackState.REMOVED]
        if live:
            means = np.stack([t.mean for t in live])
            covs = np.stack([t.covariance for t in live])
            for i, t in enumerate(live):
                if t.state is not TrackState.CONFIRMED and t.state is not TrackState.TENTATIVE:
                    means[i, 6:8] = 0.0
            pred = np.array([cfg.lifecycle.predict_tentative or t.state is not TrackState.TENTATIVE
                             for t in live])  # fmt: skip
            if pred.any():
                pm, pc = self.kf.multi_predict(means[pred], covs[pred])
                means[pred], covs[pred] = pm, pc
            warp = self.cmc.affine(self.frame_id)
            if warp is not None:
                means, covs = apply_gmc(means, covs, warp)
            for i, t in enumerate(live):
                t.mean, t.covariance = means[i], covs[i]

        tentative = [t for t in live if t.state is TrackState.TENTATIVE]
        pool = [t for t in live if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tr: dict[str, Any] | None = None
        if self.trace_enabled:
            tr = {
                "frame": self.frame_id,
                "high_idx": np.nonzero(hi_mask)[0].tolist(),
                "low_idx": np.nonzero(lo_mask)[0].tolist(),
                "det_vis": vis_all.tolist(),
                "tracks": {
                    t.track_id: {
                        "state": t.state.value,
                        "pred_tlwh": [float(v) for v in t.tlwh],
                        "frames_lost": self.frame_id - 1 - t.last_frame,
                        "visibility": float(t.visibility),
                    }
                    for t in live
                },
                "stages": [],
                "births": [],
                "expired": [],
            }

        # 3. first association: confirmed + lost ↔ high
        self.match_costs = {}
        c1 = self._cost(
            pool, d_high, use_score=True, use_appearance=use_app, vis=v_high, soften=True
        )
        c1_used = c1 if self.stage1_hook is None else self.stage1_hook(pool, c1.copy())
        m1 = self.assign(c1_used, cfg.association.match_thresh)
        if self.record:
            self.last_stage1 = Stage1Record(
                tuple(t.track_id for t in pool),
                tuple(t.state is TrackState.LOST for t in pool), c1, d_high.tlwh, m1.matches,
                cfg.association.match_thresh, v_high,
            )  # fmt: skip
        if tr is not None:
            tr["stages"].append(
                self._trace_stage(
                    "high", pool, d_high, c1_used, m1, cfg.association.match_thresh, tr["high_idx"]
                )
            )
        self._apply_matches(pool, d_high, v_high, m1, c1)

        # 4. second association: unmatched *confirmed* ↔ low, IoU only
        r_tracked = [pool[i] for i in m1.unmatched_rows if pool[i].state is TrackState.CONFIRMED]
        c2 = self._cost(
            r_tracked, d_low, use_score=False, use_appearance=False, vis=v_low, soften=True
        )
        m2 = self.assign(c2, cfg.association.second_match_thresh)
        if tr is not None:
            tr["stages"].append(
                self._trace_stage(
                    "low",
                    r_tracked,
                    d_low,
                    c2,
                    m2,
                    cfg.association.second_match_thresh,
                    tr["low_idx"],
                )
            )
        self._apply_matches(r_tracked, d_low, v_low, m2, c2)

        # 5. unmatched confirmed → lost
        for i in m2.unmatched_rows:
            r_tracked[int(i)].mark_missed(self.lifecycle)
        for i in m1.unmatched_rows:
            t = pool[int(i)]
            if t.state is TrackState.LOST:
                t.mark_missed(self.lifecycle)

        # 6. tentative ↔ leftover high
        left = m1.unmatched_cols
        d_left, v_left = d_high.subset(left), v_high[left]
        c3 = self._cost(tentative, d_left, use_score=True, use_appearance=use_app, vis=v_left)
        m3 = self.assign(c3, cfg.association.unconfirmed_match_thresh)
        if tr is not None:
            left_idx = [tr["high_idx"][int(j)] for j in left]
            tr["stages"].append(
                self._trace_stage(
                    "tentative",
                    tentative,
                    d_left,
                    c3,
                    m3,
                    cfg.association.unconfirmed_match_thresh,
                    left_idx,
                )
            )
        self._apply_matches(tentative, d_left, v_left, m3, c3)
        for i in m3.unmatched_rows:
            tentative[int(i)].mark_missed(self.lifecycle)

        # 7. births
        for j in m3.unmatched_cols:
            if d_left.scores[j] < cfg.detection.new_track_thresh:
                continue
            self._birth(d_left, int(j), float(v_left[j]))
            if tr is not None:
                tr["births"].append(
                    {"track_id": self.tracks[-1].track_id, "det": tr["high_idx"][int(left[int(j)])]}
                )

        # 8. expiry + duplicate pruning
        for t in self.tracks:
            occluded = t.visibility < self.cfg.visibility.occluded_thresh
            lc_t = self._lc_occluded if occluded else self.lifecycle
            if tr is not None and t.state is TrackState.LOST:
                tr["tracks"].setdefault(t.track_id, {}).update(
                    buffer=lc_t.max_time_lost,
                    buffer_extended=occluded
                    and lc_t is not self.lifecycle
                    and lc_t.max_time_lost != self.lifecycle.max_time_lost,
                )
            if self.cfg.lifecycle.removal_lag and t.state is TrackState.LOST:
                if t.removal_flagged:
                    t.remove()  # upstream: its id is in removed_stracks -> dropped from lost
                    continue
                if self.frame_id - t.last_frame > lc_t.max_time_lost:
                    t.removal_flagged = True  # marked removed, but still in lost_stracks once more
                    continue
            t.expire_if_stale(self.frame_id, lc_t)
        self._remove_duplicates()
        if tr is not None:
            for t in self.tracks:
                if t.state is TrackState.REMOVED:
                    lost_for = self.frame_id - t.last_frame
                    occl = t.visibility < self.cfg.visibility.occluded_thresh
                    buf = (self._lc_occluded if occl else self.lifecycle).max_time_lost
                    tr["expired"].append(
                        {
                            "track_id": t.track_id,
                            "frames_lost": lost_for,
                            "buffer": buf,
                            "last_visibility": float(t.visibility),
                            "reason": "buffer" if lost_for > buf else "duplicate_or_tentative",
                        }
                    )
            self.last_trace = tr
        self.tracks = [t for t in self.tracks if t.state is not TrackState.REMOVED]

        return FrameResult(self.frame_id, self._report())

    def _trace_stage(
        self,
        name: str,
        tracks: list[Track],
        dets: Detections,
        cost: FloatArray,
        m: Matching,
        thresh: float,
        det_idx: list[int],
    ) -> dict[str, Any]:
        """One association stage as data: rows = track ids, cols = original detection indices."""
        iou = 1.0 - iou_cost(self._boxes(tracks), dets.xyxy, self.cfg.association.iou_plus_one)
        return {
            "stage": name,
            "thresh": thresh,
            "track_ids": [t.track_id for t in tracks],
            "det_idx": list(det_idx),
            "iou": np.round(iou, 4).tolist(),
            "cost": np.round(cost, 4).tolist(),
            "matches": [[tracks[int(r)].track_id, det_idx[int(c)]] for r, c in m.matches],
        }

    def _birth(self, dets: Detections, j: int, vis: float) -> None:
        mean, cov = self.kf.initiate(tlwh_to_xywh(dets.tlwh[j])[0])
        first = self.frame_id == 1 and self.lifecycle.confirm_on_first_frame
        state = (
            TrackState.CONFIRMED if first or self.lifecycle.min_hits <= 1 else TrackState.TENTATIVE
        )
        t = Track(
            track_id=self._new_id(),
            mean=mean,
            covariance=cov,
            state=state,
            start_frame=self.frame_id,
            last_frame=self.frame_id,
            score=float(dets.scores[j]),
            visibility=vis,
        )
        if self.cfg.appearance.enabled and dets.embeddings is not None:
            t.appearance = self._new_memory(dets.embeddings[j])
        self.tracks.append(t)

    def _remove_duplicates(self) -> None:
        """ByteTrack/BoT-SORT ``remove_duplicate_stracks(tracked, lost)``.

        ``tracked`` includes tentative tracks (they live in ``tracked_stracks`` upstream). For each
        pair closer than ``duplicate_iou_dist`` the one with the shorter *tracked* history
        (``last_frame − start_frame``, upstream's ``frame_id − start_frame``) is dropped; pairs are
        judged independently and the drops applied together, as upstream.
        """
        conf = [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.TENTATIVE)]
        lost = [t for t in self.tracks if t.state is TrackState.LOST]
        if not conf or not lost:
            return
        d = iou_cost(self._boxes(conf), self._boxes(lost), self.cfg.association.iou_plus_one)
        drop: list[Track] = []
        for a, b in zip(*np.nonzero(d < self.cfg.association.duplicate_iou_dist), strict=True):
            ta, tb = conf[int(a)], lost[int(b)]
            age_a = ta.last_frame - ta.start_frame
            age_b = tb.last_frame - tb.start_frame
            drop.append(tb if age_a > age_b else ta)
        for t in drop:
            if t.state is not TrackState.REMOVED:
                t.remove()

    def _report(self) -> tuple[TrackOutput, ...]:
        out: list[TrackOutput] = []
        oc = self.cfg.output
        for t in self.tracks:
            if t.last_frame != self.frame_id:
                continue
            if t.state is not TrackState.CONFIRMED and not (
                oc.report_tentative and t.state is TrackState.TENTATIVE
            ):
                continue
            x, y, w, h = (float(v) for v in t.tlwh)
            if w * h <= oc.min_box_area:
                continue
            if oc.max_aspect_ratio > 0 and h > 0 and w / h > oc.max_aspect_ratio:
                continue
            out.append(TrackOutput(t.track_id, (x, y, w, h), t.score, t.state.value, t.visibility))
        return tuple(out)

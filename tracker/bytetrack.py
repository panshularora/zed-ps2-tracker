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

Known, intentional differences from the reference implementation (documented in
``docs/decisions.md``): ``xywh`` instead of ``xyah`` Kalman state, every track (including
tentative ones) is Kalman-predicted each frame, and IDs are per-tracker-instance counters.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tracker.assignment import Assigner, HungarianAssigner, Matching
from tracker.config import TrackerConfig
from tracker.cost import cosine_cost, fuse_iou_appearance, fuse_score, iou_cost
from tracker.gallery import EmaAppearance
from tracker.kalman import KalmanFilterXYWH
from tracker.lifecycle import LifecycleConfig, Track, TrackState
from tracker.types import Detections, FloatArray, FrameResult, TrackOutput, tlwh_to_xywh
from tracker.visibility import VisibilityEstimator, build_visibility


class ByteTrackLike:
    """Online tracker. Call :meth:`update` once per frame, in order."""

    def __init__(
        self,
        config: TrackerConfig | None = None,
        frame_rate: float = 30.0,
        *,
        assigner: Assigner | None = None,
        visibility: VisibilityEstimator | None = None,
    ) -> None:
        self.cfg = config or TrackerConfig()
        if self.cfg.delayed.enabled:
            raise NotImplementedError("delayed (ambiguity-aware) assignment is planned for phase 2")
        lc = self.cfg.lifecycle
        self.lifecycle = LifecycleConfig(
            min_hits=lc.min_hits,
            tentative_max_misses=lc.tentative_max_misses,
            max_time_lost=int(
                (frame_rate / 30.0 if lc.scale_buffer_by_fps else 1.0) * lc.track_buffer
            ),
            confirm_on_first_frame=lc.confirm_on_first_frame,
        )
        self.kf = KalmanFilterXYWH(self.cfg.kalman)
        self.assign: Assigner = assigner or HungarianAssigner(self.cfg.association.backend)
        self.visibility: VisibilityEstimator = visibility or build_visibility(
            self.cfg.visibility.estimator
        )
        self.tracks: list[Track] = []
        self.frame_id = 0
        self._next_id = 1

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
        self, tracks: list[Track], dets: Detections, *, use_score: bool, use_appearance: bool
    ) -> FloatArray:
        c = iou_cost(self._boxes(tracks), dets.xyxy)
        iou_only = c
        if use_score and self.cfg.association.fuse_score:
            c = fuse_score(c, dets.scores)
        if use_appearance and tracks and len(dets) and dets.embeddings is not None:
            mem = [t.appearance for t in tracks]
            if all(m is not None for m in mem):
                track_emb = np.stack([m.embedding for m in mem if m is not None])
                app = cosine_cost(track_emb, dets.embeddings)
                ac = self.cfg.appearance
                c = fuse_iou_appearance(
                    iou_only,
                    app,
                    proximity_thresh=ac.proximity_thresh,
                    appearance_thresh=ac.appearance_thresh,
                    motion_cost=c,
                )
        return c

    def _apply_matches(
        self, tracks: list[Track], dets: Detections, vis: FloatArray, m: Matching
    ) -> None:
        use_app = self.cfg.appearance.enabled and dets.embeddings is not None
        for r, c in m.matches:
            t = tracks[int(r)]
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

        vis_all = self.visibility(dets.xyxy)
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
            means, covs = self.kf.multi_predict(means, covs)
            for i, t in enumerate(live):
                t.mean, t.covariance = means[i], covs[i]

        tentative = [t for t in live if t.state is TrackState.TENTATIVE]
        pool = [t for t in live if t.state in (TrackState.CONFIRMED, TrackState.LOST)]

        # 3. first association: confirmed + lost ↔ high
        m1 = self.assign(
            self._cost(pool, d_high, use_score=True, use_appearance=use_app),
            cfg.association.match_thresh,
        )
        self._apply_matches(pool, d_high, v_high, m1)

        # 4. second association: unmatched *confirmed* ↔ low, IoU only
        r_tracked = [pool[i] for i in m1.unmatched_rows if pool[i].state is TrackState.CONFIRMED]
        m2 = self.assign(
            self._cost(r_tracked, d_low, use_score=False, use_appearance=False),
            cfg.association.second_match_thresh,
        )
        self._apply_matches(r_tracked, d_low, v_low, m2)

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
        m3 = self.assign(
            self._cost(tentative, d_left, use_score=True, use_appearance=use_app),
            cfg.association.unconfirmed_match_thresh,
        )
        self._apply_matches(tentative, d_left, v_left, m3)
        for i in m3.unmatched_rows:
            tentative[int(i)].mark_missed(self.lifecycle)

        # 7. births
        for j in m3.unmatched_cols:
            if d_left.scores[j] < cfg.detection.new_track_thresh:
                continue
            self._birth(d_left, int(j), float(v_left[j]))

        # 8. expiry + duplicate pruning
        for t in self.tracks:
            t.expire_if_stale(self.frame_id, self.lifecycle)
        self._remove_duplicates()
        self.tracks = [t for t in self.tracks if t.state is not TrackState.REMOVED]

        return FrameResult(self.frame_id, self._report())

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
        conf = [t for t in self.tracks if t.state is TrackState.CONFIRMED]
        lost = [t for t in self.tracks if t.state is TrackState.LOST]
        if not conf or not lost:
            return
        d = iou_cost(self._boxes(conf), self._boxes(lost))
        for a, b in zip(*np.nonzero(d < self.cfg.association.duplicate_iou_dist), strict=True):
            ta, tb = conf[int(a)], lost[int(b)]
            if ta.state is TrackState.REMOVED or tb.state is TrackState.REMOVED:
                continue
            age_a = self.frame_id - ta.start_frame
            age_b = self.frame_id - tb.start_frame
            (tb if age_a > age_b else ta).remove()

    def _report(self) -> tuple[TrackOutput, ...]:
        out: list[TrackOutput] = []
        oc = self.cfg.output
        for t in self.tracks:
            if t.state is not TrackState.CONFIRMED or t.last_frame != self.frame_id:
                continue
            x, y, w, h = (float(v) for v in t.tlwh)
            if w * h <= oc.min_box_area:
                continue
            if oc.max_aspect_ratio > 0 and h > 0 and w / h > oc.max_aspect_ratio:
                continue
            out.append(TrackOutput(t.track_id, (x, y, w, h), t.score, t.state.value, t.visibility))
        return tuple(out)

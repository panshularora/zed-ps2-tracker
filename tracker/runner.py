"""Run a tracker over a whole cached sequence, timing each ``update()`` call."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from tracker.baselines.hybrid_sort import HybridSortTracker
from tracker.baselines.ocsort import OCSortTracker
from tracker.bytetrack import ByteTrackLike
from tracker.cmc import build_cmc
from tracker.config import TrackerConfig
from tracker.delayed import DelayedTracker
from tracker.io import CacheMeta, SequenceCache
from tracker.types import Detections, FrameResult
from tracker.visibility import build_visibility


@dataclass(frozen=True)
class SequenceRun:
    results: list[FrameResult]
    latency_ms: npt.NDArray[np.float64]
    """Wall-clock time of each ``update()`` call (tracker only: no I/O, no detector)."""
    n_dets: npt.NDArray[np.int64]
    info: dict[str, Any] = field(default_factory=dict)
    """Per-sequence provenance from :func:`build_tracker` (e.g. which CMC file was used)."""


class Tracker(Protocol):
    def update(self, dets: Detections) -> FrameResult | None: ...


TRACKER_NAMES = ("bytetrack_like", "bytetrack", "botsort", "ocsort", "deep_ocsort", "hybrid_sort")


def build_tracker(
    cfg: TrackerConfig, frame_rate: float, meta: CacheMeta | None = None
) -> tuple[Tracker, dict[str, Any]]:
    """Single construction point: ``cfg.name`` picks the tracker; returns it plus provenance.

    ``meta`` (sequence, split, image size) is needed for file-based CMC.
    """
    cmc, cmc_desc = build_cmc(cfg.cmc.method, meta)
    vis = build_visibility(cfg.visibility, meta)
    info: dict[str, Any] = {"cmc": cmc_desc}
    trk: Tracker
    if cfg.name in ("bytetrack_like", "bytetrack", "botsort"):
        inner = ByteTrackLike(
            cfg, frame_rate=frame_rate, cmc=cmc, visibility=vis, allow_delayed=True
        )
        if cfg.delayed.enabled:
            d = cfg.delayed
            trk = DelayedTracker(
                inner,
                max_delay_frames=d.max_delay_frames,
                margin=d.margin,
                max_cluster=d.max_cluster,
                max_hypotheses=d.max_hypotheses,
                miss_penalty=d.miss_penalty,
                min_gain=d.min_gain,
                confirmed_only=d.confirmed_only,
                same_tracks=d.same_tracks,
                score_scope=d.score_scope,
                birth_penalty=d.birth_penalty,
                trigger_max_visibility=d.trigger_max_visibility,
                trigger_require_all=d.trigger_require_all,
            )
            k_ms = round(1000.0 * d.max_delay_frames / frame_rate, 1)
            info["delayed"] = {"K_frames": d.max_delay_frames, "K_ms": k_ms}
        else:
            trk = inner
    elif cfg.delayed.enabled:
        raise ValueError(f"delayed.enabled is implemented for ByteTrackLike only, not {cfg.name}")
    elif cfg.name in ("ocsort", "deep_ocsort"):
        if cfg.name == "ocsort" and cfg.cmc.method != "none":
            raise ValueError("OC-SORT has no CMC; use deep_ocsort or cmc.method: none")
        trk = OCSortTracker(cfg, frame_rate, cmc=cmc, visibility=vis)
    elif cfg.name == "hybrid_sort":
        if cfg.cmc.method != "none":
            raise ValueError("Hybrid-SORT (IoU variant) has no CMC; set cmc.method: none")
        trk = HybridSortTracker(cfg, frame_rate, visibility=vis)
    else:
        raise ValueError(f"unknown tracker {cfg.name!r}; known: {TRACKER_NAMES}")
    return trk, info


def run_sequence(cache: SequenceCache, cfg: TrackerConfig) -> SequenceRun:
    trk, info = build_tracker(cfg, cache.meta.frame_rate, cache.meta)
    if not cfg.appearance.enabled:
        cache = SequenceCache(cache.meta, cache.frame, cache.tlwh, cache.score, None)
    results: list[FrameResult] = []
    lat, nd = [], []
    for _, dets in cache.frames():
        t0 = time.perf_counter_ns()
        res = trk.update(dets)
        lat.append((time.perf_counter_ns() - t0) / 1e6)
        if res is not None:  # a delayed tracker returns None while its buffer fills
            results.append(res)
        nd.append(len(dets))
    if isinstance(trk, DelayedTracker):
        t0 = time.perf_counter_ns()
        results.extend(trk.flush())
        info["delayed"]["flush_ms"] = (time.perf_counter_ns() - t0) / 1e6
        st = trk.stats
        info["delayed"].update({
            "frames": st.frames, "ambiguous_frames": st.ambiguous_frames,
            "clusters": st.clusters, "changed": st.changed, "skipped_large": st.skipped_large,
            "rollouts": st.rollouts, "trigger_frames": st.trigger_frames,
            "changed_frames": st.changed_frames,
        })  # fmt: skip
    if len(results) != len(nd) or any(r.frame_id != i + 1 for i, r in enumerate(results)):
        raise RuntimeError("tracker did not emit exactly one result per frame, in order")
    return SequenceRun(results, np.asarray(lat, np.float64), np.asarray(nd, np.int64), info)

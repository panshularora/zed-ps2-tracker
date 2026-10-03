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
from tracker.io import CacheMeta, SequenceCache
from tracker.types import Detections, FrameResult


@dataclass(frozen=True)
class SequenceRun:
    results: list[FrameResult]
    latency_ms: npt.NDArray[np.float64]
    """Wall-clock time of each ``update()`` call (tracker only: no I/O, no detector)."""
    n_dets: npt.NDArray[np.int64]
    info: dict[str, Any] = field(default_factory=dict)
    """Per-sequence provenance from :func:`build_tracker` (e.g. which CMC file was used)."""


class Tracker(Protocol):
    def update(self, dets: Detections) -> FrameResult: ...


TRACKER_NAMES = ("bytetrack_like", "bytetrack", "botsort", "ocsort", "deep_ocsort", "hybrid_sort")


def build_tracker(
    cfg: TrackerConfig, frame_rate: float, meta: CacheMeta | None = None
) -> tuple[Tracker, dict[str, Any]]:
    """Single construction point: ``cfg.name`` picks the tracker; returns it plus provenance.

    ``meta`` (sequence, split, image size) is needed for file-based CMC.
    """
    cmc, cmc_desc = build_cmc(cfg.cmc.method, meta)
    info: dict[str, Any] = {"cmc": cmc_desc}
    trk: Tracker
    if cfg.name in ("bytetrack_like", "bytetrack", "botsort"):
        trk = ByteTrackLike(cfg, frame_rate=frame_rate, cmc=cmc)
    elif cfg.name in ("ocsort", "deep_ocsort"):
        if cfg.name == "ocsort" and cfg.cmc.method != "none":
            raise ValueError("OC-SORT has no CMC; use deep_ocsort or cmc.method: none")
        trk = OCSortTracker(cfg, frame_rate, cmc=cmc)
    elif cfg.name == "hybrid_sort":
        if cfg.cmc.method != "none":
            raise ValueError("Hybrid-SORT (IoU variant) has no CMC; set cmc.method: none")
        trk = HybridSortTracker(cfg, frame_rate)
    else:
        raise ValueError(f"unknown tracker {cfg.name!r}; known: {TRACKER_NAMES}")
    return trk, info


def run_sequence(cache: SequenceCache, cfg: TrackerConfig) -> SequenceRun:
    trk, info = build_tracker(cfg, cache.meta.frame_rate, cache.meta)
    if not cfg.appearance.enabled:
        cache = SequenceCache(cache.meta, cache.frame, cache.tlwh, cache.score, None)
    results, lat, nd = [], [], []
    for _, dets in cache.frames():
        t0 = time.perf_counter_ns()
        res = trk.update(dets)
        lat.append((time.perf_counter_ns() - t0) / 1e6)
        results.append(res)
        nd.append(len(dets))
    return SequenceRun(results, np.asarray(lat, np.float64), np.asarray(nd, np.int64), info)

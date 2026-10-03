"""Run a tracker over a whole cached sequence, timing each ``update()`` call."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tracker.bytetrack import ByteTrackLike
from tracker.config import TrackerConfig
from tracker.io import SequenceCache
from tracker.types import FrameResult


@dataclass(frozen=True)
class SequenceRun:
    results: list[FrameResult]
    latency_ms: npt.NDArray[np.float64]
    """Wall-clock time of each ``update()`` call (tracker only: no I/O, no detector)."""
    n_dets: npt.NDArray[np.int64]


def build_tracker(cfg: TrackerConfig, frame_rate: float) -> ByteTrackLike:
    """Single construction point, so new tracker variants slot in behind ``cfg.name``."""
    if cfg.name in ("bytetrack_like", "bytetrack"):
        return ByteTrackLike(cfg, frame_rate=frame_rate)
    raise ValueError(f"unknown tracker {cfg.name!r}")


def run_sequence(cache: SequenceCache, cfg: TrackerConfig) -> SequenceRun:
    trk = build_tracker(cfg, cache.meta.frame_rate)
    if not cfg.appearance.enabled:
        cache = SequenceCache(cache.meta, cache.frame, cache.tlwh, cache.score, None)
    results, lat, nd = [], [], []
    for _, dets in cache.frames():
        t0 = time.perf_counter_ns()
        res = trk.update(dets)
        lat.append((time.perf_counter_ns() - t0) / 1e6)
        results.append(res)
        nd.append(len(dets))
    return SequenceRun(results, np.asarray(lat, np.float64), np.asarray(nd, np.int64))

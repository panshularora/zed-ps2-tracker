"""Video → detector → tracker → events, on CPU (the demo service's job body).

``run_video(path, options, detector, progress)`` decodes the clip with OpenCV, runs the person
detector on every ``stride``-th frame (up to ``max_frames`` tracked frames), builds a Stage A
style :class:`tracker.io.SequenceCache` from the boxes (no embeddings: re-ID needs the GPU), runs
a tracker config from ``configs/`` through :func:`tracker.runner.run_sequence`, and derives the
gate events, recoveries, heuristic ID-switch flags and per-job metrics (``serve/analysis.py``).

Frame numbering in the result: tracked frame ``k`` (1-based) is video frame ``(k - 1) * stride``
(0-based), shown at ``t = (k - 1) * stride / fps``. The tracker runs at ``fps / stride``.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from eval.gate_events import Gate
from serve.analysis import AnalysisParams, gate_output, metrics, recoveries, suspected_switches
from serve.events import track_rows
from tracker.config import TrackerConfig
from tracker.io import CacheMeta, SequenceCache
from tracker.runner import run_sequence

RESULT_SCHEMA = "zed-ps2-job-result/v1"
Progress = Callable[[str, int, int], None]
"""``progress(stage, done, total)``; may raise to cancel the job."""


class Detector(Protocol):
    def detect(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps > 0 else 0.0


def probe(path: Path) -> VideoInfo:
    """Size, fps and frame count of a video, or ``ValueError`` if OpenCV cannot decode it."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("not a decodable video")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        if not ok or w <= 0 or h <= 0:
            raise ValueError("video has no decodable frames")
        if not (0 < fps <= 240):
            raise ValueError(f"unsupported frame rate {fps}")
        return VideoInfo(w, h, fps, max(n, 1))
    finally:
        cap.release()


@dataclass(frozen=True)
class JobOptions:
    config: str = "ours/visibility_botsort"
    detector: str = "yolox_s"
    stride: int = 1
    max_frames: int | None = None
    gate: tuple[float, float, float, float] | None = None
    """Normalised ``x1, y1, x2, y2`` (0..1); default: horizontal line at 60 % height."""
    event_styling: str = "neutral"
    hysteresis_px: float = 10.0
    start_time: str = "1970-01-01T00:00:00+00:00"


def detect_video(
    path: Path,
    detector: Detector,
    stride: int,
    max_frames: int | None,
    progress: Progress | None = None,
) -> tuple[SequenceCache, dict[str, Any]]:
    """Run the detector on every ``stride``-th frame; returns the cache and timing."""
    import cv2

    info = probe(path)
    total_tracked = (info.frames + stride - 1) // stride
    if max_frames:
        total_tracked = min(total_tracked, max_frames)
    cap = cv2.VideoCapture(str(path))
    fr, tl, sc, det_ms = [], [], [], []
    k, idx, t_dec = 0, 0, 0.0
    try:
        while k < total_tracked:
            t0 = time.perf_counter()
            ok = cap.grab()
            if not ok:
                break
            if idx % stride:
                idx += 1
                t_dec += time.perf_counter() - t0
                continue
            ok, img = cap.retrieve()
            t_dec += time.perf_counter() - t0
            if not ok:
                break
            idx += 1
            k += 1
            t1 = time.perf_counter()
            boxes, scores = detector.detect(img)
            det_ms.append((time.perf_counter() - t1) * 1e3)
            fr.extend([k] * len(scores))
            tl.append(np.asarray(boxes, np.float32).reshape(-1, 4))
            sc.append(np.asarray(scores, np.float32).reshape(-1))
            if progress and (k % 5 == 0 or k == total_tracked):
                progress("detect", k, total_tracked)
    finally:
        cap.release()
    if k == 0:
        raise ValueError("no frames decoded")
    meta = CacheMeta(
        sequence=path.stem,
        dataset="demo",
        split="full",
        frame_rate=info.fps / stride,
        seq_length=k,
        im_width=info.width,
        im_height=info.height,
        detector={"name": getattr(detector, "name", type(detector).__name__)},
    )
    cache = SequenceCache(
        meta,
        np.asarray(fr, np.int32),
        np.concatenate(tl) if tl else np.zeros((0, 4), np.float32),
        np.concatenate(sc) if sc else np.zeros(0, np.float32),
    )
    # the raw detector output of the first frames, for the architecture page's example payload
    samples = [{"frame": i + 1, "tlwh": np.round(t, 1).tolist(), "score": np.round(c, 3).tolist()}
               for i, (t, c) in enumerate(zip(tl[:3], sc[:3], strict=True))]  # fmt: skip
    d = np.asarray(det_ms, float)
    timing: dict[str, Any] = {
        "decode_s": round(t_dec, 2),
        "detect_s": round(float(d.sum()) / 1e3, 2),
        "detector_p50_ms": round(float(np.percentile(d, 50)), 1),
        "detector_p95_ms": round(float(np.percentile(d, 95)), 1),
        "sample_detections": samples,
    }
    return cache, timing


def gate_from_norm(norm: tuple[float, float, float, float] | None, w: int, h: int) -> Gate:
    if norm is None:
        return Gate("h60", (0.0, 0.6 * h), (float(w), 0.6 * h))
    x1, y1, x2, y2 = norm
    if (x1, y1) == (x2, y2):
        raise ValueError("gate has zero length")
    return Gate("gate", (x1 * w, y1 * h), (x2 * w, y2 * h))


def track_and_analyse(
    cache: SequenceCache, cfg: TrackerConfig, opts: JobOptions, stride: int, src_fps: float
) -> dict[str, Any]:
    """Tracker + events + metrics on a detection cache (no video needed: unit-testable)."""
    m = cache.meta
    run = run_sequence(cache, cfg)
    rows = track_rows(run.results)
    gate = gate_from_norm(opts.gate, m.im_width, m.im_height)
    start = datetime.fromisoformat(opts.start_time)
    gev = gate_output(
        run.results, m.frame_rate, gate, start, opts.event_styling, opts.hysteresis_px
    )
    p = AnalysisParams()
    rec = recoveries(rows, m.frame_rate, p)
    sus = suspected_switches(rows, m.frame_rate, m.seq_length, p)
    tracks: dict[str, list[list[float]]] = {}
    for r in rows:
        tracks.setdefault(str(int(r[1])), []).append(
            [
                int(r[0]),
                *(round(float(v), 1) for v in r[2:6]),
                round(float(r[6]), 3),
                round(float(r[7]), 3),
            ]
        )
    events = sorted(
        [{**e, "kind": "gate"} for e in gev]
        + [{**e, "kind": "recovery"} for e in rec]
        + [{**e, "kind": "identity"} for e in sus],
        key=lambda e: (e["frame"], e["type"]),
    )
    lat = np.asarray(run.latency_ms, float)
    return {
        "frames_tracked": m.seq_length,
        "fps_tracked": m.frame_rate,
        "gate": {
            "name": gate.name,
            "a": list(gate.a),
            "b": list(gate.b),
            "norm": list(opts.gate) if opts.gate else [0.0, 0.6, 1.0, 0.6],
            "hysteresis_px": opts.hysteresis_px,
        },
        "tracks": tracks,
        "events": events,
        "metrics": metrics(rows, m.frame_rate, m.seq_length, gev, rec, sus, p),
        "tracker_timing": {
            "tracker_p50_ms": round(float(np.percentile(lat, 50)), 3),
            "tracker_p95_ms": round(float(np.percentile(lat, 95)), 3),
            "tracker_total_s": round(float(lat.sum()) / 1e3, 3),
        },
        "tracker_info": run.info,
        "analysis_params": asdict(p),
        "mot_rows": rows,
    }


def run_video(
    path: Path,
    opts: JobOptions,
    cfg: TrackerConfig,
    detector: Detector,
    progress: Progress | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    info = probe(path)
    if progress:
        progress("detect", 0, 1)
    cache, det_timing = detect_video(path, detector, opts.stride, opts.max_frames, progress)
    if progress:
        progress("track", 0, 1)
    out = track_and_analyse(cache, cfg, opts, opts.stride, info.fps)
    if progress:
        progress("track", 1, 1)
    wall = time.perf_counter() - t0
    out["video"] = {
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "frames": info.frames,
        "duration_s": round(info.duration_s, 3),
        "stride": opts.stride,
    }
    out["sample_detections"] = det_timing.pop("sample_detections")
    out["timing"] = {
        **det_timing,
        **out.pop("tracker_timing"),
        "wall_s": round(wall, 2),
        "per_frame_ms": round(1e3 * wall / max(out["frames_tracked"], 1), 1),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
    }
    out["schema"] = RESULT_SCHEMA
    return out

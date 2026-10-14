"""Synthetic gate scene: a seeded top-down simulator, a fake detector, and the REAL trackers.

People (exact ground truth) walk through a 800x450 floor and cross a horizontal gate line. A fake
detector turns them into boxes with jitter, drops while occluded, merged boxes when two people
overlap and occasional false positives. The SAME Detections object is fed to every tracker each
frame (``tracker.runner.build_tracker`` with the repo configs), so differences come from the
trackers alone. Everything is driven by one seed: the same parameters replay identically.

Streamed as NDJSON by ``GET /api/gate-sim/stream`` (one JSON line per frame).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from scipy.optimize import linear_sum_assignment

from tracker.config import load_config
from tracker.runner import build_tracker
from tracker.types import Detections

ROOT = Path(__file__).resolve().parents[1]
W, H = 800, 450
GATE_Y = 225.0
BOX = 24.0  # true footprint box side (px) the detector reports around each person
MAX_PEOPLE = 40
TRACKERS = {
    "bytetrack": ("ByteTrack", "configs/baselines/bytetrack.yaml"),
    "botsort": ("BoT-SORT", "configs/baselines/botsort.yaml"),
    "ours": ("BoT-SORT + visibility (ours)", "configs/ours/visibility_botsort.yaml"),
}


@dataclass(frozen=True)
class SimParams:
    seed: int = 7
    density: float = 0.5  # 0..1 -> arrivals per second (0.3 .. 2.3), capped at MAX_PEOPLE alive
    occlusion_s: float = 1.0  # how long one occlusion lasts
    noise: float = 0.3  # 0..1 detector noise (jitter, misses, false positives, merges)
    speed: float = 1.0  # walking speed multiplier
    seconds: float = 60.0
    fps: float = 30.0

    def clamped(self) -> SimParams:
        def c(v: float, lo: float, hi: float) -> float:
            return float(min(hi, max(lo, v)))

        return SimParams(
            seed=self.seed & 0x7FFFFFFF,
            density=c(self.density, 0, 1),
            occlusion_s=c(self.occlusion_s, 0, 4),
            noise=c(self.noise, 0, 1),
            speed=c(self.speed, 0.25, 2),
            seconds=c(self.seconds, 1, 180),
            fps=c(self.fps, 10, 30),
        )


@dataclass
class Person:
    pid: int
    x: float
    y: float
    vx: float
    vy: float
    wobble: float
    phase: float
    occl_at: float | None  # y at which an occlusion starts (None: never)
    occl_left: float = 0.0
    occl_done: bool = False


@dataclass
class World:
    p: SimParams
    rng: np.random.Generator
    people: list[Person] = field(default_factory=list)
    next_pid: int = 1
    t: float = 0.0
    spawn_acc: float = 0.0

    def spawn(self) -> None:
        down = bool(self.rng.random() < 0.5)
        v = (45 + 30 * self.rng.random()) * self.p.speed
        x = 60 + (W - 120) * self.rng.random()
        occl = None
        if self.p.occlusion_s > 0 and self.rng.random() < 0.6:
            occl = GATE_Y + (self.rng.random() - 0.5) * 140
        self.people.append(
            Person(
                self.next_pid,
                x,
                -12.0 if down else H + 12.0,
                (self.rng.random() - 0.5) * 12,
                v if down else -v,
                6 + 10 * self.rng.random(),
                2 * math.pi * self.rng.random(),
                occl,
            )
        )
        self.next_pid += 1

    def step(self, dt: float) -> None:
        self.t += dt
        self.spawn_acc += dt * (0.3 + 2.0 * self.p.density)
        while self.spawn_acc >= 1:
            self.spawn_acc -= 1
            if len(self.people) < MAX_PEOPLE:
                self.spawn()
        for q in self.people:
            q.phase += dt * 1.7
            q.x += (q.vx + q.wobble * math.cos(q.phase)) * dt
            q.x = min(W - 12.0, max(12.0, q.x))
            prev_y = q.y
            q.y += q.vy * dt
            if q.occl_left > 0:
                q.occl_left = max(0.0, q.occl_left - dt)
            elif (
                q.occl_at is not None
                and not q.occl_done
                and (prev_y - q.occl_at) * (q.y - q.occl_at) <= 0
            ):
                q.occl_left, q.occl_done = self.p.occlusion_s, True
        self.people = [q for q in self.people if -20 <= q.y <= H + 20]


def detect(
    people: list[Person], p: SimParams, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Fake detector: tlwh (N, 4) and scores (N,). Deterministic given the rng state."""
    n = p.noise
    boxes: list[list[float]] = []
    scores: list[float] = []
    vis = [q for q in people if 0 <= q.y <= H]
    used: set[int] = set()
    for i, a in enumerate(vis):
        if i in used:
            continue
        if a.occl_left > 0:  # occluded: usually dropped, sometimes a weak partial box
            if rng.random() < 0.25:
                boxes.append([a.x - BOX / 2, a.y - BOX / 2 + BOX * 0.3, BOX, BOX * 0.7])
                scores.append(0.15 + 0.25 * rng.random())
            continue
        partner = None
        for j in range(i + 1, len(vis)):
            b = vis[j]
            if j not in used and b.occl_left <= 0 and math.hypot(a.x - b.x, a.y - b.y) < 16:
                partner = j
                break
        if partner is not None and rng.random() < 0.35 + 0.5 * n:
            b = vis[partner]
            used.add(partner)
            x0, y0 = min(a.x, b.x) - BOX / 2, min(a.y, b.y) - BOX / 2
            x1, y1 = max(a.x, b.x) + BOX / 2, max(a.y, b.y) + BOX / 2
            boxes.append([x0, y0, x1 - x0, y1 - y0])  # one merged box for two people
            scores.append(0.5 + 0.4 * rng.random())
            continue
        if rng.random() < 0.02 + 0.12 * n:  # plain miss
            continue
        jit = 0.5 + 3.5 * n
        s = BOX * (1 + 0.08 * n * rng.standard_normal())
        boxes.append(
            [
                a.x - s / 2 + jit * rng.standard_normal(),
                a.y - s / 2 + jit * rng.standard_normal(),
                s,
                s,
            ]
        )
        crowded = partner is not None
        scores.append((0.3 + 0.3 * rng.random()) if crowded else (0.65 + 0.3 * rng.random()))
    for _ in range(int(rng.poisson(0.03 + 0.3 * n))):  # false positives
        boxes.append([40 + (W - 80) * rng.random(), 20 + (H - 40) * rng.random(), BOX, BOX])
        scores.append(0.1 + 0.6 * rng.random())
    return np.asarray(boxes, np.float64).reshape(-1, 4), np.asarray(scores, np.float64)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = (
        a[:, 0, None],
        a[:, 1, None],
        (a[:, 0] + a[:, 2])[:, None],
        (a[:, 1] + a[:, 3])[:, None],
    )
    bx1, by1, bx2, by2 = (
        b[None, :, 0],
        b[None, :, 1],
        (b[:, 0] + b[:, 2])[None],
        (b[:, 1] + b[:, 3])[None],
    )
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    out: np.ndarray = inter / (a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None] - inter)
    return out


@dataclass
class Scoring:
    """Per tracker: truth->track matching, ID changes and gate counts."""

    last_id: dict[int, int] = field(default_factory=dict)
    side: dict[int, float] = field(default_factory=dict)
    counted: int = 0
    id_changes: int = 0

    def update(
        self, truth: list[Person], tracks: list[tuple[int, np.ndarray]]
    ) -> tuple[dict[int, int], list[list[float]]]:
        events: list[list[float]] = []
        match: dict[int, int] = {}
        if truth and tracks:
            tb = np.asarray([[q.x - BOX / 2, q.y - BOX / 2, BOX, BOX] for q in truth])
            kb = np.asarray([b for _, b in tracks])
            iou = _iou(tb, kb)
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c, strict=True):
                if iou[i, j] >= 0.3:
                    match[truth[i].pid] = tracks[j][0]
        for q in truth:
            tid = match.get(q.pid)
            if tid is None:
                continue
            prev = self.last_id.get(q.pid)
            if prev is not None and prev != tid:
                self.id_changes += 1
                events.append([q.pid, prev, tid, round(q.x, 1), round(q.y, 1)])
            self.last_id[q.pid] = tid
        for tid, b in tracks:  # a track id counts when its centre crosses the line
            cy = b[1] + b[3] / 2
            s = self.side.get(tid)
            if s is not None and (s - GATE_Y) * (cy - GATE_Y) < 0:
                self.counted += 1
            self.side[tid] = cy
        return match, events


def simulate(
    params: SimParams, trackers: tuple[str, ...] = ("bytetrack", "ours")
) -> Iterator[dict[str, Any]]:
    p = params.clamped()
    world = World(p, np.random.default_rng([p.seed, 1]))
    det_rng = np.random.default_rng([p.seed, 2])
    trk = {k: build_tracker(load_config(ROOT / TRACKERS[k][1]), p.fps)[0] for k in trackers}
    score = {k: Scoring() for k in trackers}
    true_side: dict[int, float] = {}
    true_count = 0
    yield {
        "type": "meta",
        "width": W,
        "height": H,
        "gate": [[0, GATE_Y], [W, GATE_Y]],
        "fps": p.fps,
        "params": asdict(p),
        "trackers": {k: {"label": TRACKERS[k][0], "config": TRACKERS[k][1]} for k in trackers},
        "max_people": MAX_PEOPLE,
    }
    n = round(p.seconds * p.fps)
    for f in range(1, n + 1):
        world.step(1.0 / p.fps)
        tlwh, sc = detect(world.people, p, det_rng)
        dets = Detections(tlwh, sc)
        on = [q for q in world.people if 0 <= q.y <= H]
        for q in on:
            s = true_side.get(q.pid)
            if s is not None and (s - GATE_Y) * (q.y - GATE_Y) < 0:
                true_count += 1
            true_side[q.pid] = q.y
        frame: dict[str, Any] = {
            "type": "frame",
            "f": f,
            "people": [
                [q.pid, round(q.x, 1), round(q.y, 1), 0 if q.occl_left > 0 else 1] for q in on
            ],
            "dets": [
                [round(float(v), 1) for v in b] + [round(float(s), 2)]
                for b, s in zip(tlwh, sc, strict=True)
            ],
            "true_count": true_count,
        }
        for k, t in trk.items():
            res = t.update(dets)
            tracks = [(o.track_id, np.asarray(o.tlwh)) for o in (res.tracks if res else ())]
            match, ev = score[k].update(on, tracks)
            matched = set(match.values())
            frame[k] = {
                "ids": [[pid, tid] for pid, tid in match.items()],
                "extra": [
                    [tid, round(b[0] + b[2] / 2, 1), round(b[1] + b[3] / 2, 1)]
                    for tid, b in tracks
                    if tid not in matched
                ],
                "events": ev,
                "counted": score[k].counted,
                "id_changes": score[k].id_changes,
            }
        yield frame
    yield {
        "type": "end",
        "frames": n,
        "true_count": true_count,
        "summary": {
            k: {"id_changes": score[k].id_changes, "counted": score[k].counted, "truth": true_count}
            for k in trackers
        },
    }


router = APIRouter()


@router.get("/api/gate-sim/stream")
def gate_sim_stream(
    seed: int = Query(7),
    density: float = Query(0.5),
    occlusion: float = Query(1.0),
    noise: float = Query(0.3),
    speed: float = Query(1.0),
    seconds: float = Query(60.0),
) -> StreamingResponse:
    params = SimParams(
        seed=seed, density=density, occlusion_s=occlusion, noise=noise, speed=speed, seconds=seconds
    )
    lines = (json.dumps(x, separators=(",", ":")) + "\n" for x in simulate(params))
    return StreamingResponse(
        lines, media_type="application/x-ndjson", headers={"Cache-Control": "no-store"}
    )

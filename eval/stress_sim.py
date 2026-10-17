"""Seeded stress scenarios for the one-person-one-ID question (CPU, offline, exact ground truth).

A camera-view gate scene (``W x H`` pixels, gate = horizontal line at ``GATE_Y``). Every person is a
foot point ``(x, y)`` with a person-shaped box (height grows towards the camera, width 0.41 h), so
the depth-order visibility estimate of ``configs/ours`` sees the same geometry it sees on MOT17
(the box with the lower bottom edge is in front).

Two kinds of occlusion, both producing the detector behaviour seen on MOT17 (weak, truncated boxes
while partly hidden, nothing while hidden):

* **people in front**: the true visible fraction of each box is computed from the boxes closer to
  the camera (same rule as ``tracker.visibility.DepthOrderVisibility``, on true boxes);
* **scripted hides**: an occluder the person detector cannot see (a pillar, a taller person, a
  kiosk) hides a person for exactly ``hide_s`` seconds, with a short ramp in and out
  (``RAMP_F`` frames) during which the box is partly visible.

The fake detector turns the true visible fraction into boxes: below ``VIS_DROP`` the box is
dropped, between ``VIS_DROP`` and ``VIS_WEAK`` it is a low-score box (amodal, or with
probability ``TRUNCATE_P`` cut to the visible top part, as person detectors do), above it a normal
box with jitter. ``noise`` scales jitter, plain misses and false positives.

Walking motion is not a straight line: each person's velocity follows an Ornstein-Uhlenbeck
process around their goal velocity, so a long hide really tests motion prediction.

Everything is a pure function of ``(Scenario, seed)``: :func:`render` returns per-frame truth and
detections, replayable bit-for-bit by the web app later.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

W, H = 960, 540
GATE_Y = 300.0
FPS = 30.0
RAMP_F = 6  # frames of partial visibility before / after a scripted hide
VIS_DROP = 0.25  # true visible fraction below which the detector returns nothing
VIS_WEAK = 0.6  # below this a detected box is low-score (and sometimes truncated)
TRUNCATE_P = 0.4  # share of partly hidden boxes that cover only the visible top part


def box_h(y: float) -> float:
    """Person height in pixels at foot row ``y`` (simple ground-plane perspective)."""
    return 46.0 + 0.16 * max(0.0, y)


def tlwh_of(x: float, y: float) -> tuple[float, float, float, float]:
    h = box_h(y)
    w = 0.41 * h
    return (x - w / 2, y - h, w, h)


@dataclass(frozen=True)
class Scenario:
    """One stress condition. ``kind`` selects the choreography; the rest are its parameters."""

    kind: str = "grid"
    people: int = 10
    hide_s: float = 1.0
    noise: float = 0.15
    seconds: float = 26.0
    arrive_s: float = 12.0
    speed: float = 1.0
    jitter_px: float = 0.0  # camera shake amplitude (all boxes move together)
    dropout_f: int = 0  # whole-frame-free detector dropout of one person (frames), no occluder
    label: str = ""

    def key(self) -> str:
        return self.label or f"{self.kind}-n{self.people}-h{self.hide_s:g}-z{self.noise:g}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Walker:
    pid: int
    x: float
    y: float
    vx: float
    vy: float
    gvx: float  # goal velocity
    gvy: float
    t0: float  # arrival time (s)
    hide_at: float | None = None  # foot y at which the scripted hide starts
    hide_left_f: int = 0
    hide_total_f: int = 0
    hide_done: bool = False
    hide_frames: list[int] = field(default_factory=list)
    stops: list[tuple[float, float]] = field(default_factory=list)  # (start_s, dur_s) stand still
    turn_at: float | None = None  # foot y at which the walker turns back (U-turn / step back)
    turned: bool = False
    active: bool = False
    gone: bool = False
    path: list[tuple[float, float]] = field(default_factory=list)  # waypoints (Break it yourself)
    speed_px: float = 0.0  # walking speed along ``path`` (px/s)
    hide_at_arc: float | None = None  # path walkers: arc length (px) at which the hide starts
    arc: float = 0.0


@dataclass(frozen=True)
class Frame:
    f: int
    truth: list[tuple[int, float, float, float, float, float]]
    """``(pid, x, y, w, h, visible_fraction)`` of every person on screen (tlwh box)."""
    dets: np.ndarray  # (N, 4) tlwh
    scores: np.ndarray  # (N,)
    det_pid: list[int]  # which person produced each detection (-1 = false positive)
    hidden: list[int]  # pids inside a scripted hide this frame


def _walker(rng: np.random.Generator, pid: int, t0: float, sc: Scenario) -> Walker:
    down = bool(rng.random() < 0.5)
    v = (50 + 35 * rng.random()) * sc.speed
    x = 80 + (W - 160) * rng.random()
    y = -4.0 if down else H + box_h(H) * 0.0 + 4.0
    gvx = (rng.random() - 0.5) * 30
    gvy = v if down else -v
    return Walker(pid, x, y, gvx, gvy, gvx, gvy, t0)


def build_walkers(sc: Scenario, rng: np.random.Generator) -> list[Walker]:
    """Choreography for ``sc.kind``; every walker gets a scripted hide of ``sc.hide_s`` near the
    gate unless the kind says otherwise."""
    ws: list[Walker] = []
    n = max(1, sc.people)
    kind = sc.kind

    def hide_band(w: Walker | None = None) -> float:
        """Foot row where the hide starts: the hide is centred on the gate (±40 px), so the
        person is hidden while crossing and re-emerges on screen even for a 5 s hide."""
        centre = GATE_Y + (rng.random() - 0.5) * 80
        if w is None:
            return centre
        span = abs(w.gvy) * (sc.hide_s + 2 * RAMP_F / FPS)
        sgn = 1.0 if w.gvy > 0 else -1.0
        start = centre - sgn * span / 2
        # keep the start on screen (the person must be seen before the hide)
        return (
            float(min(max(start, 40.0), H - 40.0))
            if sgn > 0
            else float(max(min(start, H - 40.0), 40.0))
        )

    if kind in ("grid", "fast_walker", "camera_shake", "density"):
        for i in range(n):
            w = _walker(rng, i + 1, sc.arrive_s * rng.random(), sc)
            if kind == "fast_walker" and i == 0:
                w.gvy *= 2.6
                w.vy = w.gvy
            if sc.hide_s > 0:
                w.hide_at = hide_band(w)
            ws.append(w)
    elif kind == "lookalike_cross":
        # pairs of same-sized people on crossing diagonals, meeting at the gate; the one behind is
        # hidden by the one in front while they overlap (natural occlusion) plus the scripted hide
        for i in range(0, n, 2):
            t0 = sc.arrive_s * rng.random()
            xm = 240 + (W - 480) * rng.random()
            v = (55 + 25 * rng.random()) * sc.speed
            ty = (GATE_Y + 4) / v  # seconds to reach the gate from the top edge
            for k, sgn in enumerate((1, -1)):
                vx = sgn * 70 * sc.speed
                w = Walker(i + k + 1, xm - vx * ty, -4.0 + 8.0 * k, vx, v, vx, v, t0)
                if k == 1 and sc.hide_s > 0:
                    w.hide_at = GATE_Y - 10
                ws.append(w)
    elif kind == "group":
        # groups of 3-4 walking side by side, ~0.6 box widths apart
        pid = 1
        while pid <= n:
            g = min(int(rng.integers(3, 5)), n - pid + 1)
            base = _walker(rng, pid, sc.arrive_s * rng.random(), sc)
            for k in range(g):
                w = replace(
                    base,
                    pid=pid,
                    x=base.x + (k - g / 2) * 22 + rng.normal(0, 2),
                    y=base.y - (4 if base.gvy > 0 else -4) * k,
                    hide_frames=[],
                    stops=[],
                )
                if k == 1 and sc.hide_s > 0:
                    w.hide_at = GATE_Y - 20 if w.gvy > 0 else GATE_Y + 20
                ws.append(w)
                pid += 1
    elif kind == "stop_and_go":
        for i in range(n):
            w = _walker(rng, i + 1, sc.arrive_s * rng.random(), sc)
            # stop at the gate (time to reach it + 0..0.3 s) for 1.5-3 s
            t_gate = abs((GATE_Y - w.y) / w.gvy)
            w.stops = [(w.t0 + t_gate - 0.4 + 0.3 * rng.random(), 1.5 + 1.5 * rng.random())]
            if sc.hide_s > 0:
                w.hide_at = hide_band(w)
            ws.append(w)
    elif kind == "step_back":
        for i in range(n):
            w = _walker(rng, i + 1, sc.arrive_s * rng.random(), sc)
            if i % 2 == 0:  # half of them cross the line, then step back over it
                w.turn_at = GATE_Y + (35 if w.gvy > 0 else -35)
            if sc.hide_s > 0:
                w.hide_at = hide_band(w)
            ws.append(w)
    elif kind == "linger":
        for i in range(n):
            w = _walker(rng, i + 1, sc.arrive_s * rng.random(), sc)
            if i % 3 == 0:
                t_gate = abs((GATE_Y - 30 - w.y) / w.gvy)
                w.stops = [(w.t0 + t_gate, 4.0 + 3.0 * rng.random())]
            if sc.hide_s > 0:
                w.hide_at = hide_band(w)
            ws.append(w)
    elif kind == "dropout":
        # no occluder at all: the detector simply misses one person for dropout_f frames
        for i in range(n):
            w = _walker(rng, i + 1, sc.arrive_s * rng.random(), sc)
            w.hide_at = hide_band(w)
            ws.append(w)
    else:
        raise ValueError(f"unknown scenario kind {kind!r}")
    return ws


def _visible_fraction(boxes: np.ndarray, grid: int = 12) -> np.ndarray:
    """True visible fraction of every tlwh box given the boxes in front (larger bottom edge)."""
    n = len(boxes)
    if n == 0:
        return np.zeros(0)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    fr = (np.arange(grid) + 0.5) / grid
    xs = x1[:, None] + fr[None] * boxes[:, 2:3]
    ys = y1[:, None] + fr[None] * boxes[:, 3:4]
    closer = y2[None, :] > y2[:, None]
    np.fill_diagonal(closer, False)
    inx = (xs[:, None, :] >= x1[None, :, None]) & (xs[:, None, :] < x2[None, :, None])
    iny = (ys[:, None, :] >= y1[None, :, None]) & (ys[:, None, :] < y2[None, :, None])
    inx &= closer[:, :, None]
    cnt = np.einsum("ija,ijb->iab", inx.astype(np.float32), iny.astype(np.float32))
    vis = 1.0 - (cnt > 0.5).reshape(n, -1).mean(axis=1)
    # image border
    cx = np.clip(np.minimum(x2, W) - np.maximum(x1, 0), 0, None) / np.maximum(boxes[:, 2], 1e-6)
    cy = np.clip(np.minimum(y2, H) - np.maximum(y1, 0), 0, None) / np.maximum(boxes[:, 3], 1e-6)
    return np.asarray(vis * cx * cy, np.float64)


def pillar_cover(
    boxes: np.ndarray,
    foot_y: np.ndarray,
    pillars: list[tuple[float, float, float, float]],
    grid: int = 12,
) -> np.ndarray:
    """Fraction of each person box hidden by pillars (tlwh) standing in front of them (pillar base
    ``y + h`` below the person's foot row)."""
    n = len(boxes)
    hid = np.zeros(n)
    if n == 0 or not pillars:
        return hid
    fr = (np.arange(grid) + 0.5) / grid
    xs = boxes[:, 0:1] + fr[None] * boxes[:, 2:3]
    ys = boxes[:, 1:2] + fr[None] * boxes[:, 3:4]
    cov = np.zeros((n, grid, grid), bool)
    for px, py, pw, ph in pillars:
        front = (py + ph) > foot_y
        inx = (xs >= px) & (xs < px + pw)
        iny = (ys >= py) & (ys < py + ph)
        cov |= front[:, None, None] & iny[:, :, None] & inx[:, None, :]
    return np.asarray(cov.reshape(n, -1).mean(axis=1), np.float64)


def path_walker(
    pid: int, path: list[tuple[float, float]], t0: float, speed_px: float, hide_s: float | None
) -> Walker:
    """A person walking a drawn path. If ``hide_s`` is set, the scripted hide is centred on the
    point of the path closest to the gate line (or its middle if it never comes near)."""
    pts = [(float(x), float(y)) for x, y in path]
    if len(pts) < 2:
        raise ValueError("a path needs at least two points")
    seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(pts)]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    (x0, y0), (x1, y1) = pts[0], pts[1]
    d = max(seg[0], 1e-6)
    vx, vy = speed_px * (x1 - x0) / d, speed_px * (y1 - y0) / d
    w = Walker(pid, x0, y0, vx, vy, vx, vy, t0, path=pts[1:], speed_px=speed_px)
    if hide_s is not None and hide_s > 0:
        k = int(np.argmin([abs(y - GATE_Y) for _, y in pts]))
        centre = float(cum[k]) if abs(pts[k][1] - GATE_Y) < 60 else float(cum[-1]) / 2
        span = speed_px * (hide_s + 2 * RAMP_F / FPS)
        w.hide_at_arc = max(20.0, centre - span / 2)
    return w


def render(
    sc: Scenario,
    seed: int,
    walkers: list[Walker] | None = None,
    pillars: list[tuple[float, float, float, float]] | None = None,
) -> list[Frame]:
    """Simulate ``sc`` with ``seed``: per-frame truth and detections (deterministic).

    ``walkers`` (default: the choreography of ``sc.kind``) and ``pillars`` (tlwh occluders the
    detector does not see as people) are for user-built scenes; ``sc.hide_s`` is the length of
    every scripted hide either way."""
    rng = np.random.default_rng([seed, 11])
    drng = np.random.default_rng([seed, 12])
    ws = build_walkers(sc, rng) if walkers is None else walkers
    pillars = pillars or []
    n_f = round(sc.seconds * FPS)
    dt = 1.0 / FPS
    hide_f = round(sc.hide_s * FPS)
    tau, sig = 1.2, 10.0 * sc.speed  # OU velocity: relaxation (s), px/s noise level
    shake = np.zeros(2)
    frames: list[Frame] = []
    z = sc.noise
    for f in range(1, n_f + 1):
        t = f * dt
        if sc.jitter_px > 0:  # camera shake: AR(1) offset + per-frame jolt, same for everyone
            shake = 0.8 * shake + drng.normal(0, sc.jitter_px * 0.6, 2)
        for w in ws:
            if w.gone or t < w.t0:
                continue
            w.active = True
            standing = any(s0 <= t < s0 + d for s0, d in w.stops)
            if standing:
                w.vx, w.vy = 0.0, 0.0
            else:
                if w.vx == 0.0 and w.vy == 0.0:
                    w.vx, w.vy = w.gvx, w.gvy
                w.vx += (w.gvx - w.vx) * dt / tau + sig * math.sqrt(2 * dt / tau) * rng.normal()
                w.vy += (w.gvy - w.vy) * dt / tau + sig * math.sqrt(2 * dt / tau) * rng.normal()
            py = w.y
            if w.path:  # steer towards the next waypoint; small OU wobble stays in vx/vy
                tx, ty = w.path[0]
                dx, dy = tx - w.x, ty - w.y
                dist = math.hypot(dx, dy)
                if dist < max(6.0, w.speed_px * dt):
                    w.path.pop(0)
                    if not w.path:
                        w.gone = True
                        continue
                    tx, ty = w.path[0]
                    dx, dy = tx - w.x, ty - w.y
                    dist = max(math.hypot(dx, dy), 1e-6)
                w.gvx, w.gvy = w.speed_px * dx / dist, w.speed_px * dy / dist
                w.vx, w.vy = 0.7 * w.vx + 0.3 * w.gvx, 0.7 * w.vy + 0.3 * w.gvy
            ax, ay = w.x, w.y
            w.x += w.vx * dt
            w.y += w.vy * dt
            arc0 = w.arc
            w.arc += math.hypot(w.x - ax, w.y - ay)
            if w.turn_at is not None and not w.turned and (py - w.turn_at) * (w.y - w.turn_at) <= 0:
                w.turned = True
                w.gvy, w.vy = -w.gvy, -w.vy * 0.5
            if not w.speed_px and (w.x < 10 or w.x > W - 10):
                w.gvx, w.vx = -w.gvx, -w.vx
            if w.hide_left_f > 0:
                w.hide_left_f -= 1
            elif not w.hide_done and (
                (w.hide_at is not None and (py - w.hide_at) * (w.y - w.hide_at) <= 0)
                or (w.hide_at_arc is not None and arc0 <= w.hide_at_arc < w.arc)
            ):
                w.hide_done = True
                total = sc.dropout_f if sc.kind == "dropout" else hide_f + 2 * RAMP_F
                w.hide_left_f = w.hide_total_f = total
            if w.y < -8 or w.y > H + box_h(H) + 8 or (w.speed_px and not -60 < w.x < W + 60):
                w.gone = True
        on = [w for w in ws if w.active and not w.gone]
        boxes = np.asarray([tlwh_of(w.x, w.y) for w in on], np.float64).reshape(-1, 4)
        vis = _visible_fraction(boxes)
        if pillars:
            foot = np.asarray([w.y for w in on], np.float64)
            vis = vis * (1.0 - pillar_cover(boxes, foot, pillars))
        hidden: list[int] = []
        for i, w in enumerate(on):
            if w.hide_left_f > 0:
                if sc.kind == "dropout":
                    vis[i] = 0.0
                else:
                    k = w.hide_total_f - w.hide_left_f  # frames into the hide
                    ramp = min(k, w.hide_left_f - 1)
                    occl = 0.0 if ramp >= RAMP_F else 1.0 - (ramp + 1) / (RAMP_F + 1)
                    vis[i] = min(vis[i], occl)
                hidden.append(w.pid)
        truth = [
            (
                w.pid,
                float(boxes[i, 0]),
                float(boxes[i, 1]),
                float(boxes[i, 2]),
                float(boxes[i, 3]),
                float(vis[i]),
            )
            for i, w in enumerate(on)
            if boxes[i, 1] + boxes[i, 3] > 0 and boxes[i, 1] < H
        ]
        d_box: list[list[float]] = []
        d_sc: list[float] = []
        d_pid: list[int] = []
        for i, w in enumerate(on):
            v = float(vis[i])
            x, y, bw, bh = boxes[i]
            if y + bh <= 0 or y >= H or v < VIS_DROP:
                continue
            if drng.random() < 0.01 + 0.10 * z:  # plain miss
                continue
            jit = (0.4 + 3.0 * z) * bh / 80.0
            if v < VIS_WEAK:  # partly hidden: weak score; amodal (full) box or only the top part
                score = 0.12 + 0.5 * v + 0.1 * drng.random()
                trunc = drng.random() < TRUNCATE_P
                b = [x, y, bw, bh * max(v, 0.4) if trunc else bh]
            else:
                score = 0.62 + 0.33 * drng.random() * (0.6 + 0.4 * v)
                b = [x, y, bw, bh]
            b = [
                b[0] + jit * drng.normal() + shake[0],
                b[1] + jit * drng.normal() + shake[1],
                b[2] * (1 + 0.05 * z * drng.normal()),
                b[3] * (1 + 0.05 * z * drng.normal()),
            ]
            d_box.append(b)
            d_sc.append(float(min(score, 0.99)))
            d_pid.append(w.pid)
        for _ in range(int(drng.poisson(0.02 + 0.25 * z))):  # false positives
            fy = 60 + (H - 60) * drng.random()
            d_box.append(list(tlwh_of(40 + (W - 80) * drng.random(), fy)))
            d_sc.append(float(0.1 + 0.6 * drng.random()))
            d_pid.append(-1)
        # truth boxes shift with the camera too (the image moved, the people did not)
        truth = [
            (p, x + float(shake[0]), y + float(shake[1]), bw, bh, v) for p, x, y, bw, bh, v in truth
        ]
        frames.append(
            Frame(
                f,
                truth,
                np.asarray(d_box, np.float64).reshape(-1, 4),
                np.asarray(d_sc, np.float64),
                d_pid,
                hidden,
            )
        )
    return frames

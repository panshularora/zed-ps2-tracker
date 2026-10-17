"""Stress Map + replay + Break it yourself: the stress simulator (``eval/stress_sim.py``) and the
REAL trackers, served to the web app.

* ``GET /api/stress/map/{config}``: the committed grid (``docs/results/stress/stress_map*.json``),
  compacted for the heatmap. Read-only; nothing is recomputed.
* ``GET /api/stress/replay``: one (people, hidden, seed) cell re-simulated with both trackers, per
  frame, with the tracker trace's reason for every ID change.
* ``POST /api/break/run``: a user-built scene (drawn path, pillars, groups, a hidden person),
  simulated and tracked the same way. Deterministic: the same spec gives the same frames.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from eval import stress_run as sr
from eval.stress_metrics import GATE_HYST, gate_counts, id_change_events, match_frames
from eval.stress_sim import FPS, GATE_Y, Frame, H, Scenario, W, Walker, _walker, path_walker, render

ROOT = Path(__file__).resolve().parents[1]
STRESS = ROOT / "docs" / "results" / "stress"
OURS_CFG = {
    "default": "configs/ours/visibility_botsort.yaml",
    "mem6": "configs/ours/visibility_botsort_mem6.yaml",
}
MAP_FILE = {"default": "stress_map.json", "mem6": "stress_map_mem6.json"}
LABEL = {"bytetrack": "ByteTrack", "ours": "Ours"}


# ---------------------------------------------------------------- one-line reasons
def short_why(x: dict[str, Any], tracker: str) -> str:
    """'ByteTrack: memory expired after 1.0 s hidden' — one plain line from the trace fields."""
    who = LABEL.get(tracker, tracker)
    r = x["reason"]
    if r == "expired":
        e = x.get("from_expired") or {}
        lost = e.get("frames_lost")
        mem = e.get("buffer")
        s = f"{lost / FPS:.1f} s" if isinstance(lost, int | float) else "a while"
        m = (
            f" (memory {mem / FPS:.1f} s)"
            if isinstance(mem, int | float) and isinstance(lost, int | float) and lost - mem > 1
            else ""
        )
        return f"{who}: memory expired after {s} hidden{m}"
    if r == "cost_above_threshold":
        c: dict[str, Any] = next((k for k in x["candidates"] if k["track_id"] == x["from_id"]), {})
        return (
            f"{who}: still remembered #{x['from_id']}, but its predicted box was off "
            f"(overlap {c.get('iou', 0):.2f}), so it started a new number"
        )
    if r == "taken_by_other_box":
        return f"{who}: #{x['from_id']} was matched to someone else's box that frame"
    if r == "low_score_box":
        return f"{who}: the person came back as a weak box; lost numbers only take confident boxes"
    if r == "transfer":
        return f"{who}: #{x['to_id']} slid over from another person"
    return f"{who}: #{x['from_id']} was not a candidate for this box"


# ---------------------------------------------------------------- map
def compact_map(d: dict[str, Any], tag: str) -> dict[str, Any]:
    cells = []
    for c in d["cells"]:
        s, df = c["scenario"], c["id_change_diff"]
        diffs = [
            o - b
            for o, b in zip(
                c["ours"]["per_seed"]["id_changes"],
                c["bytetrack"]["per_seed"]["id_changes"],
                strict=True,
            )
        ]
        # representative seed: the one whose difference is the median (ties: lowest seed)
        order = sorted(range(len(diffs)), key=lambda i: (diffs[i], c["seeds"][i]))
        mid = order[(len(order) - 1) // 2]
        fd = c["first_disagreement"][mid] if c.get("first_disagreement") else None
        cells.append(
            {
                "people": s["people"],
                "hide_s": s["hide_s"],
                "diff_mean": df["mean"],
                "ci95": df["ci95"],
                "p": df["sign_test_p"],
                "verdict": df["verdict"],
                "ours_mean": c["ours"]["id_changes"]["mean"],
                "bytetrack_mean": c["bytetrack"]["id_changes"]["mean"],
                "seeds": c["seeds"],
                "replay_seed": c["seeds"][mid],
                "replay_diff": diffs[mid],
                "first_disagreement": (
                    {k: fd.get(k) for k in ("seed", "frame", "pid", "changed")} if fd else None
                ),
            }
        )
    return {
        "config": tag,
        "ours_config": d["trackers"]["ours"],
        "baseline_config": d["trackers"]["bytetrack"],
        "seeds": d["seeds"],
        "git": d["git"],
        "axes": d["axes"],
        "ours_memory_s": _memory_s(d["trackers"]["ours"]),
        "baseline_memory_s": _memory_s(d["trackers"]["bytetrack"]),
        "cells": cells,
    }


def _memory_s(path: str) -> float:
    cfg = sr.config(path)
    scale = cfg.visibility.occluded_buffer_scale if cfg.visibility.estimator != "constant" else 1.0
    return round(cfg.lifecycle.track_buffer * scale / FPS, 2)


@lru_cache(maxsize=4)
def stress_map(tag: str) -> dict[str, Any]:
    f = STRESS / MAP_FILE[tag]
    return compact_map(json.loads(f.read_text()), tag)


# ---------------------------------------------------------------- replay of one scene
def _crossing_frames(seq: list[tuple[int, float]]) -> list[tuple[int, int]]:
    """(frame, +1 down / -1 up) for each side change of a foot-y sequence (same rule as
    ``eval.stress_metrics._crossings``)."""
    side, out = 0, []
    for f, y in seq:
        s = 1 if y > GATE_Y + GATE_HYST else (-1 if y < GATE_Y - GATE_HYST else 0)
        if s == 0:
            continue
        if side and s != side:
            out.append((f, s))
        side = s
    return out


def match_crossings(
    true_by: list[tuple[int, int, int]],
    trk_by: list[tuple[int, int, int]],
    fps: float,
    window_s: float = 2.0,
) -> tuple[list[tuple[int, int, int]], int]:
    """Which true crossings a tracker missed. ``true_by``: (frame, person, direction);
    ``trk_by``: (frame, owner person, direction), owner = the person the track matched most often.
    A true crossing is counted when an unused tracker crossing by the same person, in the same
    direction, lies within ``window_s``. Returns (missed true crossings, tracker crossings left
    over, i.e. counted crossings that match no true one)."""
    used = [False] * len(trk_by)
    missed = []
    for f, p, d in sorted(true_by):
        best = None
        for i, (g, q, e) in enumerate(trk_by):
            if used[i] or q != p or e != d or abs(g - f) > window_s * fps:
                continue
            if best is None or abs(g - f) < abs(trk_by[best][0] - f):
                best = i
        if best is None:
            missed.append((f, p, d))
        else:
            used[best] = True
    return missed, used.count(False)


def track_scene(
    frames: list[Frame], ours_cfg: str, seed: int, settings: dict[str, Any]
) -> dict[str, Any]:
    trackers = {"bytetrack": sr.TRACKERS["bytetrack"], "ours": ours_cfg}
    out: dict[str, Any] = {
        "seed": seed,
        "settings": settings,
        "width": W,
        "height": H,
        "gate_y": GATE_Y,
        "fps": FPS,
        "trackers": trackers,
        "truth": [
            {
                "f": fr.f,
                "p": [
                    [p, round(x + w / 2, 1), round(y + h, 1), round(v, 2)]
                    for p, x, y, w, h, v in fr.truth
                ],
                "hidden": fr.hidden,
            }
            for fr in frames
        ],
    }
    tru: dict[int, list[tuple[int, float]]] = {}
    for fr in frames:
        for p, _x, y, _w, h, _v in fr.truth:
            tru.setdefault(p, []).append((fr.f, y + h))
    out["true_crossings"] = sorted(f for s in tru.values() for f, _ in _crossing_frames(s))
    true_by = sorted((f, p, d) for p, s in tru.items() for f, d in _crossing_frames(s))
    out["true_crossings_by"] = [list(x) for x in true_by]  # [frame, person, direction]
    panes: dict[str, Any] = {}
    for name, path in trackers.items():
        cfg = sr.config(path)
        rows, traces = sr.run_tracker(cfg, frames, trace=True)
        m = match_frames(frames, rows)
        evs = []
        for e in id_change_events(m):
            x = sr.explain(e, frames, m, traces, cfg)
            evs.append(
                {
                    "frame": e[0],
                    "pid": e[1],
                    "from": e[2],
                    "to": e[3],
                    "reason": x["reason"],
                    "line": short_why(x, name),
                    "why": sr.why_text(x),
                }
            )
        trk: dict[int, list[tuple[int, float]]] = {}
        for k, tr in enumerate(rows):
            for t, b in tr:
                trk.setdefault(t, []).append((k + 1, b[1] + b[3]))
        true_n, trk_n = gate_counts(frames, rows)
        inv = [{t: p for p, t in mm.items()} for mm in m]
        votes: dict[int, Counter[int]] = {}
        for mm in m:
            for p, t in mm.items():
                votes.setdefault(t, Counter())[p] += 1
        owner = {t: c.most_common(1)[0][0] for t, c in votes.items()}
        trk_by = [(f, owner.get(t, -1), d) for t, s in trk.items() for f, d in _crossing_frames(s)]
        missed, extra = match_crossings(true_by, trk_by, FPS)
        panes[name] = {
            "label": LABEL[name],
            "config": path,
            # per frame: [track id, matched person (-1 none), foot x, foot y]
            "tracks": [
                [
                    [t, inv[k].get(t, -1), round(b[0] + b[2] / 2, 1), round(b[1] + b[3], 1)]
                    for t, b in tr
                ]
                for k, tr in enumerate(rows)
            ],
            "matched": [{str(p): t for p, t in mm.items()} for mm in m],
            "events": evs,
            "crossings": sorted(f for s in trk.values() for f, _ in _crossing_frames(s)),
            "count": trk_n,
            "true_count": true_n,
            # true crossings this tracker did not count: [frame, person, +1 down / -1 up]
            "missed": [list(x) for x in missed],
            "extra": extra,
        }
    out["panes"] = panes
    return out


def replay(people: int, hide_s: float, seed: int, config: str) -> dict[str, Any]:
    sc = Scenario(people=people, hide_s=hide_s)
    frames = render(sc, seed)
    return track_scene(
        frames,
        OURS_CFG[config],
        seed,
        {"people": people, "hide_s": hide_s, "noise": sc.noise, "config": config},
    )


# ---------------------------------------------------------------- Break it yourself
PILLAR_W, PILLAR_H = 46.0, 150.0
SPEED_PX = 70.0


class BreakSpec(BaseModel):
    seed: int = 7
    hide_s: float = Field(1.5, ge=0.0, le=8.0)
    hide_pid: int | None = 1
    crowd: int = Field(0, ge=0, le=12)
    pillars: list[tuple[float, float]] = Field(default_factory=list, max_length=8)  # foot points
    groups: list[tuple[float, float]] = Field(default_factory=list, max_length=4)
    path: list[tuple[float, float]] = Field(default_factory=list, max_length=400)
    seconds: float = Field(14.0, ge=2.0, le=30.0)
    config: Literal["default", "mem6"] = "default"


DEFAULT_PATH = [(150.0, 40.0), (330.0, 200.0), (480.0, 300.0), (620.0, 420.0), (760.0, 560.0)]


def break_walkers(spec: BreakSpec) -> tuple[list[Walker], list[tuple[float, float, float, float]]]:
    """Walkers and pillars of a spec. pids: 1 = the drawn path, then 3 per group, then the crowd."""
    import numpy as np

    rng = np.random.default_rng([spec.seed, 21])

    def hide(pid: int) -> float | None:
        return spec.hide_s if spec.hide_pid == pid else None

    path = spec.path if len(spec.path) >= 2 else DEFAULT_PATH
    ws = [path_walker(1, path, 0.3, SPEED_PX, hide(1))]
    pid = 2
    for gx, gy in spec.groups:
        for k in range(3):
            x = gx + (k - 1) * 26.0
            # the group walks top to bottom through the point where it was dropped
            ws.append(
                path_walker(
                    pid, [(x, -4.0), (x, gy), (x, H + 120.0)], 0.2 * k + 0.5, 62.0, hide(pid)
                )
            )
            pid += 1
    sc = Scenario(
        people=max(spec.crowd, 1),
        hide_s=spec.hide_s,
        seconds=spec.seconds,
        arrive_s=spec.seconds * 0.6,
    )
    for _ in range(spec.crowd):
        w = _walker(rng, pid, sc.arrive_s * rng.random(), sc)
        if spec.hide_pid == pid and spec.hide_s > 0:
            w.hide_at = GATE_Y - (20.0 if w.gvy > 0 else -20.0)
        ws.append(w)
        pid += 1
    pillars = [(x - PILLAR_W / 2, y - PILLAR_H, PILLAR_W, PILLAR_H) for x, y in spec.pillars]
    return ws, pillars


def break_run(spec: BreakSpec) -> dict[str, Any]:
    ws, pillars = break_walkers(spec)
    sc = Scenario(people=len(ws), hide_s=spec.hide_s, seconds=spec.seconds)
    frames = render(sc, spec.seed, walkers=ws, pillars=pillars)
    out = track_scene(frames, OURS_CFG[spec.config], spec.seed, spec.model_dump())
    out["pillars"] = [list(p) for p in pillars]
    out["people"] = [
        {"pid": w.pid, "kind": "path" if w.pid == 1 else ("group" if w.speed_px else "crowd")}
        for w in ws
    ]
    return out


# ---------------------------------------------------------------- routes
router = APIRouter()


@router.get("/api/stress/map/{config}")
def get_stress_map(config: str) -> dict[str, Any]:
    if config not in MAP_FILE or not (STRESS / MAP_FILE[config]).exists():
        raise HTTPException(404, f"no stress map for config {config!r}")
    return stress_map(config)


@router.get("/api/stress/replay")
def get_replay(
    people: int = Query(10, ge=1, le=60),
    hide_s: float = Query(1.0, ge=0.0, le=8.0),
    seed: int = Query(1000),
    config: Literal["default", "mem6"] = Query("default"),
) -> dict[str, Any]:
    return replay(people, hide_s, seed, config)


@router.post("/api/break/run")
def post_break(spec: BreakSpec) -> dict[str, Any]:
    try:
        return break_run(spec)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

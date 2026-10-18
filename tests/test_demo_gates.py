"""Demo gates: one documented rule for every clip, from where people walk (never from counts)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.demo.guide import motion_gate, rows_of
from serve.clips import load_manifest

ROOT = Path(__file__).resolve().parents[1]
SCN = ROOT / "web" / "public" / "scenarios"


def _rows(paths: list[list[tuple[float, float]]]) -> np.ndarray:
    """Tracks from foot-point paths (one box per frame, 40 x 100 px)."""
    out = []
    for tid, path in enumerate(paths, 1):
        for f, (x, y) in enumerate(path, 1):
            out.append([f, tid, x - 20, y - 100, 40, 100])
    return np.asarray(out, float)


def _down(x: float, n: int = 30) -> list[tuple[float, float]]:
    return [(x, 200 + 25 * f) for f in range(n)]


def test_people_walking_down_get_a_horizontal_line_and_in_is_down():
    rows = {"baseline": _rows([_down(300), _down(900), _down(1500)])}
    g = motion_gate(rows, 10.0, 1920, 1080)
    assert g["a"][1] == g["b"][1]  # horizontal: perpendicular to the walk
    assert abs(g["walk_deg"] - 90.0) < 1e-6  # walking +y (down the image)
    # "in" = left-hand normal of a->b = the way most people walk (eval/gate_events convention)
    u = np.subtract(g["b"], g["a"]) / np.hypot(*np.subtract(g["b"], g["a"]))
    assert np.dot([-u[1], u[0]], [0, 1]) > 0


def test_camera_pan_is_removed_before_finding_the_walking_direction():
    """People walk down while the phone pans 60 px/frame to the right: image motion is mostly
    sideways, but with the background shift removed the line is still horizontal."""
    n = 30
    pan = [[60.0 * f, 0.0] for f in range(n)]
    paths = [[(x + 60.0 * f, 200 + 25 * f) for f in range(n)] for x in (100, 300, 500)]
    rows = {"baseline": _rows(paths)}
    raw = motion_gate(rows, 10.0, 1920, 1080)
    fixed = motion_gate(rows, 10.0, 1920, 1080, pan)
    assert abs(raw["walk_deg"]) < 45  # fooled by the pan without compensation
    assert fixed["a"][1] == fixed["b"][1]


def test_gate_does_not_depend_on_which_tracker_is_which():
    a = _rows([_down(300), _down(900)])
    b = _rows([_down(1200)])
    g1 = motion_gate({"baseline": a, "ours": b}, 10.0, 1920, 1080)
    g2 = motion_gate({"baseline": b, "ours": a}, 10.0, 1920, 1080)
    assert g1 == g2


def test_committed_guides_and_clips_yaml_use_the_rule_output():
    man = {c.id: c for c in load_manifest()}
    idx = json.loads((SCN / "index.json").read_text())
    assert idx, "no scenarios"
    for meta in idx:
        d = json.loads((SCN / f"{meta['id']}.json").read_text())
        m = d["meta"]
        rows = {k: rows_of(v["frames"]) for k, v in d["runs"].items()}
        g = motion_gate(rows, m["src_fps"] / m["stride"], m["width"], m["height"], d["camera"])
        assert d["guide"]["gate"]["norm"] == g["norm"], meta["id"]
        assert [d["guide"]["gate"]["a"], d["guide"]["gate"]["b"]] == [g["a"], g["b"]]
        assert list(man[meta["id"]].gate) == g["norm"], meta["id"]  # configs/demo/clips.yaml
        assert len(d["camera"]) == m["frames_tracked"]


def test_every_clip_on_the_rail_counts_someone():
    """A clip stays on the rail only if at least one tracker counts a crossing at its gate."""
    for meta in json.loads((SCN / "index.json").read_text()):
        d = json.loads((SCN / f"{meta['id']}.json").read_text())
        n = sum(e["t"] == "cross" for ev in d["guide"]["events"].values() for e in ev)
        assert n > 0, meta["id"]

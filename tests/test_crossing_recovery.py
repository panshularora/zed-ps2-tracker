"""Crossing recovery (eval/crossing_recovery.py): one crossing per ended + new number pair."""

from __future__ import annotations

import numpy as np

from eval.crossing_recovery import RecoveryParams, recover_crossings
from eval.gate_events import Gate, crossings

GATE = Gate("g", (0.0, 300.0), (1000.0, 300.0))
P = RecoveryParams(max_gap_s=2.0, max_dist_rel=1.5, near_rel=1.0)
H = 10.0


def walk(tid: int, f0: int, x: float, y0: float, y1: float, n: int) -> list[list[float]]:
    """MOT rows of a 40x100 box whose foot point goes from (x, y0) to (x, y1) over n frames."""
    ys = np.linspace(y0, y1, n)
    return [[f0 + k, tid, x - 20, y - 100, 40, 100] for k, y in enumerate(ys)]


def rows(*parts: list[list[float]]) -> np.ndarray:
    return np.asarray([r for p in parts for r in p], float)


def test_ended_and_new_number_across_the_line_count_once() -> None:
    r = rows(walk(1, 1, 500, 150, 280, 20), walk(2, 40, 505, 320, 450, 20))
    assert crossings(r, GATE, H) == []  # neither number crossed
    ev = recover_crossings(r, GATE, H, 30.0, P)
    assert len(ev) == 1
    e = ev[0]
    assert (e.frame, e.track_id, e.from_track_id, e.direction) == (40, 2, 1, "in")
    up = recover_crossings(
        rows(walk(1, 1, 500, 450, 320, 20), walk(2, 40, 505, 280, 150, 20)), GATE, H, 30.0, P
    )
    assert [x.direction for x in up] == ["out"]


def test_no_false_recovery_when_two_different_people_cross() -> None:
    # two people cross at the same time in opposite directions, side by side; both keep their
    # numbers, so the plain count has both crossings and recovery must add nothing
    r = rows(walk(1, 1, 480, 150, 450, 60), walk(2, 1, 540, 450, 150, 60))
    assert len(crossings(r, GATE, H)) == 2
    assert recover_crossings(r, GATE, H, 30.0, P) == []
    assert recover_crossings(r, GATE, H, 30.0, RecoveryParams()) == []  # tuned limits too
    # a second pair right after: B walks up and leaves, C enters from the top later; still only
    # their own crossings, because each number crosses by itself
    r2 = rows(walk(1, 1, 480, 150, 450, 60), walk(2, 70, 480, 450, 150, 60))
    assert len(crossings(r2, GATE, H)) == 2
    assert recover_crossings(r2, GATE, H, 30.0, RecoveryParams()) == []


def test_no_recovery_far_away_same_side_too_late_or_overlapping() -> None:
    a = walk(1, 1, 100, 150, 280, 20)
    assert recover_crossings(rows(a, walk(2, 30, 800, 320, 450, 20)), GATE, H, 30.0, P) == []  # far
    assert (
        recover_crossings(rows(a, walk(2, 30, 105, 285, 150, 20)), GATE, H, 30.0, P) == []
    )  # same side
    assert (
        recover_crossings(rows(a, walk(2, 200, 105, 320, 450, 20)), GATE, H, 30.0, P) == []
    )  # 6 s later
    assert (
        recover_crossings(rows(a, walk(2, 10, 105, 320, 450, 20)), GATE, H, 30.0, P) == []
    )  # overlap
    # ended far from the line (left the scene upstream): not a crossing
    far = walk(1, 1, 500, 50, 120, 20)
    assert recover_crossings(rows(far, walk(2, 30, 505, 320, 450, 20)), GATE, H, 30.0, P) == []


def test_one_to_one_closest_pair_wins() -> None:
    r = rows(
        walk(1, 1, 500, 150, 280, 20),  # ends frame 20
        walk(3, 1, 560, 150, 282, 25),  # ends frame 25
        walk(2, 30, 505, 320, 450, 20),  # starts frame 30
    )
    ev = recover_crossings(r, GATE, H, 30.0, P)
    assert [(e.from_track_id, e.track_id) for e in ev] == [(3, 2)]


def test_tuned_defaults_are_the_documented_ones() -> None:
    import json
    from pathlib import Path

    doc = json.loads(Path("docs/results/stress/crossing_recovery.json").read_text())
    assert RecoveryParams() == RecoveryParams(**doc["tuning"]["chosen"]["params"])


def test_recovered_crossings_are_suggestions_never_counts() -> None:
    from serve.analysis import metrics

    ev = [
        {"type": "crossing", "direction": "in", "track_id": 1, "frame": 5},
        {"type": "recovered_crossing", "direction": "out", "track_id": 3, "from_track_id": 2,
         "frame": 9, "recovered": True},
    ]  # fmt: skip
    m = metrics(np.zeros((0, 8)), 30.0, 10, ev, [], [])
    assert (m["crossings_in"], m["crossings_out"], m["possible_missed_crossings"]) == (1, 0, 1)
    assert not any("with_recovered" in k for k in m)


def test_recovered_events_are_json_serialisable() -> None:
    import json
    from dataclasses import asdict

    r = rows(walk(1, 1, 500, 150, 280, 20), walk(2, 40, 505, 320, 450, 20))
    ev = recover_crossings(r, GATE, H, 30.0, P)
    assert ev
    json.dumps([asdict(e) for e in ev])  # numpy ints here once broke upload jobs
    assert all(type(e.track_id) is int and type(e.from_track_id) is int for e in ev)

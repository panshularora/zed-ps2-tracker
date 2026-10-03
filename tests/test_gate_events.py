"""Gate events: hysteresis crossings, event matching, attribution, tailgating."""

import numpy as np
import pytest

from eval.gate_events import (
    Event,
    Gate,
    crossings,
    evaluate_sequence,
    id_bijection,
    match_events,
    summarise,
    tailgate_pairs,
)

GATE = Gate("h", (0.0, 500.0), (1000.0, 500.0))  # normal (0, 1) points down: up = "out"
PARAMS = {"hysteresis_px": 10, "delta_frames": 5, "tailgate_seconds": 1.0}


def walk(tid: int, x: float, ys: list[float], f0: int = 1) -> np.ndarray:
    """Rows whose bottom-centre is (x, y) for consecutive frames."""
    return np.array([[f0 + i, tid, x - 10, y - 50, 20, 50, 1, 1, 1.0] for i, y in enumerate(ys)])


def test_crossing_with_hysteresis() -> None:
    ys = [520, 515, 505, 499, 503, 497, 492, 480, 470]  # jitters around the line, then crosses
    ev = crossings(walk(1, 100, ys), GATE, h=10)
    assert len(ev) == 1
    assert ev[0].direction == "out"
    assert ev[0].frame == 8  # first frame beyond -10 px (y = 480)
    back = crossings(walk(1, 100, [*ys, 490, 505, 515]), GATE, h=10)
    assert [e.direction for e in back] == ["out", "in"]
    # outside the segment: ignored
    assert crossings(walk(2, 1500, ys), GATE, h=10) == []


def test_match_events_window_and_direction() -> None:
    gt = [Event(10, 1, "h", "in"), Event(50, 2, "h", "out")]
    tr = [Event(13, 9, "h", "in"), Event(50, 8, "h", "in"), Event(56, 7, "h", "out")]
    assert match_events(gt, tr, delta=5) == [(0, 0)]


def test_tailgate_pairs() -> None:
    ev = [Event(10, 1, "h", "in"), Event(20, 2, "h", "in"), Event(100, 3, "h", "in"),
          Event(105, 3, "h", "in")]  # fmt: skip
    pairs = tailgate_pairs(ev, window=30)
    assert [(e.track_id, p) for e, p in pairs] == [(2, 1)]  # same id never tailgates itself


def test_bijection_and_attribution() -> None:
    ys = [520.0 - 8 * i for i in range(10)]
    gt = np.vstack([walk(1, 100, ys), walk(2, 600, ys)])
    tr = gt.copy()
    tr[:, 1] = np.where(gt[:, 1] == 1, 11, 12)
    assert id_bijection(gt, tr) == {11: 1, 12: 2}
    r = evaluate_sequence(gt, tr, [GATE], PARAMS, fps=30.0)
    s = summarise({"seq": r})
    assert s["gt_crossings"] == 2
    assert s["count_error"] == 0
    assert s["attribution_error"] == 0
    assert s["gt_tailgates"] == 1  # both crossed within 1 s
    assert s["tailgate_recall"] == 1.0
    # swap identities just before the line: crossing events still match, attribution fails
    tr2 = tr.copy()
    late = tr2[:, 0] >= 3
    tr2[late, 1] = np.where(tr[late, 1] == 11, 12, 11)
    s2 = summarise({"seq": evaluate_sequence(gt, tr2, [GATE], PARAMS, fps=30.0)})
    assert s2["matched"] == 2
    assert s2["attribution_error"] == pytest.approx(1.0)

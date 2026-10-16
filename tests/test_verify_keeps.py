"""Kept-vs-swap check (scripts/demo/verify_keeps.py): a "kept #N" chip is only allowed when
the reappearance looks like the same person (Pexels heuristic) or is the same labelled person
(MOT GT)."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from scripts.demo.verify_keeps import (
    APP_MIN,
    PUB,
    Video,
    appearance,
    bhattacharyya,
    decide,
    gaps_of,
    hsv_hist,
    motion_check,
    occluder_of,
    tracks_of,
    verify_gt,
)


def _walk(
    n: int, gap: tuple[int, int] | None = None, vx: float = 10.0
) -> dict[int, tuple[float, float, float, float]]:
    return {
        f: (100 + vx * f, 200.0, 50.0, 100.0)
        for f in range(1, n + 1)
        if not (gap and gap[0] <= f <= gap[1])
    }


def test_gaps_and_motion_plausible_for_constant_walk() -> None:
    tr = _walk(30, (11, 20))
    assert gaps_of(tr) == [(10, 21)]
    err, tol, ratio = motion_check(tr, 10, 21, fps=10)
    assert err < 0.05
    assert ratio == 1.0
    assert tol == pytest.approx(0.6 + 0.9 * 1.1)


def test_motion_rejects_a_jump_to_the_other_side() -> None:
    tr = _walk(30, (11, 20))
    tr[21] = (900.0, 200.0, 50.0, 100.0)
    err, tol, _ = motion_check(tr, 10, 21, fps=10)
    assert err > tol


def test_histogram_similarity_same_vs_different_colour() -> None:
    orange = np.zeros((20, 10, 3), np.uint8)
    orange[..., 0] = 230
    orange[..., 1] = 110
    orange[..., 2] = 30
    white = np.full((20, 10, 3), 235, np.uint8)
    a, b = hsv_hist(orange), hsv_hist(white)
    assert a is not None
    assert b is not None
    assert bhattacharyya(a, a) == pytest.approx(1.0)
    assert bhattacharyya(a, b) < APP_MIN


def test_decide_needs_both_signals() -> None:
    assert decide(0.9, 0.1, 1.0, 1.1) == ("kept", [])
    v, why = decide(0.6, 0.1, 1.0, 1.1)
    assert v == "swap"
    assert "appearance" in why[0]
    v, why = decide(0.95, 2.0, 1.0, 1.1)
    assert v == "swap"
    assert "path" in why[0]
    v, why = decide(0.95, 0.1, 1.0, 1.9)
    assert v == "swap"
    assert "size" in why[0]
    assert decide(None, 0.1, 1.0, 1.0)[0] == "swap"  # no evidence -> never "kept"


def test_gt_matching_kept_vs_swap() -> None:
    gt = {1: _walk(30), 2: dict.fromkeys(range(1, 31), (600.0, 200.0, 50.0, 100.0))}
    kept = {7: _walk(30, (11, 20))}
    swapped = {
        7: {
            **dict(_walk(10).items()),
            **dict.fromkeys(range(21, 31), (600.0, 200.0, 50.0, 100.0)),
        }
    }
    assert verify_gt(kept, gt, "ours", 10)[0].verdict == "kept"
    r = verify_gt(swapped, gt, "ours", 10)[0]
    assert (r.verdict, r.gt_before, r.gt_after) == ("swap", 1, 2)


def test_occluder_is_the_box_in_front() -> None:
    tracks = {
        1: _walk(30, (11, 20), vx=0),
        2: dict.fromkeys(range(1, 31), (90.0, 190.0, 80.0, 120.0)),
        3: dict.fromkeys(range(1, 31), (900.0, 0.0, 10.0, 10.0)),
    }
    assert occluder_of(tracks, 1, 10, 21) == 2


# ---- regression: the subway turnstile "kept #4" that was really another person --------------
def test_committed_verdicts_flag_turnstile_4() -> None:
    doc = json.loads((PUB / "verify" / "subway_turnstile.json").read_text())
    ev = {(e["run"], e["id"], e["back"]): e for e in doc["events"]}
    four = ev[("ours", 4, 128)]
    assert four["last"] == 95
    assert four["verdict"] == "swap"
    assert any("appearance" in r for r in four["reasons"])
    assert any("size" in r for r in four["reasons"])
    assert ev[("ours", 10, 86)]["verdict"] == "kept"  # a story moment stays confirmed


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_turnstile_4_recomputed_from_video() -> None:
    scn = json.loads((PUB / "scenarios" / "subway_turnstile.json").read_text())
    tr = tracks_of(scn["runs"]["ours"]["frames"])[4]
    video = Video(PUB / "demo" / "subway_turnstile.mp4")
    scale = video.W / scn["meta"]["width"]
    app = appearance(video, tr, 95, 128, 1, scale)
    err, tol, ratio = motion_check(tr, 95, 128, 30.0)
    assert app is not None
    assert app < APP_MIN  # white top before, orange top after
    assert decide(app, err, tol, ratio)[0] == "swap"
    app_ok = appearance(video, tracks_of(scn["runs"]["ours"]["frames"])[10], 62, 86, 1, scale)
    assert app_ok is not None
    assert app_ok >= APP_MIN


def test_per_person_same_denominator_for_every_tracker():
    from scripts.demo.verify_keeps import per_person

    gt = {1: {f: (0.0, 0.0, 10.0, 20.0) for f in range(1, 31)}, 2: {f: (100.0, 0.0, 10.0, 20.0) for f in range(1, 31)}, 3: {1: (0.0, 0.0, 1.0, 1.0)}}
    keep = {7: {f: (0.0, 0.0, 10.0, 20.0) for f in range(1, 31) if not 10 <= f <= 15}, 8: {f: (100.0, 0.0, 10.0, 20.0) for f in range(1, 31)}}
    change = {7: {f: (0.0, 0.0, 10.0, 20.0) for f in range(1, 10)}, 9: {f: (0.0, 0.0, 10.0, 20.0) for f in range(16, 31)}}
    a, b = per_person(keep, gt, 30), per_person(change, gt, 30)
    assert a == {"labelled": 2, "kept": 2, "changed": 0, "missed": 0}  # gap allowed; short gt id 3 excluded
    assert b == {"labelled": 2, "kept": 0, "changed": 1, "missed": 1}
    assert a["labelled"] == b["labelled"]

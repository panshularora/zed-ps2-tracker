"""Stress suite: seeded scenario generator, scoring, verdict statistics and the decision trace."""

from __future__ import annotations

import numpy as np
import pytest

from eval import stress_run as sr
from eval.stress_metrics import first_disagreement, id_change_events, match_frames, tally
from eval.stress_metrics import trackeval_metrics as te_metrics
from eval.stress_sim import FPS, RAMP_F, Scenario, render
from tracker.config import load_config
from tracker.runner import build_tracker
from tracker.types import Detections


def _truth_tracks(frames):  # a perfect tracker: every scored true box, with the person's id
    return [[(p, (x, y, w, h)) for p, x, y, w, h, v in fr.truth if v >= 0.25] for fr in frames]


def test_render_is_deterministic_and_seeded() -> None:
    sc = Scenario(people=6, hide_s=1.0, seconds=8)
    a, b, c = render(sc, 3), render(sc, 3), render(sc, 4)
    assert [f.truth for f in a] == [f.truth for f in b]
    assert all(np.array_equal(x.dets, y.dets) for x, y in zip(a, b, strict=True))
    assert [f.truth for f in a] != [f.truth for f in c]


@pytest.mark.parametrize("hide_s", [0.5, 2.0])
def test_scripted_hide_lasts_hide_s(hide_s: float) -> None:
    fr = render(Scenario(people=1, hide_s=hide_s, seconds=16, arrive_s=0.0), 1)
    hidden = sum(1 for f in fr if f.hidden)
    assert hidden == round(hide_s * FPS) + 2 * RAMP_F
    zero = sum(1 for f in fr for t in f.truth if t[5] == 0.0)
    assert abs(zero - round(hide_s * FPS)) <= 2
    # no detection of the person while fully hidden
    for f in fr:
        if any(t[5] == 0.0 for t in f.truth):
            assert 1 not in f.det_pid


@pytest.mark.parametrize("kind", sorted({s.kind for s in sr.SCENARIOS.values()} | {"grid"}))
def test_every_scenario_kind_renders(kind: str) -> None:
    sc = next((s for s in sr.SCENARIOS.values() if s.kind == kind), Scenario())
    fr = render(Scenario(**{**sc.as_dict(), "seconds": 6.0}), 0)
    assert len(fr) == 180
    assert any(f.truth for f in fr)


def test_perfect_tracker_scores_perfectly() -> None:
    fr = render(Scenario(people=8, hide_s=1.0, seconds=12), 2)
    tr = _truth_tracks(fr)
    m = match_frames(fr, tr)
    t = tally(fr, m)
    assert t.kept == t.people > 0
    assert id_change_events(m) == []
    te = te_metrics(fr, tr)
    assert te["IDF1"] == pytest.approx(1.0)
    assert te["AssA"] == pytest.approx(1.0)
    assert te["IDSW"] == 0


def test_tally_kept_never_exceeds_people_and_shared_id_is_not_kept() -> None:
    fr = render(Scenario(people=4, hide_s=0.0, seconds=10), 5)
    tr = [[(1, b) for _, b in f] for f in _truth_tracks(fr)]  # one ID for everybody
    t = tally(fr, match_frames(fr, tr))
    assert t.kept == 0
    assert t.kept <= t.people


def test_first_disagreement() -> None:
    a = [(50, 3, 1, 9)]
    b = [(80, 3, 1, 7), (90, 4, 2, 8)]
    d = first_disagreement(a, b)
    assert d == {"frame": 50, "pid": 3, "changed": "a"}
    assert first_disagreement(a, a) is None


def test_verdict_statistics() -> None:
    assert sr.verdict(np.full(20, -2.0)) == "ours_better"
    assert sr.verdict(np.full(20, 2.0)) == "ours_worse"
    assert sr.verdict(np.zeros(20)) == "tie"
    assert sr.verdict(np.array([3, -3] * 10, float)) == "unclear"
    assert sr.sign_test_p(np.array([1.0] * 10)) == pytest.approx(2 / 2**10)


def test_trace_does_not_change_tracking() -> None:
    fr = render(Scenario(people=10, hide_s=1.5, seconds=10), 7)
    cfg = load_config(sr.ROOT / sr.TRACKERS["ours"])
    off, _ = sr.run_tracker(cfg, fr)
    on, traces = sr.run_tracker(cfg, fr, trace=True)
    assert off == on
    assert len(traces) == len(fr)
    st = traces[50]["stages"][0]
    assert st["stage"] == "high"
    assert len(st["iou"]) == len(st["track_ids"])


def test_explain_reads_an_expiry_from_the_trace() -> None:
    # one person hidden 2 s: ByteTrack's 30-frame memory runs out, the trace must say so
    sc = Scenario(people=1, hide_s=2.0, seconds=16, arrive_s=0.0, noise=0.0)
    fr = render(sc, 1)
    cfg = load_config(sr.ROOT / sr.TRACKERS["bytetrack"])
    rows, traces = sr.run_tracker(cfg, fr, trace=True)
    m = match_frames(fr, rows)
    ev = id_change_events(m)
    assert len(ev) == 1
    x = sr.explain(ev[0], fr, m, traces, cfg)
    assert x["reason"] == "expired"
    assert x["from_expired"]["frames_lost"] > x["from_expired"]["buffer"] == 30
    assert "deleted" in sr.why_text(x)


def test_tracker_without_trace_flag_is_untouched() -> None:
    trk, _ = build_tracker(load_config(sr.ROOT / sr.TRACKERS["ours"]), 30.0)
    trk.update(Detections(np.array([[10.0, 10, 20, 50]]), np.array([0.9])))
    assert trk.last_trace is None  # type: ignore[attr-defined]

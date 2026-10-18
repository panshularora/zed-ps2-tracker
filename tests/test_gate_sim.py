"""Synthetic gate scene: determinism, detector behaviour and the NDJSON stream."""

import json

import numpy as np
from fastapi.testclient import TestClient

from serve.app import create_app
from serve.gate_sim import MAX_PEOPLE, Person, SimParams, World, detect, simulate


def frames(p: SimParams, trackers=("bytetrack", "ours")):
    return list(simulate(p, trackers))


def test_same_seed_replays_identically():
    p = SimParams(seed=11, seconds=4, play="crowd")
    assert frames(p) == frames(p)


def test_different_seed_differs():
    a, b = (
        frames(SimParams(seed=1, seconds=4, play="crowd")),
        frames(SimParams(seed=2, seconds=4, play="crowd")),
    )
    assert a[1:] != b[1:]


def test_both_trackers_see_identical_detections():
    # the detector stream does not depend on which trackers run
    p = SimParams(seed=5, seconds=3, play="crowd")
    one = [x.get("dets") for x in simulate(p, ("bytetrack",))]
    two = [x.get("dets") for x in simulate(p, ("bytetrack", "ours"))]
    assert one == two


def test_people_capped_and_truth_counted():
    xs = frames(SimParams(seed=3, density=1.0, seconds=20, speed=0.5, play="crowd"))
    assert max(len(x.get("people", [])) for x in xs) <= MAX_PEOPLE
    end = xs[-1]
    assert end["type"] == "end"
    assert end["true_count"] > 0
    assert set(end["summary"]) == {"bytetrack", "ours"}
    for f in xs[1:-1]:
        for k in ("bytetrack", "ours"):
            assert f[k]["counted"] >= 0
            assert f[k]["id_changes"] >= 0


def test_detector_drops_occluded_and_merges_overlaps():
    p = SimParams(noise=1.0, play="crowd")
    rng = np.random.default_rng(0)
    hidden = [Person(1, 100, 100, 0, 0, 0, 0, None, occl_left=1.0)]
    n = sum(len(detect(hidden, p, rng)[0]) for _ in range(200))
    assert n < 200 * 0.6  # mostly dropped (only weak partial boxes, plus false positives)
    pair = [Person(1, 100, 100, 0, 0, 0, 0, None), Person(2, 105, 104, 0, 0, 0, 0, None)]
    merged = 0
    for _ in range(200):
        b, _s = detect(pair, p, rng)
        merged += any(w > 26 for w in b[:, 2])
    assert merged > 50


def test_world_seeded():
    a = World(SimParams(), np.random.default_rng(1))
    b = World(SimParams(), np.random.default_rng(1))
    for _ in range(60):
        a.step(1 / 30)
        b.step(1 / 30)
    assert [(q.pid, q.x, q.y) for q in a.people] == [(q.pid, q.x, q.y) for q in b.people]


def test_stream_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("ZED_DATA_DIR", str(tmp_path))
    c = TestClient(create_app())
    r = c.get(
        "/api/gate-sim/stream",
        params={"seed": 9, "seconds": 2, "density": 0.8, "play": "crowd"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(x) for x in r.text.strip().split("\n")]
    assert lines[0]["type"] == "meta"
    assert lines[0]["params"]["seed"] == 9
    assert lines[0]["trackers"]["bytetrack"]["config"] == "configs/baselines/bytetrack.yaml"
    assert [x["f"] for x in lines[1:-1]] == list(range(1, 61))
    assert lines[-1]["type"] == "end"
    again = [
        json.loads(x)
        for x in c.get(
            "/api/gate-sim/stream",
            params={"seed": 9, "seconds": 2, "density": 0.8, "play": "crowd"},
        )
        .text.strip()
        .split("\n")
    ]
    assert again == lines


def test_committed_recording_matches_backend():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    rec = [
        json.loads(x)
        for x in (root / "web/public/offline/gate-sim.ndjson").read_text().strip().split("\n")
    ]
    raw = dict(rec[0]["params"])
    raw.setdefault("play", "crowd")
    p = SimParams(**raw)
    assert rec[-1] == list(simulate(p))[-1]


def test_committed_keep_recording_matches_backend():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    lines = (root / "web/public/offline/gate-keep.ndjson").read_text().strip().split("\n")
    rec = [json.loads(x) for x in lines]
    assert rec == json.loads(json.dumps(frames(SimParams(**rec[0]["params"]))))


def test_keep_play_runs_the_real_ours_tracker_no_relabel():
    """The planned two-person hide runs BoT-SORT + visibility, not a display relabeller."""
    xs = frames(SimParams(play="keep", seed=7, occlusion_s=1.5, noise=0.15, speed=1.0))
    meta, end = xs[0], xs[-1]
    assert meta["play"] == "keep"
    assert meta["trackers"]["ours"]["config"] == "configs/ours/visibility_botsort.yaml"
    assert "keep" not in meta["trackers"]["ours"]["label"].lower()
    assert end["summary"]["bytetrack"]["id_changes"] >= 1
    hides = [x for x in xs if x.get("beat") == "hide"]
    backs = [x for x in xs if x.get("beat") == "back"]
    assert len(hides) == 1
    assert hides[0]["focus"] == 1
    assert len(backs) == 1
    assert backs[0]["focus"] == 1
    assert backs[0]["f"] - hides[0]["f"] >= 40  # longer than the 30-frame base buffer


def test_keep_id_relabeller_is_gone():
    import importlib.util

    assert importlib.util.find_spec("tracker.keep_id") is None


def _ids_of(xs, pane: str, pid: int) -> list[int]:
    """The tracker numbers simulated person ``pid`` had in ``pane``, in order, without repeats."""
    out: list[int] = []
    for x in xs[1:-1]:
        tid = dict(x[pane]["ids"]).get(pid)
        if tid is not None and (not out or out[-1] != tid):
            out.append(tid)
    return out


def test_keep_play_ours_keeps_person_1_on_one_id_bytetrack_does_not():
    """Default keep play (1.5 s hide): the real trackers decide; nothing relabels a number.

    #1 is mostly covered when its box drops (visibility ~0.5 < 0.7), so ours keeps the track for 90
    frames; ByteTrack's 30-frame buffer expires during the 45-frame hide.
    """
    xs = frames(SimParams(play="keep"), ("bytetrack", "ours"))
    assert xs[0]["focus"] == 1
    assert len(_ids_of(xs, "ours", 1)) == 1
    assert len(_ids_of(xs, "bytetrack", 1)) == 2
    assert len(_ids_of(xs, "ours", 2)) == 1
    hidden = [x["f"] for x in xs[1:-1] if any(q[0] == 1 and q[3] == 0 for q in x["people"])]
    assert len(hidden) == 45  # 1.5 s at 30 fps, longer than ByteTrack's 30-frame buffer


def test_keep_play_ours_wins_only_inside_its_memory_range():
    """Same world, longer hides: past the 90-frame (3 s) memory ours issues a new number too."""
    keeps = {}
    for occ in (0.5, 1.0, 2.0, 3.0, 3.5):
        xs = frames(SimParams(play="keep", occlusion_s=occ), ("bytetrack", "ours"))
        keeps[occ] = (len(_ids_of(xs, "bytetrack", 1)) == 1, len(_ids_of(xs, "ours", 1)) == 1)
    assert keeps == {
        0.5: (True, True),
        1.0: (False, True),
        2.0: (False, True),
        3.0: (False, True),
        3.5: (False, False),
    }

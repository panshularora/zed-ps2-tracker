"""Synthetic gate scene: determinism, detector behaviour and the NDJSON stream."""

import json

import numpy as np
from fastapi.testclient import TestClient

from serve.app import create_app
from serve.gate_sim import MAX_PEOPLE, Person, SimParams, World, detect, simulate


def frames(p: SimParams, trackers=("bytetrack", "ours")):
    return list(simulate(p, trackers))


def test_same_seed_replays_identically():
    p = SimParams(seed=11, seconds=4)
    assert frames(p) == frames(p)


def test_different_seed_differs():
    a, b = frames(SimParams(seed=1, seconds=4)), frames(SimParams(seed=2, seconds=4))
    assert a[1:] != b[1:]


def test_both_trackers_see_identical_detections():
    # the detector stream does not depend on which trackers run
    p = SimParams(seed=5, seconds=3)
    one = [x.get("dets") for x in simulate(p, ("bytetrack",))]
    two = [x.get("dets") for x in simulate(p, ("bytetrack", "ours"))]
    assert one == two


def test_people_capped_and_truth_counted():
    xs = frames(SimParams(seed=3, density=1.0, seconds=20, speed=0.5))
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
    p = SimParams(noise=1.0)
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
    r = c.get("/api/gate-sim/stream", params={"seed": 9, "seconds": 2, "density": 0.8})
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
        for x in c.get("/api/gate-sim/stream", params={"seed": 9, "seconds": 2, "density": 0.8})
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
    p = SimParams(**rec[0]["params"])
    assert rec[-1] == list(simulate(p))[-1]

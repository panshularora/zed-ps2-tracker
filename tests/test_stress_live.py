"""Stress Map / replay / Break it yourself routes (serve/stress_live.py)."""

from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from eval import stress_run as sr
from eval.stress_sim import Scenario, path_walker, render
from serve.app import create_app
from serve.stress_live import BreakSpec, break_run, short_why, stress_map

client = TestClient(create_app())


def test_map_is_the_committed_grid_compacted() -> None:
    raw = json.loads((sr.OUT / "stress_map.json").read_text())
    m = client.get("/api/stress/map/default").json()
    assert m["ours_config"] == "configs/ours/visibility_botsort.yaml"
    assert len(m["cells"]) == len(raw["cells"]) == 48
    for a, b in zip(m["cells"], raw["cells"], strict=True):
        assert a["diff_mean"] == b["id_change_diff"]["mean"]
        assert a["verdict"] == b["id_change_diff"]["verdict"]
        assert a["replay_seed"] in b["seeds"]
    assert m["ours_memory_s"] == 3.0
    assert m["baseline_memory_s"] == 1.0
    assert stress_map("mem6")["ours_memory_s"] == 6.0
    assert client.get("/api/stress/map/nope").status_code == 404


def test_replay_matches_the_stored_cell_and_explains_changes() -> None:
    m = stress_map("default")
    c = next(x for x in m["cells"] if x["people"] == 10 and x["hide_s"] == 1.0)
    r = client.get(f"/api/stress/replay?people=10&hide_s=1&seed={c['replay_seed']}").json()
    n_ours = len(r["panes"]["ours"]["events"])
    n_bt = len(r["panes"]["bytetrack"]["events"])
    assert n_ours - n_bt == c["replay_diff"]  # same simulator + trackers as the stored grid
    fd = c["first_disagreement"]
    first = min(
        (
            e["frame"]
            for k in ("ours", "bytetrack")
            for e in r["panes"][k]["events"]
            if e["pid"] == fd["pid"]
        ),
    )
    assert first == fd["frame"]
    e = r["panes"]["bytetrack"]["events"][0]
    assert e["line"].startswith("ByteTrack: ")
    assert e["why"]


def test_short_why_lines() -> None:
    x = {
        "reason": "expired",
        "from_id": 7,
        "to_id": 19,
        "from_expired": {"frames_lost": 31, "buffer": 30},
        "candidates": [],
    }
    assert short_why(x, "bytetrack") == "ByteTrack: memory expired after 1.0 s hidden"
    x = {
        "reason": "cost_above_threshold",
        "from_id": 7,
        "to_id": 19,
        "candidates": [{"track_id": 7, "iou": 0.12}],
    }
    assert short_why(x, "ours").startswith("Ours: still remembered #7")


def test_pillar_hides_a_person_behind_it() -> None:
    w = path_walker(1, [(480.0, 60.0), (480.0, 520.0)], 0.0, 70.0, None)
    sc = Scenario(people=1, hide_s=0.0, seconds=8.0)
    free = render(sc, 1, walkers=[w])
    w2 = path_walker(1, [(480.0, 60.0), (480.0, 520.0)], 0.0, 70.0, None)
    blocked = render(sc, 1, walkers=[w2], pillars=[(457.0, 250.0, 46.0, 150.0)])
    vis_free = [v for fr in free for *_, v in fr.truth]
    vis_blk = [v for fr in blocked for *_, v in fr.truth]
    assert min(vis_free) > 0.9
    assert min(vis_blk) < 0.25  # fully hidden for a while


def test_break_scene_is_deterministic_and_hides_the_chosen_person() -> None:
    spec = BreakSpec(seed=11, hide_s=2.0, hide_pid=1, pillars=[(470, 340)], groups=[(640, 260)])
    a, b = break_run(spec), break_run(spec)
    assert json.dumps(a) == json.dumps(b)
    hidden = [t["f"] for t in a["truth"] if 1 in t["hidden"]]
    assert len(hidden) == round(2.0 * 30) + 12  # hide + ramps
    assert {p["pid"] for p in a["people"]} >= {1, 2, 3, 4}
    assert a["panes"]["ours"]["true_count"] == len(a["true_crossings"])


def test_break_route_validates() -> None:
    r = client.post(
        "/api/break/run", json={"seed": 3, "hide_s": 1.0, "path": [[100, 40], [700, 520]]}
    )
    assert r.status_code == 200
    assert client.post("/api/break/run", json={"hide_s": 99}).status_code == 422


def test_path_walker_reaches_the_gate_inside_its_hide() -> None:
    w = path_walker(1, [(100.0, 40.0), (800.0, 520.0)], 0.0, 70.0, 1.0)
    fr = render(Scenario(people=1, hide_s=1.0, seconds=14.0), 2, walkers=[w])
    ys = {f.f: y + h for f in fr for _, _x, y, _w, h, _v in f.truth}
    hid = [f.f for f in fr if 1 in f.hidden]
    assert hid
    assert min(abs(ys[f] - 300) for f in hid) < 10
    assert np.isfinite(list(ys.values())).all()


def test_match_crossings_missed_and_extra() -> None:
    from serve.stress_live import match_crossings

    true_by = [(100, 1, 1), (200, 2, 1), (300, 3, -1)]
    # person 1 counted on time, person 2 counted by a track owned by someone else, person 3 missed;
    # plus one crossing that matches nothing
    trk_by = [(104, 1, 1), (201, 5, 1), (305, 3, 1)]
    missed, extra = match_crossings(true_by, trk_by, 30.0)
    assert missed == [(200, 2, 1), (300, 3, -1)]
    assert extra == 2
    assert match_crossings(true_by, [(100 + 61, 1, 1)], 30.0)[0][0] == (100, 1, 1)  # > 2 s late

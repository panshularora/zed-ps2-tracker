"""Guided-replay annotations (scripts/demo/guide.py): gate events, gaps, renumber, moments."""

from __future__ import annotations

from scripts.demo.guide import annotate


def _data() -> dict:
    # one person walks down through y=540 (h50); baseline renumbers 1 -> 5 at frame 10, ours keeps 1
    meta = {"width": 1920, "height": 1080, "src_fps": 5.0, "stride": 1, "kind": "illustrative"}
    base, ours = {}, {}
    for f in range(1, 21):
        y = 200 + 30 * f  # bottom = y + 200 crosses 540 near f=5
        bid = 1 if f < 10 else 5
        base[str(f)] = [[bid, 900, y, 100, 200]]
        if f not in (14, 15, 16, 17, 18):  # ours loses them for 1.2 s
            ours[str(f)] = [[1, 900, y, 100, 200]]
    dis = [
        {
            "frame": 10,
            "kept": "ours",
            "ours_id": 1,
            "baseline_from": 1,
            "baseline_to": 5,
            "box": [900, 500, 100, 200],
        }
    ]
    return {
        "meta": meta,
        "runs": {"baseline": {"frames": base}, "ours": {"frames": ours}},
        "disagreements": dis,
    }


def test_annotate() -> None:
    g = annotate(_data())
    assert g["gate"]["name"] == "h50"
    ours = g["events"]["ours"]
    assert [e["t"] for e in ours if e["t"] != "cross"] == ["lost", "back"]
    assert {"t": "renumber", "f": 10, "from": 1, "to": 5} in g["events"]["baseline"]
    assert sum(e["t"] == "cross" for e in ours) == 1
    (m,) = g["moments"]
    assert m["f"] == 10
    assert m["changed"] == "baseline"
    assert m["kept"] == "ours"
    assert m["people"][0]["ids"]["baseline"] == {"before": 1, "after": 5}
    assert m["counted_twice"] is False  # #5 never crosses: no double count claimed
    assert g["kind"] == "illustrative"

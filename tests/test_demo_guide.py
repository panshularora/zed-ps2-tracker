"""Guided-replay annotations (scripts/demo/guide.py): gate events, gaps, and the moment rule:
every moment where at least one tracker changed a person's number (no cap)."""

from __future__ import annotations

from typing import Any

from scripts.demo.guide import RULE, annotate, renumbers, rows_of


def _walk(f: int) -> list[float]:
    return [900, 200 + 30 * f, 100, 200]  # walks straight down the frame


def _data(base_changes: dict[int, int], ours_changes: dict[int, int]) -> dict[str, Any]:
    """One walking person; `{frame: new_id}` = from that frame the tracker uses a new number,
    after a 2-frame gap (lost then renumbered)."""
    meta = {"width": 1920, "height": 1080, "src_fps": 5.0, "stride": 1, "kind": "illustrative"}

    def run(changes: dict[int, int]) -> dict[str, Any]:
        frames: dict[str, list[list[float]]] = {}
        tid = 1
        for f in range(1, 31):
            if f in changes:
                tid = changes[f]
            if any(f in (c - 1, c - 2) for c in changes):
                continue  # short gap before each change
            frames[str(f)] = [[tid, *_walk(f)]]
        return {"frames": frames}

    return {
        "meta": meta,
        "runs": {"baseline": run(base_changes), "ours": run(ours_changes)},
        "disagreements": [],
    }


def test_every_change_is_a_moment_and_both_is_merged() -> None:
    g = annotate(_data({10: 5, 20: 6, 26: 7}, {10: 3}))
    assert g["rule"] == RULE
    ms = g["moments"]
    assert [m["f"] for m in ms] == [10, 20, 26]  # not capped, one per change
    assert ms[0]["changed"] == {"baseline": True, "ours": True}  # both at the same spot
    assert ms[1]["changed"] == {"baseline": True, "ours": False}
    assert ms[0]["people"][0]["ids"]["ours"] == {"before": 1, "after": 3}
    assert ms[1]["people"][0]["ids"]["ours"] == {"before": 3, "after": 3}  # kept
    assert ms[1]["counted_twice"] == {"baseline": False, "ours": False}


def test_no_change_no_moment() -> None:
    g = annotate(_data({}, {}))
    assert g["moments"] == []
    assert sum(e["t"] == "cross" for e in g["events"]["ours"]) == 1
    assert g["gate"]["name"] == "walk"  # from motion: the walker goes straight down
    assert g["gate"]["a"][1] == g["gate"]["b"][1]  # so the line is horizontal


def test_renumber_needs_same_place() -> None:
    d = _data({10: 5}, {})
    rows = rows_of(d["runs"]["baseline"]["frames"])
    assert [(e["from"], e["to"]) for e in renumbers(rows, 5.0)] == [(1, 5)]
    far = rows.copy()
    far[far[:, 1] == 5, 2] += 800  # new number appears elsewhere: a different person
    assert renumbers(far, 5.0) == []


def test_no_crossing_counted_in_the_first_second():
    """A box present on frame 1 that slides across the line in the first second is not counted."""
    import json
    from pathlib import Path

    from scripts.demo.guide import SETTLE_S

    root = Path(__file__).resolve().parents[1]
    for cid in ("subway_turnstile", "shibuya_crossing"):
        d = json.loads((root / f"web/public/scenarios/{cid}.json").read_text())
        fps = d["meta"]["src_fps"] / d["meta"]["stride"]
        g = annotate(d)
        assert g == d["guide"]  # committed guide is what the code produces
        for run in ("baseline", "ours"):
            crosses = [e for e in g["events"][run] if e["t"] == "cross"]
            assert crosses
            assert min(e["f"] for e in crosses) >= 1 + SETTLE_S * fps

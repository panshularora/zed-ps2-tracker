import numpy as np

from eval.operator_demo import analyse


def _walk(pid: int, f0: int, y0: float, y1: float, n: int = 20) -> list[list[float]]:
    # MOT rows: frame,id,x,y,w,h,conf,cls,vis ; foot point = y + h
    return [[f0 + k, pid, 100.0, y0 + (y1 - y0) * k / (n - 1) - 50.0, 20.0, 50.0, 1, 1, 1.0]
            for k in range(n)]  # fmt: skip


def test_only_suggestions_matching_a_missed_true_crossing_are_real() -> None:
    # two people walk down across y=100 (frames ~10 and ~60); the tracker counted the first only
    gt = np.array(_walk(1, 1, 50, 150) + _walk(2, 51, 50, 150))
    result = {
        "video": {"width": 200, "height": 200},
        "gate": {"a": [0, 100], "b": [200, 100], "hysteresis_px": 5.0},
        "events": [
            {"kind": "gate", "type": "crossing", "frame": 10, "track_id": 1, "direction": d}
            for d in [_dir(gt)]
        ] + [
            {"kind": "gate", "type": "recovered_crossing", "frame": 61, "track_id": 9,
             "from_track_id": 8, "direction": _dir(gt), "pair_score": 0.2},
            {"kind": "gate", "type": "recovered_crossing", "frame": 200, "track_id": 7,
             "from_track_id": 6, "direction": _dir(gt), "pair_score": 0.9},
        ],
    }  # fmt: skip
    r = analyse(result, gt, 200, 200, 15)
    assert sum(r["truth"].values()) == 2
    assert r["plain_matched_to_truth"] == "1 of 1"
    assert r["real_keys"] == ["61:8:9"]
    assert r["real_score_ranks"] == [2]


def _dir(gt: np.ndarray) -> str:
    from eval.gate_events import Gate, crossings

    return crossings(gt, Gate("g", (0, 100), (200, 100)), 5.0)[0].direction

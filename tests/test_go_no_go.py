"""Go/no-go helpers: corruption model, CLEAR-style switch frames, flagged windows."""

from types import SimpleNamespace

import numpy as np

from eval.corrupt import Corruption, corrupt_sequence
from eval.go_no_go import idsw_frames, in_windows


def _gt(n_frames: int = 200, vis: float = 1.0) -> np.ndarray:
    rows = []
    for f in range(1, n_frames + 1):
        for i in range(10):
            rows.append([f, i + 1, 50.0 * i, 100.0, 30.0, 80.0, 1, 1, vis])
    return np.array(rows)


def test_corruption_deterministic_and_visibility_dependent() -> None:
    info = SimpleNamespace(im_width=1920, im_height=1080)
    c = Corruption(fp_rate=0.0)
    a = corrupt_sequence(_gt(), info, "MOT17", np.random.default_rng(0), c)
    b = corrupt_sequence(_gt(), info, "MOT17", np.random.default_rng(0), c)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    n_vis = len(a[0])
    n_occ = len(corrupt_sequence(_gt(vis=0.1), info, "MOT17", np.random.default_rng(0), c)[0])
    assert n_occ < 0.7 * n_vis  # p_drop 0.05 vs 0.5
    assert a[2].min() >= 0.05
    assert a[2].max() <= 1.0
    fp = corrupt_sequence(_gt(), info, "MOT17", np.random.default_rng(0), Corruption(fp_rate=2.0))
    assert len(fp[0]) > n_vis
    assert np.all(np.diff(fp[0]) >= 0)


def test_idsw_frames_and_windows() -> None:
    gt = np.array([[f, 1, 0, 0, 10, 10, 1, 1, 1.0] for f in range(1, 11)])
    trk = np.array([[f, 7 if f < 6 else 8, 0, 0, 10, 10, 1, -1, -1, -1] for f in range(1, 11)])
    sw = idsw_frames(gt, trk, "MOT17")
    np.testing.assert_array_equal(sw, [6])
    assert in_windows(sw, [14]) == 1  # |6 - 14| = 8
    assert in_windows(sw, [15]) == 0
    assert in_windows(sw, []) == 0

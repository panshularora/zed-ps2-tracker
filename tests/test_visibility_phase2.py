"""Phase-2 visibility: softened cost, occluded buffer, score fusion, eval helpers."""

import numpy as np
import pytest

from eval.visibility_eval import auroc, average_precision, metrics
from tracker.bytetrack import ByteTrackLike
from tracker.config import config_from_dict
from tracker.cost import ioa_matrix, iou_matrix, visibility_softened_iou_cost
from tracker.types import Detections
from tracker.visibility import DepthOrderVisibility


def test_softened_cost() -> None:
    trk = np.array([[0, 0, 10, 20.0]])
    det = np.array([[0, 0, 10, 10.0]])  # the visible upper half of the predicted box
    iou = iou_matrix(trk, det)[0, 0]
    assert iou == pytest.approx(0.5)
    assert ioa_matrix(trk, det)[0, 0] == pytest.approx(1.0)
    plain = visibility_softened_iou_cost(trk, det, np.array([0.5]), 0.0)
    np.testing.assert_allclose(plain, 1 - iou)
    soft = visibility_softened_iou_cost(trk, det, np.array([0.5]), 1.0)  # w = 0.5
    assert soft[0, 0] == pytest.approx(1 - (0.5 * 0.5 + 0.5 * 1.0))
    full = visibility_softened_iou_cost(trk, det, np.array([1.0]), 1.0)  # w = 0: unchanged
    np.testing.assert_allclose(full, plain)


def test_score_fusion() -> None:
    est = DepthOrderVisibility((100, 100), origin=0.0, score_fusion="product")
    v = est(np.array([[10, 10, 20, 40.0]]), np.array([0.6]))
    assert v[0] == pytest.approx(0.6)
    with pytest.raises(ValueError, match="score_fusion"):
        DepthOrderVisibility(score_fusion="max")


def _cfg(scale: float, est: str = "constant") -> dict:
    return {
        "visibility": {"estimator": est, "occluded_buffer_scale": scale, "occluded_thresh": 0.5},
        "lifecycle": {"track_buffer": 5},
    }


class _Half:
    def __call__(self, xyxy: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
        return np.full(len(np.asarray(xyxy).reshape(-1, 4)), 0.3)


def _lives(scale: float, vis: object) -> int:
    trk = ByteTrackLike(config_from_dict(_cfg(scale)), frame_rate=30, visibility=vis)  # type: ignore
    box = Detections(np.array([[100, 100, 40, 100.0]]), np.array([0.9]))
    for _ in range(5):
        trk.update(box)
    empty = Detections(np.zeros((0, 4)), np.zeros(0))
    for k in range(1, 40):
        trk.update(empty)
        if not trk.tracks:
            return k
    return 40


def test_occluded_buffer_scale() -> None:
    base = _lives(1.0, _Half())
    assert _lives(3.0, _Half()) > base
    # a visible track keeps the normal buffer even with the scale set
    assert _lives(3.0, None) == _lives(1.0, None)


def test_auroc_and_ap() -> None:
    lab = np.array([1, 1, 0, 0])
    assert auroc(lab, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert auroc(lab, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert auroc(lab, np.ones(4)) == 0.5
    assert np.isnan(auroc(np.zeros(3), np.ones(3)))
    assert average_precision(lab, np.array([0.9, 0.1, 0.8, 0.2])) == pytest.approx((1 + 2 / 4) / 2)
    m = metrics(np.array([0.2, 0.9, 1.0]), np.array([0.1, 0.8, 1.0]))
    assert m["mae"] == pytest.approx(0.2 / 3)
    assert m["auroc_vis_lt_0.5"] == 1.0


def test_occluded_buffer_scale_with_removal_lag() -> None:
    """The BoT-SORT parity path (removal_lag) honours the occluded buffer too."""

    def lives(scale: float) -> int:
        cfg = _cfg(scale)
        cfg["lifecycle"]["removal_lag"] = True
        trk = ByteTrackLike(config_from_dict(cfg), frame_rate=30, visibility=_Half())  # type: ignore[arg-type]
        box = Detections(np.array([[100, 100, 40, 100.0]]), np.array([0.9]))
        for _ in range(5):
            trk.update(box)
        empty = Detections(np.zeros((0, 4)), np.zeros(0))
        for k in range(1, 40):
            trk.update(empty)
            if not trk.tracks:
                return k
        return 40

    assert lives(3.0) > lives(1.0)

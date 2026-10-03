import numpy as np
import pytest

from tracker.cost import (
    cosine_cost,
    fuse_iou_appearance,
    fuse_score,
    gate_cost,
    iou_cost,
    iou_matrix,
)


def test_iou_known_values() -> None:
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    iou = iou_matrix(a, b)
    assert iou.shape == (2, 3)
    np.testing.assert_allclose(iou[0], [1.0, 50 / 150, 0.0])
    np.testing.assert_allclose(iou_cost(a, b), 1 - iou)


def test_iou_symmetric_and_bounded() -> None:
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 100, (20, 2))
    wh = rng.uniform(1, 50, (20, 2))
    boxes = np.hstack([xy, xy + wh])
    iou = iou_matrix(boxes, boxes)
    np.testing.assert_allclose(iou, iou.T)
    np.testing.assert_allclose(np.diag(iou), 1.0)
    assert iou.min() >= 0
    assert iou.max() <= 1 + 1e-12


@pytest.mark.parametrize(("n", "m"), [(0, 3), (3, 0), (0, 0)])
def test_iou_empty(n: int, m: int) -> None:
    out = iou_matrix(np.zeros((n, 4)), np.zeros((m, 4)))
    assert out.shape == (n, m)


def test_iou_degenerate_box_is_zero() -> None:
    iou = iou_matrix(np.array([[5, 5, 5, 5]], float), np.array([[0, 0, 10, 10]], float))
    assert iou[0, 0] == 0.0


def test_fuse_score() -> None:
    cost = np.array([[0.2, 0.6]])
    out = fuse_score(cost, np.array([0.5, 1.0]))
    np.testing.assert_allclose(out, [[1 - 0.8 * 0.5, 1 - 0.4]])
    assert fuse_score(np.zeros((0, 2)), np.ones(2)).shape == (0, 2)


def test_cosine_cost() -> None:
    t = np.array([[1, 0], [0, 2], [0, 0]], float)
    d = np.array([[3, 0], [0, -1]], float)
    c = cosine_cost(t, d)
    np.testing.assert_allclose(c, [[0, 1], [1, 2], [1, 1]])
    assert cosine_cost(np.zeros((0, 4)), np.zeros((2, 4))).shape == (0, 2)
    with pytest.raises(ValueError, match="dims"):
        cosine_cost(np.ones((1, 3)), np.ones((1, 4)))


def test_fuse_iou_appearance_gates() -> None:
    iou_c = np.array([[0.3, 0.9, 0.45]])
    app_c = np.array([[0.1, 0.1, 0.8]])  # halved -> 0.05, 0.05, 0.4
    out = fuse_iou_appearance(iou_c, app_c, proximity_thresh=0.5, appearance_thresh=0.25)
    # pair 0: close + similar -> appearance wins; pair 1: too far -> iou only;
    # pair 2: appearance too dissimilar -> iou only
    np.testing.assert_allclose(out, [[0.05, 0.9, 0.45]])
    motion = np.array([[0.02, 0.95, 0.5]])
    out2 = fuse_iou_appearance(
        iou_c, app_c, proximity_thresh=0.5, appearance_thresh=0.25, motion_cost=motion
    )
    np.testing.assert_allclose(out2, [[0.02, 0.95, 0.5]])


def test_gate_cost_copies() -> None:
    c = np.zeros((2, 2))
    g = gate_cost(c, np.eye(2, dtype=bool))
    assert np.isinf(g[0, 0])
    assert c[0, 0] == 0

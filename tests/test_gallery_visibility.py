import numpy as np
import pytest

from tracker.gallery import EmaAppearance
from tracker.visibility import ConstantVisibility, DepthOrderVisibility, build_visibility


def test_ema_update_normalised_and_moves() -> None:
    m = EmaAppearance(np.array([1, 0], np.float32), alpha=0.5)
    assert m.update(np.array([0, 1], np.float32))
    e = m.embedding
    assert np.linalg.norm(e) == pytest.approx(1.0)
    np.testing.assert_allclose(e, [np.sqrt(0.5), np.sqrt(0.5)], rtol=1e-6)


def test_ema_visibility_gate() -> None:
    m = EmaAppearance(np.array([1, 0], np.float32), alpha=0.5, min_visibility=0.5)
    assert not m.update(np.array([0, 1], np.float32), visibility=0.2)
    np.testing.assert_allclose(m.embedding, [1, 0])
    assert m.n_gated == 1
    assert m.update(np.array([0, 1], np.float32), visibility=0.9)


def test_ema_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        EmaAppearance(np.ones(2, np.float32), alpha=1.0)


def test_constant_visibility() -> None:
    v = ConstantVisibility()(np.zeros((3, 4)))
    np.testing.assert_array_equal(v, [1, 1, 1])
    assert ConstantVisibility()(np.zeros((0, 4))).shape == (0,)
    assert isinstance(build_visibility("constant"), ConstantVisibility)
    assert isinstance(build_visibility("depth_order"), DepthOrderVisibility)
    with pytest.raises(ValueError, match="unknown"):
        build_visibility("x")


def raster_visibility(b: np.ndarray, occ: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Pixel rasterisation reference (implementation brief's vis_oracle_check, 0-based here)."""
    W, H = size
    out = []
    for i in range(len(b)):
        x1, y1, x2, y2 = b[i]
        w, h = round(x2 - x1), round(y2 - y1)
        xs, ys = np.arange(w) + x1 + 0.5, np.arange(h) + y1 + 0.5
        m = ~((ys[:, None] >= 0) & (ys[:, None] < H) & (xs[None, :] >= 0) & (xs[None, :] < W))
        for j in np.nonzero(occ)[0]:
            if j == i or b[j, 3] <= b[i, 3]:
                continue
            m |= (
                (xs[None, :] >= b[j, 0])
                & (xs[None, :] < b[j, 2])
                & (ys[:, None] >= b[j, 1])
                & (ys[:, None] < b[j, 3])
            )
        out.append(1 - m.mean())
    return np.array(out)


def test_depth_order_rules() -> None:
    est = DepthOrderVisibility((100, 100), origin=0.0, grid=20)
    front = [10, 10, 30, 60]  # bottom at 60: in front
    back = [20, 0, 40, 50]  # bottom at 50: behind, right half 20..30 x 10..50 covered
    v = est(np.array([front, back], float))
    assert v[0] == 1.0
    assert v[1] == pytest.approx(1 - (10 * 40) / (20 * 50))
    # the border crops a box half outside the image
    assert est(np.array([[-10, 0, 10, 10.0]]))[0] == pytest.approx(0.5)
    # low-score detections do not occlude
    gated = DepthOrderVisibility((100, 100), origin=0.0, grid=20, occluder_min_score=0.5)
    v2 = gated(np.array([front, back], float), np.array([0.3, 0.9]))
    assert v2[1] == 1.0
    assert est(np.zeros((0, 4))).shape == (0,)


def test_depth_order_matches_rasterisation() -> None:
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 400, (40, 2))
    wh = rng.uniform(20, 120, (40, 2)) * [0.5, 1.5]
    b = np.concatenate([xy, xy + wh], 1)
    occ = rng.uniform(size=40) > 0.3
    ref = raster_visibility(b, occ, (420, 450))
    got = DepthOrderVisibility((420, 450), origin=0.0, grid=32)(b, occluder_mask=occ)
    assert np.abs(got - ref).mean() < 0.01

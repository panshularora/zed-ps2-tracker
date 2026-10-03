import numpy as np
import pytest

from tracker.gallery import EmaAppearance
from tracker.visibility import ConstantVisibility, build_visibility


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
    with pytest.raises(NotImplementedError):
        build_visibility("depth_order")
    with pytest.raises(ValueError, match="unknown"):
        build_visibility("x")

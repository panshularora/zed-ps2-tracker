import numpy as np
import pytest

from tracker.kalman import CHI2INV95, KalmanConfig, KalmanFilterXYWH


@pytest.fixture
def kf() -> KalmanFilterXYWH:
    return KalmanFilterXYWH()


def test_initiate_zero_velocity_and_size_scaled_cov(kf: KalmanFilterXYWH) -> None:
    z = np.array([100.0, 200.0, 40.0, 100.0])
    mean, cov = kf.initiate(z)
    np.testing.assert_allclose(mean[:4], z)
    np.testing.assert_allclose(mean[4:], 0.0)
    wp, wv = 1 / 20, 1 / 160
    expected = np.square([2 * wp * 40, 2 * wp * 100, 2 * wp * 40, 2 * wp * 100,
                          10 * wv * 40, 10 * wv * 100, 10 * wv * 40, 10 * wv * 100])  # fmt: skip
    np.testing.assert_allclose(np.diag(cov), expected)
    assert np.allclose(cov, np.diag(np.diag(cov)))


def test_predict_constant_velocity(kf: KalmanFilterXYWH) -> None:
    mean, cov = kf.initiate(np.array([10.0, 20.0, 5.0, 10.0]))
    mean[4:] = [1.0, -2.0, 0.5, 0.0]
    m2, c2 = kf.predict(mean, cov)
    np.testing.assert_allclose(m2[:4], [11.0, 18.0, 5.5, 10.0])
    np.testing.assert_allclose(m2[4:], mean[4:])
    # uncertainty grows and stays symmetric PSD
    assert np.all(np.diag(c2) >= np.diag(cov))
    np.testing.assert_allclose(c2, c2.T)
    assert np.all(np.linalg.eigvalsh(c2) > 0)


def test_multi_predict_matches_single(kf: KalmanFilterXYWH) -> None:
    rng = np.random.default_rng(0)
    means, covs = [], []
    for _ in range(5):
        m, c = kf.initiate(rng.uniform(10, 200, 4))
        m[4:] = rng.normal(size=4)
        means.append(m)
        covs.append(c)
    mm, cc = kf.multi_predict(np.stack(means), np.stack(covs))
    for i in range(5):
        m1, c1 = kf.predict(means[i], covs[i])
        np.testing.assert_allclose(mm[i], m1)
        np.testing.assert_allclose(cc[i], c1)


def test_multi_predict_empty(kf: KalmanFilterXYWH) -> None:
    m, c = kf.multi_predict(np.zeros((0, 8)), np.zeros((0, 8, 8)))
    assert m.shape == (0, 8)
    assert c.shape == (0, 8, 8)


def test_update_pulls_towards_measurement_and_shrinks_cov(kf: KalmanFilterXYWH) -> None:
    mean, cov = kf.initiate(np.array([100.0, 100.0, 50.0, 100.0]))
    mean, cov = kf.predict(mean, cov)
    z = np.array([110.0, 95.0, 52.0, 104.0])
    m2, c2 = kf.update(mean, cov, z)
    # posterior lies strictly between prior and measurement on each measured dim
    for i in range(4):
        lo, hi = sorted([mean[i], z[i]])
        assert lo < m2[i] < hi
    assert np.all(np.diag(c2)[:4] < np.diag(cov)[:4])
    np.testing.assert_allclose(c2, c2.T, atol=1e-9)


def test_update_matches_textbook_formula(kf: KalmanFilterXYWH) -> None:
    mean, cov = kf.initiate(np.array([50.0, 60.0, 20.0, 40.0]))
    mean, cov = kf.predict(mean, cov)
    z = np.array([52.0, 59.0, 21.0, 41.0])
    h = kf.observation
    r = kf.measurement_noise(mean)
    s = h @ cov @ h.T + r
    k = cov @ h.T @ np.linalg.inv(s)
    exp_mean = mean + k @ (z - h @ mean)
    exp_cov = (np.eye(8) - k @ h) @ cov
    m2, c2 = kf.update(mean, cov, z)
    np.testing.assert_allclose(m2, exp_mean, rtol=1e-9)
    np.testing.assert_allclose(c2, exp_cov, rtol=1e-7, atol=1e-9)


def test_converges_to_true_velocity(kf: KalmanFilterXYWH) -> None:
    mean, cov = kf.initiate(np.array([0.0, 0.0, 30.0, 80.0]))
    for t in range(1, 40):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([3.0 * t, -1.0 * t, 30.0, 80.0]))
    np.testing.assert_allclose(mean[4:6], [3.0, -1.0], atol=0.05)


def test_gating_distance(kf: KalmanFilterXYWH) -> None:
    mean, cov = kf.initiate(np.array([100.0, 100.0, 40.0, 100.0]))
    d = kf.gating_distance(mean, cov, np.array([[100.0, 100.0, 40.0, 100.0], [400, 400, 40, 100]]))
    assert d[0] == pytest.approx(0.0)
    assert d[1] > CHI2INV95[4]


def test_custom_noise_weights() -> None:
    kf = KalmanFilterXYWH(KalmanConfig(std_weight_position=0.1, std_weight_velocity=0.01))
    _, cov = kf.initiate(np.array([0.0, 0.0, 10.0, 10.0]))
    assert cov[0, 0] == pytest.approx((2 * 0.1 * 10) ** 2)

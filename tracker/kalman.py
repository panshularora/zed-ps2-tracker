"""Constant-velocity Kalman filter over boxes in ``xywh`` space (BoT-SORT style).

State ``x = [cx, cy, w, h, vcx, vcy, vw, vh]``; measurement ``z = [cx, cy, w, h]``.

Unlike the SORT/DeepSORT ``xyah`` parameterisation, width is a state of its own, so its
uncertainty scales with the box width rather than the height (Aharon et al., BoT-SORT, 2022).
All noise standard deviations are proportional to the current box size, so a filter tuned on
small far-away pedestrians behaves the same for large close ones.

Equations (F: constant-velocity transition with dt = 1 frame, H = [I_4 | 0]):

    predict:  x⁻ = F x,              P⁻ = F P Fᵀ + Q(x)
    project:  ẑ  = H x⁻,             S  = H P⁻ Hᵀ + R(x⁻)
    update:   K  = P⁻ Hᵀ S⁻¹,        x  = x⁻ + K (z − ẑ),   P = P⁻ − K S Kᵀ
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from tracker.types import FloatArray

#: 0.95 quantile of the chi-square distribution with N degrees of freedom (as in DeepSORT).
CHI2INV95: dict[int, float] = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}

NDIM = 4


@dataclass(frozen=True)
class KalmanConfig:
    std_weight_position: float = 1.0 / 20.0
    std_weight_velocity: float = 1.0 / 160.0


class KalmanFilterXYWH:
    """Stateless filter: the caller owns ``(mean, covariance)``; methods are pure functions."""

    def __init__(self, config: KalmanConfig | None = None) -> None:
        cfg = config or KalmanConfig()
        self._wp = cfg.std_weight_position
        self._wv = cfg.std_weight_velocity
        self._F = np.eye(2 * NDIM)
        for i in range(NDIM):
            self._F[i, NDIM + i] = 1.0
        self._H = np.eye(NDIM, 2 * NDIM)

    @property
    def transition(self) -> FloatArray:
        return self._F.copy()

    @property
    def observation(self) -> FloatArray:
        return self._H.copy()

    # ------------------------------------------------------------------ noise models
    def _pos_std(self, w: float, h: float) -> FloatArray:
        return np.array([self._wp * w, self._wp * h, self._wp * w, self._wp * h])

    def _vel_std(self, w: float, h: float) -> FloatArray:
        return np.array([self._wv * w, self._wv * h, self._wv * w, self._wv * h])

    # ------------------------------------------------------------------ filter steps
    def initiate(self, measurement: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Create a track from an unassociated ``xywh`` measurement (zero initial velocity)."""
        z = np.asarray(measurement, dtype=np.float64).reshape(NDIM)
        mean = np.concatenate([z, np.zeros(NDIM)])
        w, h = float(z[2]), float(z[3])
        std = np.concatenate([2.0 * self._pos_std(w, h), 10.0 * self._vel_std(w, h)])
        return mean, np.diag(np.square(std))

    def process_noise(self, mean: FloatArray) -> FloatArray:
        w, h = float(mean[2]), float(mean[3])
        std = np.concatenate([self._pos_std(w, h), self._vel_std(w, h)])
        return np.diag(np.square(std))

    def measurement_noise(self, mean: FloatArray) -> FloatArray:
        w, h = float(mean[2]), float(mean[3])
        return np.diag(np.square(self._pos_std(w, h)))

    def predict(self, mean: FloatArray, covariance: FloatArray) -> tuple[FloatArray, FloatArray]:
        """One-frame prediction for a single track."""
        q = self.process_noise(mean)
        new_mean = self._F @ mean
        new_cov = self._F @ covariance @ self._F.T + q
        return new_mean, new_cov

    def multi_predict(
        self, means: FloatArray, covariances: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Vectorised :meth:`predict` for ``(N, 8)`` means and ``(N, 8, 8)`` covariances."""
        means = np.asarray(means, dtype=np.float64).reshape(-1, 2 * NDIM)
        covariances = np.asarray(covariances, dtype=np.float64).reshape(-1, 2 * NDIM, 2 * NDIM)
        if means.shape[0] == 0:
            return means.copy(), covariances.copy()
        w, h = means[:, 2], means[:, 3]
        std = np.stack(
            [self._wp * w, self._wp * h, self._wp * w, self._wp * h,
             self._wv * w, self._wv * h, self._wv * w, self._wv * h],
            axis=1,
        )  # fmt: skip
        q = np.zeros_like(covariances)
        idx = np.arange(2 * NDIM)
        q[:, idx, idx] = np.square(std)
        new_means = means @ self._F.T
        new_covs = self._F[None] @ covariances @ self._F.T[None] + q
        return new_means, new_covs

    def project(self, mean: FloatArray, covariance: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Map state distribution to measurement space: returns ``(ẑ, S)``."""
        r = self.measurement_noise(mean)
        return self._H @ mean, self._H @ covariance @ self._H.T + r

    def update(
        self, mean: FloatArray, covariance: FloatArray, measurement: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Correct the predicted state with an associated ``xywh`` measurement."""
        z = np.asarray(measurement, dtype=np.float64).reshape(NDIM)
        z_hat, s = self.project(mean, covariance)
        chol = scipy.linalg.cho_factor(s, lower=True, check_finite=False)
        # K = P Hᵀ S⁻¹, solved as S Kᵀ = H P (S is symmetric positive definite)
        gain = scipy.linalg.cho_solve(chol, (covariance @ self._H.T).T, check_finite=False).T
        innovation = z - z_hat
        new_mean = mean + gain @ innovation
        new_cov = covariance - gain @ s @ gain.T
        return new_mean, new_cov

    def gating_distance(
        self, mean: FloatArray, covariance: FloatArray, measurements: FloatArray
    ) -> FloatArray:
        """Squared Mahalanobis distance between the state and ``(N, 4)`` ``xywh`` measurements.

        Compare against :data:`CHI2INV95` ``[4]`` to gate implausible associations.
        """
        z_hat, s = self.project(mean, covariance)
        d = np.asarray(measurements, dtype=np.float64).reshape(-1, NDIM) - z_hat
        chol = np.linalg.cholesky(s)
        y = scipy.linalg.solve_triangular(chol, d.T, lower=True, check_finite=False)
        return np.asarray(np.sum(y * y, axis=0), dtype=np.float64)

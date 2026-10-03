"""SORT-family Kalman filter with OC-SORT's Observation-centric Re-Update (ORU).

Ported from OC-SORT ``trackers/ocsort_tracker/kalmanfilter.py`` (``KalmanFilterNew``: the
filterpy ``predict``/``update`` equations plus ``freeze``/``unfreeze``), commit 8462e7e,
MIT, Copyright (c) 2021 Yifu Zhang. Deep OC-SORT's variant (``integrated_ocsort_embedding/
kalmanfilter.py``, commit 6bb51d0, MIT, Copyright (c) 2023 Gerard Maggiolino) anchors the
virtual trajectory on the (CMC-corrected) last measurement and adds ``apply_affine_correction``;
Hybrid-SORT's (``hybrid_sort_tracker/kalmanfilter_score_new.py``, commit 396f8d3, MIT,
Copyright (c) 2021 Yifu Zhang) carries the detection score as a fifth measured quantity.

ORU: when a track that was coasting gets an observation again, roll the filter back to the
state saved at the first missed frame, then re-run predict/update along a straight line between
the last two real observations (the virtual trajectory), then apply the real update. In the
reference code the last virtual box *is* the new observation, so the new observation is applied
twice; that is kept here.

Only the parts the trackers use are ported (no control input, no fading memory, constant F/H/Q/R).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Layout = Literal["xysr", "xyscr"]
"""Measurement layout: ``[x, y, s, r]`` (SORT/OC-SORT) or ``[x, y, s, score, r]`` (Hybrid)."""


class OruKalmanFilter:
    def __init__(self, dim_x: int, dim_z: int, layout: Layout = "xysr") -> None:
        self.dim_x, self.dim_z, self.layout = dim_x, dim_z, layout
        # filterpy defaults
        self.x: FloatArray = np.zeros((dim_x, 1))
        self.P: FloatArray = np.eye(dim_x)
        self.Q: FloatArray = np.eye(dim_x)
        self.F: FloatArray = np.eye(dim_x)
        self.H: FloatArray = np.zeros((dim_z, dim_x))
        self.R: FloatArray = np.eye(dim_z)
        self._I = np.eye(dim_x)
        self.oru = True
        self.anchor: Literal["history", "last_measurement"] = "history"
        self.history_obs: list[FloatArray | None] = []
        self.observed = False
        self.last_measurement: FloatArray | None = None
        self._saved: dict[str, object] | None = None

    # ---------------------------------------------------------------- filterpy core
    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _correct(self, z: FloatArray) -> None:
        z = np.asarray(z, np.float64).reshape(self.dim_z, 1)
        y = z - self.H @ self.x
        PHT = self.P @ self.H.T
        S = self.H @ PHT + self.R
        K = PHT @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = self._I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form, as filterpy

    # ---------------------------------------------------------------- ORU
    def update(self, z: FloatArray | None) -> None:
        self.history_obs.append(z)
        if z is None:
            if self.observed and self.oru:
                self._freeze()
            self.observed = False
            return
        if not self.observed and self.oru:
            self._unfreeze()
        self.observed = True
        self._correct(z)
        self.last_measurement = np.asarray(z, np.float64).reshape(self.dim_z, 1).copy()

    def _freeze(self) -> None:
        lm = self.last_measurement
        self._saved = {
            "x": self.x.copy(),
            "P": self.P.copy(),
            "history_obs": list(self.history_obs),
            "last_measurement": None if lm is None else lm.copy(),
        }

    def _box(self, z: FloatArray) -> tuple[float, float, float, float, float]:
        v = np.asarray(z, np.float64).reshape(-1)
        if self.layout == "xysr":
            x, y, s, r = (float(a) for a in v[:4])
            c = 0.0
        else:
            x, y, s, c, r = (float(a) for a in v[:5])
        return x, y, float(np.sqrt(s * r)), float(np.sqrt(s / r)), c

    def _unfreeze(self) -> None:
        saved = self._saved
        if saved is None:
            return
        new_history = list(self.history_obs)
        self.x = saved["x"]  # type: ignore[assignment]
        self.P = saved["P"]  # type: ignore[assignment]
        self.last_measurement = saved["last_measurement"]  # type: ignore[assignment]
        self.history_obs = list(saved["history_obs"])[:-1]  # type: ignore[call-overload]
        self.observed = True
        self._saved = None
        idx = [i for i, d in enumerate(new_history) if d is not None]
        if len(idx) < 2:
            return
        i1, i2 = idx[-2], idx[-1]
        z1 = new_history[i1]
        if self.anchor == "last_measurement" and self.last_measurement is not None:
            z1 = self.last_measurement
        assert z1 is not None
        z2 = new_history[i2]
        assert z2 is not None
        x1, y1, w1, h1, c1 = self._box(z1)
        x2, y2, w2, h2, c2 = self._box(z2)
        gap = i2 - i1
        dx, dy, dw, dh, dc = ((x2 - x1) / gap, (y2 - y1) / gap, (w2 - w1) / gap,
                              (h2 - h1) / gap, (c2 - c1) / gap)  # fmt: skip
        for i in range(gap):
            x, y = x1 + (i + 1) * dx, y1 + (i + 1) * dy
            w, h = w1 + (i + 1) * dw, h1 + (i + 1) * dh
            s, r = w * h, w / float(h)
            if self.layout == "xysr":
                box = np.array([x, y, s, r]).reshape(4, 1)
            else:
                box = np.array([x, y, s, c1 + (i + 1) * dc, r]).reshape(5, 1)
            self.update(box)
            if i != gap - 1:
                self.predict()

    # ---------------------------------------------------------------- Deep OC-SORT CMC
    def apply_affine_correction(self, m: FloatArray, t: FloatArray) -> None:
        """Deep OC-SORT ``apply_affine_correction`` for the SORT (``new_kf_off``) state.

        Position and position-velocity are rotated/translated; scale and aspect are not, and only
        the diagonal 2×2 covariance blocks are transformed (as upstream).
        """
        t = t.reshape(2, 1)
        self.x[:2] = m @ self.x[:2] + t
        self.x[4:6] = m @ self.x[4:6]
        self.P[:2, :2] = m @ self.P[:2, :2] @ m.T
        self.P[4:6, 4:6] = m @ self.P[4:6, 4:6] @ m.T
        saved = self._saved
        if not self.observed and saved is not None:
            sx: FloatArray = saved["x"]  # type: ignore[assignment]
            sP: FloatArray = saved["P"]  # type: ignore[assignment]
            sx[:2] = m @ sx[:2] + t
            sx[4:6] = m @ sx[4:6]
            sP[:2, :2] = m @ sP[:2, :2] @ m.T
            sP[4:6, 4:6] = m @ sP[4:6, 4:6] @ m.T
            lm: FloatArray | None = saved["last_measurement"]  # type: ignore[assignment]
            if lm is not None:
                lm[:2] = m @ lm[:2] + t


def bbox_to_z(bbox: FloatArray) -> FloatArray:
    """``[x1, y1, x2, y2, ...]`` → ``[x, y, s, r]`` column (OC-SORT ``convert_bbox_to_z``)."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r], np.float64).reshape(4, 1)


def x_to_bbox(x: FloatArray) -> FloatArray:
    """``[x, y, s, r, ...]`` → ``(1, 4)`` ``[x1, y1, x2, y2]`` (OC-SORT ``convert_x_to_bbox``)."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array(
        [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0], np.float64
    ).reshape(1, 4)

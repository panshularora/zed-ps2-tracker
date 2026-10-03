"""Per-detection visibility estimation at inference time.

Phase 1 ships only :class:`ConstantVisibility` (every detection fully visible), which reproduces
the baselines exactly. The planned phase 2 estimator (``DepthOrderVisibility``, *not built yet*)
uses pairwise box overlap and depth order (the box whose bottom edge is lower in the image is
taken to be in front) to estimate the visible fraction of each box, and is scored against the
MOT ``gt.txt`` visibility column.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from tracker.types import FloatArray


class VisibilityEstimator(Protocol):
    def __call__(self, xyxy: FloatArray) -> FloatArray:
        """Return an ``(N,)`` visibility in ``[0, 1]`` for ``(N, 4)`` boxes of one frame."""
        ...


class ConstantVisibility:
    def __init__(self, value: float = 1.0) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("visibility must be in [0, 1]")
        self.value = value

    def __call__(self, xyxy: FloatArray) -> FloatArray:
        return np.full(np.asarray(xyxy).reshape(-1, 4).shape[0], self.value, dtype=np.float64)


def build_visibility(name: str) -> VisibilityEstimator:
    if name == "constant":
        return ConstantVisibility()
    if name == "depth_order":
        raise NotImplementedError("depth_order visibility is planned for phase 2")
    raise ValueError(f"unknown visibility estimator {name!r}")

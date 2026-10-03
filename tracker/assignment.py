"""Thresholded linear assignment (Hungarian / Jonker–Volgenant).

``linear_assignment(cost, thresh)`` returns the minimum-cost matching that only uses pairs with
cost < ``thresh``. It is solved exactly as ``lap.lapjv(cost, extend_cost=True,
cost_limit=thresh)`` (the call ByteTrack makes): the matrix is padded so that leaving a row and
a column unmatched costs ``thresh`` in total, which makes "unmatched" a priced option inside a
single global optimisation instead of a post-hoc filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from tracker.types import FloatArray, IntArray

Backend = Literal["scipy", "lap"]

#: stands in for +inf inside the solver (both solvers need finite entries to be robust)
_BIG = 1e6


@dataclass(frozen=True)
class Matching:
    matches: npt.NDArray[np.int64]
    """``(K, 2)`` array of ``(row, col)`` pairs."""
    unmatched_rows: IntArray
    unmatched_cols: IntArray

    @staticmethod
    def none(n_rows: int, n_cols: int) -> Matching:
        return Matching(
            np.zeros((0, 2), dtype=np.int64),
            np.arange(n_rows, dtype=np.int64),
            np.arange(n_cols, dtype=np.int64),
        )


class Assigner(Protocol):
    """Interface for an assignment step (phase 2's delayed assigner implements this too)."""

    def __call__(self, cost: FloatArray, thresh: float) -> Matching: ...


def _extended(cost: FloatArray, thresh: float) -> FloatArray:
    n, m = cost.shape
    c = np.where(np.isfinite(cost), cost, _BIG)
    ext = np.full((n + m, n + m), thresh / 2.0)
    ext[n:, m:] = 0.0
    ext[:n, :m] = c
    return ext


def linear_assignment(cost: FloatArray, thresh: float, backend: Backend = "scipy") -> Matching:
    """Solve the thresholded assignment problem.

    Args:
        cost: ``(N, M)`` cost matrix; ``inf`` / ``nan`` entries are forbidden pairs.
        thresh: only pairs with ``cost < thresh`` may be matched.
        backend: ``"scipy"`` (``linear_sum_assignment`` on the padded matrix) or ``"lap"``
            (``lap.lapjv``, needs the optional ``lap`` package).
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2-D, got shape {cost.shape}")
    n, m = cost.shape
    if n == 0 or m == 0:
        return Matching.none(n, m)
    cost = np.where(np.isnan(cost), np.inf, cost)
    if not np.any(cost < thresh):
        return Matching.none(n, m)

    if backend == "lap":
        import lap

        c = np.where(np.isfinite(cost), cost, _BIG)
        _, x, _ = lap.lapjv(c, extend_cost=True, cost_limit=thresh)
        rows = np.flatnonzero(np.asarray(x) >= 0)
        cols = np.asarray(x)[rows]
    elif backend == "scipy":
        r, c_idx = linear_sum_assignment(_extended(cost, thresh))
        keep = (r < n) & (c_idx < m)
        rows, cols = r[keep], c_idx[keep]
    else:
        raise ValueError(f"unknown backend {backend!r}")

    # Belt and braces: a forbidden or over-threshold pair is never returned.
    ok = cost[rows, cols] < thresh
    rows, cols = rows[ok].astype(np.int64), cols[ok].astype(np.int64)
    matches = np.stack([rows, cols], axis=1) if rows.size else np.zeros((0, 2), np.int64)
    return Matching(
        matches=matches,
        unmatched_rows=np.setdiff1d(np.arange(n, dtype=np.int64), rows),
        unmatched_cols=np.setdiff1d(np.arange(m, dtype=np.int64), cols),
    )


class HungarianAssigner:
    """Immediate (commit-this-frame) assignment. The phase 1 default."""

    def __init__(self, backend: Backend = "scipy") -> None:
        self.backend: Backend = backend

    def __call__(self, cost: FloatArray, thresh: float) -> Matching:
        return linear_assignment(cost, thresh, self.backend)

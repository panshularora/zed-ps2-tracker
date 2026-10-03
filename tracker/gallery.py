"""Appearance memory of a track.

Phase 1 ships the BoT-SORT / DeepSORT exponential moving average. The update takes the
estimated visibility of the detection so that phase 2 can freeze the update when a person is
mostly hidden (the classic cause of EMA pollution and ID swaps) without changing callers.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

Embedding = npt.NDArray[np.float32]


class AppearanceMemory(Protocol):
    @property
    def embedding(self) -> Embedding: ...

    def update(self, embedding: Embedding, visibility: float = 1.0) -> bool:
        """Fold in a new observation. Returns False if the update was gated out."""
        ...


def _normalise(x: npt.NDArray[np.floating]) -> Embedding:
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class EmaAppearance:
    """``e ← normalise(α e + (1 − α) x)`` with optional visibility gate.

    Args:
        alpha: EMA momentum (BoT-SORT uses 0.9).
        min_visibility: updates with ``visibility < min_visibility`` are skipped. ``0.0``
            (phase 1 default) means "always update", i.e. plain BoT-SORT behaviour.
    """

    def __init__(self, embedding: Embedding, alpha: float = 0.9, min_visibility: float = 0.0):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        self._e = _normalise(embedding)
        self.alpha = alpha
        self.min_visibility = min_visibility
        self.n_updates = 0
        self.n_gated = 0

    @property
    def embedding(self) -> Embedding:
        return self._e

    def update(self, embedding: Embedding, visibility: float = 1.0) -> bool:
        if visibility < self.min_visibility:
            self.n_gated += 1
            return False
        self._e = _normalise(self.alpha * self._e + (1.0 - self.alpha) * _normalise(embedding))
        self.n_updates += 1
        return True

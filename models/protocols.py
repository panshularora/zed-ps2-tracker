"""Interfaces Stage A is written against, so the orchestration is testable without torch/GPU."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np
import numpy.typing as npt

BGRImage = npt.NDArray[np.uint8]


class Detector(Protocol):
    name: str

    def info(self) -> dict[str, object]:
        """Provenance stored in the cache meta (weights, sha256, test size, floors...)."""
        ...

    def __call__(self, images: list[BGRImage]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return per image ``(tlwh float32 (N, 4), score float32 (N,))`` in pixel coords."""
        ...


class Embedder(Protocol):
    name: str
    dim: int

    def info(self) -> dict[str, object]: ...

    def __call__(self, image: BGRImage, tlwh: np.ndarray) -> npt.NDArray[np.float16]:
        """Return ``(N, dim)`` L2-normalised float16 embeddings for the boxes of one image."""
        ...


class FrameSource(Protocol):
    name: str
    frame_rate: float
    seq_length: int
    width: int
    height: int

    def __iter__(self) -> Iterator[tuple[int, BGRImage]]:
        """Yield ``(frame_id, image)`` with 1-based, consecutive frame ids."""
        ...

"""Core value types shared by every tracker component.

Boxes are always ``float64`` in image pixels. Two layouts are used and named explicitly:

* ``tlwh``: top-left x, top-left y, width, height (the MOTChallenge file layout)
* ``xyxy``: x1, y1, x2, y2 (what IoU is computed on)
* ``xywh``: centre x, centre y, width, height (the Kalman measurement space)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def tlwh_to_xyxy(tlwh: FloatArray) -> FloatArray:
    out = np.asarray(tlwh, dtype=np.float64).reshape(-1, 4).copy()
    out[:, 2:] += out[:, :2]
    return out


def tlwh_to_xywh(tlwh: FloatArray) -> FloatArray:
    out = np.asarray(tlwh, dtype=np.float64).reshape(-1, 4).copy()
    out[:, :2] += out[:, 2:] / 2.0
    return out


def xywh_to_tlwh(xywh: FloatArray) -> FloatArray:
    out = np.asarray(xywh, dtype=np.float64).reshape(-1, 4).copy()
    out[:, :2] -= out[:, 2:] / 2.0
    return out


def xywh_to_xyxy(xywh: FloatArray) -> FloatArray:
    return tlwh_to_xyxy(xywh_to_tlwh(xywh))


@dataclass(frozen=True)
class Detections:
    """All detections of one frame.

    Attributes:
        tlwh: ``(N, 4)`` boxes.
        scores: ``(N,)`` detector confidences in ``[0, 1]``.
        embeddings: optional ``(N, D)`` L2-normalised re-ID embeddings (``float32``).
    """

    tlwh: FloatArray
    scores: FloatArray
    embeddings: npt.NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        tlwh = np.asarray(self.tlwh, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        if tlwh.shape[0] != scores.shape[0]:
            raise ValueError(f"{tlwh.shape[0]} boxes but {scores.shape[0]} scores")
        object.__setattr__(self, "tlwh", tlwh)
        object.__setattr__(self, "scores", scores)
        if self.embeddings is not None:
            emb = np.asarray(self.embeddings, dtype=np.float32)
            if emb.ndim != 2 or emb.shape[0] != tlwh.shape[0]:
                raise ValueError(
                    f"embeddings shape {emb.shape} does not match {tlwh.shape[0]} boxes"
                )
            object.__setattr__(self, "embeddings", emb)

    def __len__(self) -> int:
        return int(self.tlwh.shape[0])

    @property
    def xyxy(self) -> FloatArray:
        return tlwh_to_xyxy(self.tlwh)

    @property
    def xywh(self) -> FloatArray:
        return tlwh_to_xywh(self.tlwh)

    def subset(self, idx: IntArray | npt.NDArray[np.bool_]) -> Detections:
        emb = None if self.embeddings is None else self.embeddings[idx]
        return Detections(self.tlwh[idx], self.scores[idx], emb)

    @staticmethod
    def empty(embedding_dim: int | None = None) -> Detections:
        emb = None if embedding_dim is None else np.zeros((0, embedding_dim), np.float32)
        return Detections(np.zeros((0, 4)), np.zeros(0), emb)


@dataclass(frozen=True)
class TrackOutput:
    """One reported track box for one frame (what gets written to the MOT result file)."""

    track_id: int
    tlwh: tuple[float, float, float, float]
    score: float
    state: str
    """Lifecycle state name at output time (``confirmed`` for reported tracks)."""
    visibility: float = 1.0
    """Estimated visibility used by the tracker this frame (1.0 until phase 2)."""


@dataclass(frozen=True)
class PendingDecision:
    """An ambiguous association that :class:`tracker.delayed.DelayedTracker` held open.

    It is attached to the committed result of ``opened_frame``. The decision was taken with
    look-ahead up to ``deadline_frame`` (= opened + K), so the viewer can draw the candidates as
    dashed boxes with a countdown.
    """

    track_id: int
    candidate_boxes_tlwh: tuple[tuple[float, float, float, float], ...]
    candidate_costs: tuple[float, ...]
    opened_frame: int
    deadline_frame: int
    chosen_index: int = -1
    """Index into the candidates that was committed; -1 = left unmatched in this association."""
    competing_track_ids: tuple[int, ...] = ()
    changed: bool = False
    """True if the look-ahead overrode the immediate solver's choice for this cluster."""


@dataclass(frozen=True)
class FrameResult:
    """Everything the tracker reports for one frame."""

    frame_id: int
    tracks: tuple[TrackOutput, ...]
    pending: tuple[PendingDecision, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

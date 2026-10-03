"""Track lifecycle: an explicit state machine plus the per-track record.

States and the only legal transitions::

    TENTATIVE ──(min_hits matched frames)──► CONFIRMED ◄──(re-associated)── LOST
        │                                       │                            │
        └─(missed > tentative_max_misses)─┐     └──(missed this frame)──────►┘
                                          ▼                                  │
                                       REMOVED ◄──(lost > max_time_lost)─────┘

``REMOVED`` is terminal. Any other transition raises :class:`IllegalTransition`, so a logic bug in
an association stage fails loudly in tests instead of silently corrupting IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from tracker.gallery import AppearanceMemory
from tracker.kalman import KalmanFilterXYWH
from tracker.types import FloatArray, tlwh_to_xywh, xywh_to_tlwh, xywh_to_xyxy


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


ALLOWED_TRANSITIONS: dict[TrackState, frozenset[TrackState]] = {
    TrackState.TENTATIVE: frozenset({TrackState.CONFIRMED, TrackState.REMOVED}),
    TrackState.CONFIRMED: frozenset({TrackState.LOST, TrackState.REMOVED}),
    TrackState.LOST: frozenset({TrackState.CONFIRMED, TrackState.REMOVED}),
    TrackState.REMOVED: frozenset(),
}


class IllegalTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class LifecycleConfig:
    min_hits: int = 2
    """Matched frames (including the birth frame) needed to confirm a tentative track."""
    tentative_max_misses: int = 0
    """A tentative track is removed after more than this many consecutive misses."""
    max_time_lost: int = 30
    """Frames a lost track is kept for re-association before removal."""
    confirm_on_first_frame: bool = True
    """ByteTrack convention: tracks born on the sequence's first frame start confirmed."""


@dataclass
class Track:
    """Mutable per-track record. Owned and mutated only by the tracker."""

    track_id: int
    mean: FloatArray
    covariance: FloatArray
    state: TrackState
    start_frame: int
    last_frame: int
    """Last frame with an associated detection."""
    score: float
    hits: int = 1
    misses: int = 0
    """Consecutive frames without an associated detection."""
    appearance: AppearanceMemory | None = None
    visibility: float = 1.0
    history: list[TrackState] = field(default_factory=list)
    removal_flagged: bool = False
    """Only with ``lifecycle.removal_lag``: upstream already put this id in ``removed_stracks``."""

    # ------------------------------------------------------------------ geometry
    @property
    def xywh(self) -> FloatArray:
        return np.asarray(self.mean[:4], dtype=np.float64).copy()

    @property
    def tlwh(self) -> FloatArray:
        return np.asarray(xywh_to_tlwh(self.xywh)[0], dtype=np.float64)

    @property
    def xyxy(self) -> FloatArray:
        return np.asarray(xywh_to_xyxy(self.xywh)[0], dtype=np.float64)

    @property
    def is_active(self) -> bool:
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)

    # ------------------------------------------------------------------ transitions
    def transition(self, new: TrackState) -> None:
        if new not in ALLOWED_TRANSITIONS[self.state]:
            raise IllegalTransition(
                f"track {self.track_id}: {self.state.value} -> {new.value} is not allowed"
            )
        self.history.append(self.state)
        self.state = new

    def mark_matched(
        self,
        kf: KalmanFilterXYWH,
        tlwh: FloatArray,
        score: float,
        frame_id: int,
        cfg: LifecycleConfig,
    ) -> None:
        """Kalman update with an associated detection, then advance the lifecycle."""
        self.mean, self.covariance = kf.update(self.mean, self.covariance, tlwh_to_xywh(tlwh)[0])
        self.score = float(score)
        self.last_frame = frame_id
        self.hits += 1
        self.misses = 0
        confirm_tentative = self.state is TrackState.TENTATIVE and self.hits >= cfg.min_hits
        if confirm_tentative or self.state is TrackState.LOST:  # LOST → re-associated
            self.transition(TrackState.CONFIRMED)

    def mark_missed(self, cfg: LifecycleConfig) -> None:
        """No detection this frame. Tentative → removed (after grace); confirmed → lost."""
        self.misses += 1
        if self.state is TrackState.TENTATIVE:
            if self.misses > cfg.tentative_max_misses:
                self.transition(TrackState.REMOVED)
        elif self.state is TrackState.CONFIRMED:
            self.transition(TrackState.LOST)
        # LOST stays LOST; removal happens in expire_if_stale() once the window has passed.

    def expire_if_stale(self, frame_id: int, cfg: LifecycleConfig) -> bool:
        """Remove a lost track whose re-association window has passed. Returns True if removed."""
        if self.state is TrackState.LOST and frame_id - self.last_frame > cfg.max_time_lost:
            self.transition(TrackState.REMOVED)
            return True
        return False

    def remove(self) -> None:
        self.transition(TrackState.REMOVED)

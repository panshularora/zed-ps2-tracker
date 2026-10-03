import numpy as np
import pytest

from tracker.kalman import KalmanFilterXYWH
from tracker.lifecycle import (
    ALLOWED_TRANSITIONS,
    IllegalTransition,
    LifecycleConfig,
    Track,
    TrackState,
)

KF = KalmanFilterXYWH()
BOX = np.array([10.0, 10.0, 20.0, 40.0])


def make(state: TrackState = TrackState.TENTATIVE, frame: int = 1) -> Track:
    mean, cov = KF.initiate(np.array([20.0, 30.0, 20.0, 40.0]))
    return Track(1, mean, cov, state, start_frame=frame, last_frame=frame, score=0.9)


def test_tentative_confirms_after_min_hits() -> None:
    cfg = LifecycleConfig(min_hits=3)
    t = make()
    t.mark_matched(KF, BOX, 0.9, 2, cfg)
    assert t.state is TrackState.TENTATIVE
    t.mark_matched(KF, BOX, 0.9, 3, cfg)
    assert t.state is TrackState.CONFIRMED
    assert t.history == [TrackState.TENTATIVE]


def test_tentative_removed_on_miss() -> None:
    t = make()
    t.mark_missed(LifecycleConfig(tentative_max_misses=0))
    assert t.state is TrackState.REMOVED


def test_tentative_grace_period() -> None:
    cfg = LifecycleConfig(tentative_max_misses=1)
    t = make()
    t.mark_missed(cfg)
    assert t.state is TrackState.TENTATIVE
    t.mark_missed(cfg)
    assert t.state is TrackState.REMOVED


def test_confirmed_to_lost_to_confirmed() -> None:
    cfg = LifecycleConfig()
    t = make(TrackState.CONFIRMED)
    t.mark_missed(cfg)
    assert t.state is TrackState.LOST
    t.mark_missed(cfg)  # stays lost
    assert t.state is TrackState.LOST
    t.mark_matched(KF, BOX, 0.8, 4, cfg)
    assert t.state is TrackState.CONFIRMED
    assert t.misses == 0
    assert t.last_frame == 4


def test_lost_expires_after_buffer() -> None:
    cfg = LifecycleConfig(max_time_lost=5)
    t = make(TrackState.CONFIRMED, frame=10)
    t.mark_missed(cfg)
    assert not t.expire_if_stale(15, cfg)
    assert t.state is TrackState.LOST
    assert t.expire_if_stale(16, cfg)
    assert t.state is TrackState.REMOVED


def test_confirmed_never_expires_directly() -> None:
    t = make(TrackState.CONFIRMED)
    assert not t.expire_if_stale(10_000, LifecycleConfig(max_time_lost=1))


def test_removed_is_terminal() -> None:
    t = make()
    t.remove()
    for s in TrackState:
        with pytest.raises(IllegalTransition):
            t.transition(s)


@pytest.mark.parametrize("src", list(TrackState))
@pytest.mark.parametrize("dst", list(TrackState))
def test_transition_table_enforced(src: TrackState, dst: TrackState) -> None:
    t = make(src)
    if dst in ALLOWED_TRANSITIONS[src]:
        t.transition(dst)
        assert t.state is dst
    else:
        with pytest.raises(IllegalTransition):
            t.transition(dst)


def test_geometry_roundtrip() -> None:
    t = make()
    np.testing.assert_allclose(t.tlwh, [10, 10, 20, 40])
    np.testing.assert_allclose(t.xyxy, [10, 10, 30, 50])

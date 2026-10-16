"""Keep-ID: the same body keeps the same number through a dropped-box gap."""

import numpy as np

from tracker.keep_id import KeepIdTracker
from tracker.types import Detections


def _box(x: float, y: float, s: float = 24.0) -> np.ndarray:
    return np.array([[x - s / 2, y - s / 2, s, s]], np.float64)


def test_same_id_after_40_frame_gap():
    k = KeepIdTracker(fps=30, buffer=90, max_dist=64)
    y = 80.0
    first = None
    for f in range(1, 21):
        y += 2.0
        r = k.update(Detections(_box(200, y), np.array([0.9])))
        assert len(r.tracks) == 1
        first = r.tracks[0].track_id
    for _ in range(40):  # detector drop: person still walking
        y += 2.0
        r = k.update(Detections(np.zeros((0, 4)), np.zeros(0)))
        assert r.tracks[0].track_id == first
        assert r.tracks[0].state == "lost"
    y += 2.0
    r = k.update(Detections(_box(200, y), np.array([0.85])))
    assert r.tracks[0].track_id == first
    assert r.tracks[0].state == "confirmed"


def test_new_id_only_when_far_from_every_lost_track():
    k = KeepIdTracker(fps=30, buffer=90, max_dist=64)
    k.update(Detections(_box(100, 100), np.array([0.9])))
    k.update(Detections(np.zeros((0, 4)), np.zeros(0)))
    r = k.update(Detections(_box(500, 300), np.array([0.9])))
    ids = sorted(t.track_id for t in r.tracks)
    assert ids == [1, 2]

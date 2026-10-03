"""Behavioural tests for ByteTrackLike on synthetic scenes."""

from collections import defaultdict

import numpy as np
import pytest

from tracker.bytetrack import ByteTrackLike
from tracker.config import TrackerConfig, config_from_dict, load_config
from tracker.types import Detections


def frame(boxes: list[tuple[float, float, float, float]], scores: list[float],
          emb: np.ndarray | None = None) -> Detections:  # fmt: skip
    return Detections(np.array(boxes, float).reshape(-1, 4), np.array(scores, float), emb)


def run(tracker: ByteTrackLike, frames: list[Detections]) -> list[dict[int, tuple[float, ...]]]:
    out = []
    for d in frames:
        res = tracker.update(d)
        out.append({t.track_id: t.tlwh for t in res.tracks})
    return out


def crossing_scene(n: int = 80, speed: float = 6.0) -> tuple[list[Detections], np.ndarray]:
    """Two people walk towards each other, fully overlap mid-way and separate.

    Returns frames and, per frame, the true x of target A (to identify who is who).
    """
    frames, xa = [], []
    for t in range(n):
        a = (100 + speed * t, 200.0, 40.0, 100.0)
        b = (100 + speed * (n - 1) - speed * t, 205.0, 40.0, 100.0)
        frames.append(frame([a, b], [0.9, 0.9]))
        xa.append(a[0])
    return frames, np.array(xa)


def ids_following(out: list[dict[int, tuple[float, ...]]], xs: np.ndarray) -> list[int]:
    """Track id closest to the given x in each frame where it is reported."""
    ids = []
    for f, tracks in enumerate(out):
        if tracks:
            best = min(tracks, key=lambda k: abs(tracks[k][0] - xs[f]))
            if abs(tracks[best][0] - xs[f]) < 10:
                ids.append(best)
    return ids


def test_empty_frames() -> None:
    trk = ByteTrackLike()
    for _ in range(3):
        assert trk.update(Detections.empty()).tracks == ()


def test_first_frame_confirmed_then_steady_id() -> None:
    trk = ByteTrackLike()
    out = run(trk, [frame([(10 + 2 * t, 10, 30, 80)], [0.9]) for t in range(10)])
    assert all(list(o) == [1] for o in out)


def test_later_birth_is_tentative_for_one_frame() -> None:
    trk = ByteTrackLike()
    trk.update(Detections.empty())
    r1 = trk.update(frame([(10, 10, 30, 80)], [0.9]))
    assert r1.tracks == ()  # tentative, not reported
    r2 = trk.update(frame([(11, 10, 30, 80)], [0.9]))
    assert [t.track_id for t in r2.tracks] == [1]


def test_low_score_never_starts_track_but_continues_one() -> None:
    trk = ByteTrackLike()
    assert run(trk, [frame([(10, 10, 30, 80)], [0.4])] * 3) == [{}, {}, {}]
    trk = ByteTrackLike()
    seq = [frame([(10 + t, 10, 30, 80)], [0.9]) for t in range(5)]
    seq += [frame([(15 + t, 10, 30, 80)], [0.3]) for t in range(5)]  # occluded: low score
    out = run(trk, seq)
    assert all(list(o) == [1] for o in out)


def test_below_low_thresh_is_ignored() -> None:
    trk = ByteTrackLike()
    out = run(trk, [frame([(10, 10, 30, 80)], [0.9]), frame([(10, 10, 30, 80)], [0.05])])
    assert out[1] == {}


@pytest.mark.parametrize("speed", [3.0, 6.0, 10.0])
def test_crossing_targets_keep_ids(speed: float) -> None:
    frames, xa = crossing_scene(speed=speed)
    out = run(ByteTrackLike(), frames)
    xb = 100 + speed * (len(frames) - 1) - speed * np.arange(len(frames))
    ids_a, ids_b = ids_following(out, xa), ids_following(out, xb)
    assert len(ids_a) >= len(frames) - 2
    assert len(ids_b) >= len(frames) - 2
    assert set(ids_a) == {ids_a[0]}, f"A switched: {sorted(set(ids_a))}"
    assert set(ids_b) == {ids_b[0]}, f"B switched: {sorted(set(ids_b))}"
    assert ids_a[0] != ids_b[0]


def test_reassociation_after_short_full_occlusion() -> None:
    trk = ByteTrackLike()
    seq = [frame([(100 + 3 * t, 100, 40, 100)], [0.9]) for t in range(20)]
    seq += [Detections.empty()] * 10  # fully hidden for 10 frames
    seq += [frame([(100 + 3 * t, 100, 40, 100)], [0.9]) for t in range(30, 45)]
    out = run(trk, seq)
    seen = {k for o in out for k in o}
    assert seen == {1}
    assert out[25] == {}  # lost tracks are not reported
    assert list(out[30]) == [1]  # re-associated immediately on reappearance


def test_lost_track_removed_after_buffer() -> None:
    cfg = load_config(overrides=["lifecycle.track_buffer=5"])
    trk = ByteTrackLike(cfg)
    run(trk, [frame([(100, 100, 40, 100)], [0.9])] * 3 + [Detections.empty()] * 7)
    assert trk.tracks == []
    out = run(trk, [frame([(100, 100, 40, 100)], [0.9])] * 2)
    assert list(out[1]) == [2]


def test_frame_rate_buffer_scaling_is_opt_in() -> None:
    assert ByteTrackLike(frame_rate=15).lifecycle.max_time_lost == 30  # ByteTrack behaviour
    cfg = config_from_dict({"lifecycle": {"scale_buffer_by_fps": True}})
    assert ByteTrackLike(cfg, frame_rate=15).lifecycle.max_time_lost == 15
    assert ByteTrackLike(cfg, frame_rate=60).lifecycle.max_time_lost == 60


def test_score_equal_to_high_thresh_is_in_neither_set() -> None:
    trk = ByteTrackLike()
    trk.update(frame([(10, 10, 40, 100)], [0.9]))
    trk.update(frame([(10, 10, 40, 100)], [0.9]))
    # exactly 0.6: neither high nor low, so the track is not updated this frame
    res = trk.update(frame([(12, 10, 40, 100)], [0.6]))
    assert len(res.tracks) == 0


def test_output_filters() -> None:
    trk = ByteTrackLike()
    res = trk.update(frame([(0, 0, 2, 2), (0, 100, 100, 20), (200, 200, 30, 80)], [0.9] * 3))
    assert len(res.tracks) == 1  # tiny + wide boxes are tracked but not reported
    assert len(trk.tracks) == 3


def test_appearance_resolves_ambiguous_swap() -> None:
    """Two people swap sides while hidden; IoU alone is uninformative, appearance decides."""
    ea, eb = np.eye(2, dtype=np.float32)
    start = frame([(100, 100, 40, 100), (300, 100, 40, 100)], [0.9, 0.9], np.stack([ea, eb]))
    # both reappear far from either prediction; b's embedding is now on the left
    boxes = [(190, 100, 40, 100), (210, 100, 40, 100)]

    def x_after(
        cfg: TrackerConfig, left_emb: np.ndarray, right_emb: np.ndarray
    ) -> dict[int, float]:
        trk = ByteTrackLike(cfg)
        run(trk, [start] * 5)
        res = trk.update(frame(boxes, [0.9, 0.9], np.stack([left_emb, right_emb])))
        return {t.track_id: t.tlwh[0] for t in res.tracks}

    # IoU-only: no overlap, so neither track is continued (both are re-born later)
    assert x_after(TrackerConfig(), eb, ea) == {}
    # appearance (with the proximity gate opened) decides who went where
    app = load_config(overrides=["appearance.enabled=true", "appearance.proximity_thresh=1.0"])
    swapped = x_after(app, eb, ea)  # a (id 1) is now on the right, b (id 2) on the left
    same = x_after(app, ea, eb)
    assert set(swapped) == set(same) == {1, 2}
    assert swapped[1] > same[1]  # id 1 followed the right-hand box
    assert swapped[2] < same[2]  # id 2 followed the left-hand box


def test_reports_are_causal_and_deterministic() -> None:
    frames, _ = crossing_scene()
    a = run(ByteTrackLike(), frames)
    b = run(ByteTrackLike(), frames)
    assert a == b
    # prefix property: output for frame t does not depend on frames after t
    c = run(ByteTrackLike(), frames[:40])
    assert a[:40] == c


def test_delayed_assignment_not_built() -> None:
    with pytest.raises(NotImplementedError):
        ByteTrackLike(load_config(overrides=["delayed.enabled=true"]))


def test_dense_random_scene_ids_unique_per_frame() -> None:
    rng = np.random.default_rng(3)
    trk = ByteTrackLike()
    pos = rng.uniform(0, 1000, (40, 2))
    vel = rng.normal(0, 3, (40, 2))
    counts: dict[int, int] = defaultdict(int)
    for _ in range(100):
        pos += vel
        keep = rng.uniform(size=40) > 0.1
        boxes = [(x, y, 30.0, 80.0) for (x, y) in pos[keep]]
        res = trk.update(frame(boxes, rng.uniform(0.2, 1.0, len(boxes)).tolist()))
        ids = [t.track_id for t in res.tracks]
        assert len(ids) == len(set(ids))
        for i in ids:
            counts[i] += 1
    assert len(counts) > 0

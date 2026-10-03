"""Delayed-decision wrapper: exact K=0 equivalence, lagged emission, cluster logic."""

import numpy as np
import pytest

from tracker.bytetrack import ByteTrackLike, Stage1Record
from tracker.config import load_config
from tracker.delayed import Cluster, DelayedTracker, enumerate_hypotheses, find_clusters, force_hook
from tracker.lifecycle import Track, TrackState
from tracker.runner import build_tracker
from tracker.types import Detections


def dense_scene(n_frames: int = 60, n: int = 25, seed: int = 3) -> list[Detections]:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 600, (n, 2))
    vel = rng.normal(0, 4, (n, 2))
    frames = []
    for _ in range(n_frames):
        pos += vel
        keep = rng.uniform(size=n) > 0.15
        boxes = np.c_[pos[keep], np.full((keep.sum(), 2), [40.0, 100.0])]
        boxes += rng.normal(0, 2, boxes.shape)
        frames.append(Detections(boxes, rng.uniform(0.3, 1.0, keep.sum())))
    return frames


def run_delayed(k: int, frames: list[Detections], **kw: object) -> tuple[list, DelayedTracker]:
    cfg = load_config(overrides=["delayed.enabled=true"])
    d = DelayedTracker(ByteTrackLike(cfg, allow_delayed=True), max_delay_frames=k, margin=0.05,
                       **kw)  # type: ignore[arg-type]  # fmt: skip
    out = [r for r in (d.update(f) for f in frames) if r is not None]
    return out + d.flush(), d


def as_tuples(results: list) -> list:
    return [(r.frame_id, tuple((t.track_id, t.tlwh) for t in r.tracks)) for r in results]


def test_k0_identical_to_baseline() -> None:
    frames = dense_scene()
    base = ByteTrackLike()
    ref = [base.update(f) for f in frames]
    got, d = run_delayed(0, frames)
    assert as_tuples(got) == as_tuples(ref)
    assert d.stats.rollouts == 0


@pytest.mark.parametrize("k", [1, 3, 5])
def test_lagged_emission_once_per_frame(k: int) -> None:
    frames = dense_scene()
    cfg = load_config(overrides=["delayed.enabled=true"])
    d = DelayedTracker(ByteTrackLike(cfg, allow_delayed=True), max_delay_frames=k, margin=0.05)
    emitted = []
    for i, f in enumerate(frames, start=1):
        r = d.update(f)
        if i <= k:
            assert r is None
        else:
            assert r is not None
            assert r.frame_id == i - k
            emitted.append(r)
    emitted += d.flush()
    assert [r.frame_id for r in emitted] == list(range(1, len(frames) + 1))
    assert d.stats.clusters > 0  # the dense scene does produce ambiguity
    for r in emitted:
        ids = [t.track_id for t in r.tracks]
        assert len(ids) == len(set(ids))
        for p in r.pending:
            assert p.deadline_frame == p.opened_frame + k == r.frame_id + k


def test_huge_min_gain_reproduces_baseline() -> None:
    frames = dense_scene()
    base = ByteTrackLike()
    ref = [base.update(f) for f in frames]
    got, d = run_delayed(3, frames, min_gain=1e9)
    assert d.stats.clusters > 0
    assert d.stats.changed == 0
    assert as_tuples(got) == as_tuples(ref)


def _rec(cost: list[list[float]], matches: list[tuple[int, int]]) -> Stage1Record:
    c = np.array(cost, float)
    return Stage1Record(tuple(range(10, 10 + c.shape[0])), (False,) * c.shape[0], c,
                        np.zeros((c.shape[1], 4)), np.array(matches, np.int64).reshape(-1, 2),
                        0.9)  # fmt: skip


def test_find_clusters_and_hypotheses() -> None:
    # tracks 10, 11 compete for det 0 (0.30 vs 0.32); track 12 is alone on det 2.
    rec = _rec([[0.30, 0.95, 1.0], [0.32, 0.50, 1.0], [1.0, 1.0, 0.2]], [(0, 0), (1, 1), (2, 2)])
    clusters, skipped = find_clusters(rec, margin=0.05, max_cluster=4)
    assert skipped == 0
    assert len(clusters) == 1
    cl = clusters[0]
    assert set(cl.track_ids) == {10, 11}
    assert set(cl.cols) == {0, 1}
    assert cl.greedy == ((10, 0), (11, 1))
    hyps = enumerate_hypotheses(cl, 0.9, 6)
    assert hyps[0] == cl.greedy
    assert ((10, 0),) in hyps  # track 11 unmatched
    assert len(set(hyps)) == len(hyps)
    swaps = enumerate_hypotheses(cl, 0.9, 6, same_tracks=True)
    assert all({t for t, _ in h} == {10, 11} for h in swaps)
    assert find_clusters(rec, margin=0.05, max_cluster=1)[1] == 1
    lost = Stage1Record(rec.track_ids, (False, True, False), rec.cost, rec.dets_tlwh, rec.matches,
                        0.9)  # fmt: skip
    assert find_clusters(lost, 0.05, 4, confirmed_only=True)[0] == []


def test_force_hook() -> None:
    cl = Cluster((10, 11), (0, 1), {(10, 0): 0.3, (11, 0): 0.32, (11, 1): 0.5}, ((10, 0), (11, 1)))
    tracks = [
        Track(t, np.zeros(8), np.eye(8), TrackState.CONFIRMED, 1, 1, 0.9) for t in (10, 11, 12)
    ]
    cost = np.array([[0.30, 0.95, 1.0], [0.32, 0.50, 1.0], [1.0, 1.0, 0.2]])
    out = force_hook([(cl, ((11, 0),))])(tracks, cost.copy())
    assert out[1, 0] == pytest.approx(0.32)
    assert np.isinf(out[0]).all()  # track 10 forced unmatched
    assert np.isinf(out[1, 1:]).all()
    assert out[2, 2] == pytest.approx(0.2)  # outside the cluster: untouched


def test_runner_wraps_and_rejects_other_trackers() -> None:
    trk, info = build_tracker(load_config(overrides=["delayed.enabled=true"]), 25.0)
    assert isinstance(trk, DelayedTracker)
    assert info["delayed"]["K_ms"] == pytest.approx(1000 * 3 / 25, abs=0.05)
    cfg = load_config(overrides=["delayed.enabled=true", "name=ocsort"])
    with pytest.raises(ValueError, match="ByteTrackLike only"):
        build_tracker(cfg, 30.0)

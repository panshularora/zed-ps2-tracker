"""Rules of the ported baselines (OC-SORT, Deep OC-SORT, Hybrid-SORT, BoT-SORT) and CMC.

Frame-by-frame parity against the upstream code is checked by
``scripts/research/parity_upstream.py`` (needs the upstream clones); its recorded reports in
``docs/results/parity/`` are asserted here so a regression can't silently ship.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tracker.baselines.common import angle_cost, iou_batch, k_previous_obs, speed_direction
from tracker.baselines.hybrid_sort import HybridSortTracker, hmiou
from tracker.baselines.ocsort import OCSortTracker, aw_weights
from tracker.baselines.sort_kf import OruKalmanFilter, bbox_to_z
from tracker.bytetrack import ByteTrackLike, apply_gmc
from tracker.cmc import FileCMC, build_cmc, find_cmc_file, parse_cmc_file
from tracker.config import load_config
from tracker.cost import iou_matrix
from tracker.io import CacheMeta
from tracker.runner import build_tracker
from tracker.types import Detections

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "baselines"


def dets(boxes, scores):
    return Detections(np.array(boxes, float).reshape(-1, 4), np.array(scores, float))


def sort_kf() -> OruKalmanFilter:
    kf = OruKalmanFilter(7, 4)
    kf.F = np.eye(7)
    for i in range(3):
        kf.F[i, 4 + i] = 1
    kf.H = np.eye(4, 7)
    kf.R[2:, 2:] *= 10.0
    kf.P[4:, 4:] *= 1000.0
    kf.P *= 10.0
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01
    return kf


def test_kf_update_matches_textbook_equations():
    kf = sort_kf()
    kf.x[:4] = bbox_to_z(np.array([10, 20, 50, 120.0]))
    kf.predict()
    P0, x0 = kf.P.copy(), kf.x.copy()
    z = bbox_to_z(np.array([12, 21, 52, 121.0]))
    kf.update(z)
    S = kf.H @ P0 @ kf.H.T + kf.R
    K = P0 @ kf.H.T @ np.linalg.inv(S)
    np.testing.assert_allclose(kf.x, x0 + K @ (z - kf.H @ x0))
    np.testing.assert_allclose(kf.P, (np.eye(7) - K @ kf.H) @ P0, atol=1e-8)


def run_gap(oru: bool) -> np.ndarray:
    """5 px/frame for 10 frames, then the person stops while unobserved (frames 10..19)."""
    kf = sort_kf()
    kf.oru = oru
    x = lambda t: 100 + 5 * min(t, 9)  # noqa: E731
    box = lambda t: np.array([x(t), 50, x(t) + 40, 150.0])  # noqa: E731
    kf.x[:4] = bbox_to_z(box(0))
    for t in range(1, 21):
        kf.predict()
        kf.update(None if 10 <= t < 20 else bbox_to_z(box(t)))
    return kf.x[:, 0]


def test_oru_corrects_velocity_after_a_gap():
    # Without ORU the filter coasts at +5 px/frame through the gap and one update only partly
    # corrects it; ORU re-runs the gap on the straight line between the two observations.
    with_oru, without = run_gap(True), run_gap(False)
    assert abs(with_oru[4]) < abs(without[4])
    assert abs(with_oru[0] - 165.0) < abs(without[0] - 165.0)


def test_oru_freeze_happens_once_and_history_is_rewritten():
    kf = sort_kf()
    kf.x[:4] = bbox_to_z(np.array([0, 0, 10, 20.0]))
    kf.update(bbox_to_z(np.array([0, 0, 10, 20.0])))
    kf.predict()
    kf.update(None)
    saved = kf._saved
    kf.predict()
    kf.update(None)
    assert kf._saved is saved  # frozen at the first miss only
    kf.predict()
    kf.update(bbox_to_z(np.array([6, 0, 16, 20.0])))
    assert all(h is not None for h in kf.history_obs)  # Nones replaced by virtual boxes
    assert len(kf.history_obs) == 4


def test_speed_direction_and_ocm_prefer_consistent_direction():
    v = speed_direction(np.array([0, 0, 10, 10.0]), np.array([10, 0, 20, 10.0]))
    np.testing.assert_allclose(v, [0, 1], atol=1e-6)  # [dy, dx]: moving right
    prev = np.array([[20, 0, 30, 10, 0.9]])
    d = np.array([[30, 0, 40, 10, 1.0], [10, 0, 20, 10, 1.0]])  # ahead vs behind
    c = angle_cost(d, np.array([[0.0, 1.0]]), prev, 0.2)
    assert c[0, 0] == pytest.approx(0.1, abs=1e-3)
    assert c[1, 0] == pytest.approx(-0.1, abs=1e-3)
    # placeholder previous observation (score < 0) disables the term
    assert np.all(angle_cost(d, np.array([[0.0, 1.0]]), -np.ones((1, 5)), 0.2) == 0)


def test_k_previous_obs_rule():
    obs = {3: np.full(5, 3.0), 5: np.full(5, 5.0), 6: np.full(5, 6.0)}
    assert k_previous_obs(obs, 6, 3)[0] == 3.0  # age − Δt first
    assert k_previous_obs(obs, 9, 3)[0] == 6.0  # else newest
    assert k_previous_obs({}, 1, 3)[0] == -1


def ocs(**over):
    ov = [f"{k}={v}" for k, v in over.items()]
    return load_config(CFG / "ocsort.yaml", ["output.min_box_area=0.0", *ov])


def test_ocsort_min_hits_rule():
    trk = OCSortTracker(ocs())
    seen = []
    for t in range(8):
        r = trk.update(dets([(100 + t, 100, 40, 100)], [0.9]))
        seen.append([o.track_id for o in r.tracks])
    # frames 1..3 (frame_count <= min_hits) report; then the track needs 3 consecutive hits
    assert seen[0] == [1]
    assert seen[1] == [1]
    assert seen[2] == [1]
    assert all(s == [1] for s in seen[3:])
    trk2 = OCSortTracker(ocs())
    for _ in range(4):
        trk2.update(dets([], []))
    out = [trk2.update(dets([(100, 100, 40, 100)], [0.9])).tracks for _ in range(5)]
    # born at call 1 (no update), matched at 2, 3, 4 -> hit_streak 3 at call 4
    assert [len(o) for o in out] == [0, 0, 0, 1, 1]


def test_ocr_recovers_track_from_last_observation():
    def scene(ocr: bool) -> set[int]:
        trk = OCSortTracker(ocs(**{"ocsort.ocr": str(ocr).lower(), "ocsort.inertia": 0.0}))
        ids = set()
        for t in range(10):  # moving right fast
            trk.update(dets([(100 + 12 * t, 100, 40, 100)], [0.9]))
        trk.update(dets([], []))  # one miss: the prediction keeps moving right
        # the person stopped: box at the last observation, far from the prediction
        for _ in range(5):
            ids |= {o.track_id for o in trk.update(dets([(208, 100, 40, 100)], [0.9])).tracks}
        return ids

    assert scene(True) == {1}
    assert 1 not in scene(False)


def test_deep_ocsort_adaptive_weight_and_dynamic_alpha():
    e = np.array([[0.9, 0.5], [0.4, 0.8]])
    w = aw_weights(e, 0.75, 0.5)
    # row bonus (top1 − top2)/2, col bonus likewise, capped at max_diff
    np.testing.assert_allclose(
        w, [[0.75 + 0.2 + 0.25, 0.75 + 0.2 + 0.15], [0.75 + 0.2 + 0.25, 0.75 + 0.2 + 0.15]]
    )
    cfg = load_config(CFG / "deep_ocsort.yaml", ["cmc.method=none"])
    trk = OCSortTracker(cfg)
    e1 = np.array([[1.0, 0.0]], np.float32)
    e2 = np.array([[0.0, 1.0]], np.float32)
    trk.update(Detections(np.array([[0, 0, 40, 100.0]]), np.array([0.9]), e1))
    trk.update(Detections(np.array([[1, 0, 40, 100.0]]), np.array([0.6 + 1e-9]), e2))
    # score at det_thresh -> trust 0 -> alpha 1 -> embedding unchanged
    np.testing.assert_allclose(trk.trackers[0].emb, e1[0], atol=1e-6)
    trk.update(Detections(np.array([[2, 0, 40, 100.0]]), np.array([1.0]), e2))
    exp = 0.95 * e1[0] + 0.05 * e2[0]
    np.testing.assert_allclose(trk.trackers[0].emb, exp / np.linalg.norm(exp), atol=1e-6)


def test_hmiou_is_iou_times_height_overlap():
    a = np.array([[0, 0, 10, 10.0]])
    b = np.array([[0, 5, 10, 15.0]])
    assert hmiou(a, b)[0, 0] == pytest.approx(iou_batch(a, b)[0, 0] * 5 / 15)


def test_hybrid_tcm_prefers_score_consistent_detection():
    cfg = load_config(CFG / "hybrid_sort.yaml", ["output.min_box_area=0.0"])
    trk = HybridSortTracker(cfg)
    for _ in range(6):
        trk.update(dets([(100, 100, 40, 100)], [0.95]))
    # two equally overlapping candidates, scores 0.95 and 0.65
    r = trk.update(dets([(104, 100, 40, 100), (96, 100, 40, 100)], [0.65, 0.95]))
    (o,) = [o for o in r.tracks if o.track_id == 1]
    assert o.tlwh[0] == pytest.approx(96)


def test_cmc_file_lookup_and_indexing(tmp_path):
    f = tmp_path / "GMC-X.txt"
    f.write_text("0\t1\t0\t0\t0\t1\t0\n1\t1\t0\t5\t0\t1\t-2\n")
    t = parse_cmc_file(f)
    assert t.shape == (2, 2, 3)
    c = FileCMC(f, 0, 2)
    np.testing.assert_allclose(c.affine(2)[:, 2], [5, -2])
    with pytest.raises(IndexError):
        c.affine(3)
    with pytest.raises(ValueError, match="lines <"):
        FileCMC(f, 1, 2)
    hit = find_cmc_file("MOT17-02-FRCNN", "val_half")
    assert hit is not None
    assert hit[0].parent.name == "MOT17_ablation"
    assert hit[1] == 0
    full = find_cmc_file("MOT17-02-SDP", "train_half", 0)
    assert full is not None
    assert full[0].parent.name == "MOTChallenge"
    assert find_cmc_file("nonexistent-01", "val_half") is None
    meta = CacheMeta("MOT17-02-FRCNN", "MOT17", "val_half", 30.0, 299, 1920, 1080)
    prov, desc = build_cmc("auto", meta)
    assert desc.startswith("file:MOT17_ablation")
    assert prov.affine(1) is not None
    meta2 = CacheMeta("nope", "MOT17", "val_half", 30.0, 10, 10, 10)
    assert build_cmc("auto", meta2)[1].startswith("off")
    with pytest.raises(FileNotFoundError):
        build_cmc("file", meta2)



def test_cmc_absolute_indices_map_rows_to_local_frames(tmp_path):
    f = tmp_path / "GMC-MOT20-01.txt"
    f.write_text("215 1 0 10 0 1 -10\n216 1 0 20 0 1 -20\n217 1 0 30 0 1 -30\n")
    assert parse_cmc_file(f).shape == (3, 2, 3)
    cmc = FileCMC(f, 0, 3)
    np.testing.assert_allclose(cmc.affine(1)[:, 2], [10, -10])
    np.testing.assert_allclose(cmc.affine(3)[:, 2], [30, -30])


def test_cmc_parser_rejects_index_gaps_and_malformed_rows(tmp_path):
    f = tmp_path / "GMC-bad.txt"
    f.write_text("215 1 0 0 0 1 0\n217 1 0 0 0 1 0\n")
    with pytest.raises(ValueError, match="line 1 has index 217"):
        parse_cmc_file(f)
    f.write_text("215 1 0 0 0 1\n")
    with pytest.raises(ValueError, match="expected 7"):
        parse_cmc_file(f)


def test_vendored_cmc_files_cover_the_val_half():
    for seq, n in {
        "02": 299,
        "04": 524,
        "05": 418,
        "09": 262,
        "10": 326,
        "11": 449,
        "13": 374,
    }.items():
        assert parse_cmc_file(ROOT / f"third_party/cmc_files/MOT17_ablation/GMC-MOT17-{seq}.txt"
                              ).shape[0] == n  # fmt: skip


def test_apply_gmc_translation_and_identity():
    m = np.arange(16, dtype=float).reshape(2, 8)
    c = np.stack([np.eye(8)] * 2)
    m2, c2 = apply_gmc(m.copy(), c.copy(), np.eye(2, 3))
    np.testing.assert_allclose(m2, m)
    np.testing.assert_allclose(c2, c)
    w = np.array([[1, 0, 3.0], [0, 1, -4.0]])
    m3, _ = apply_gmc(m.copy(), c.copy(), w)
    np.testing.assert_allclose(m3[:, :2] - m[:, :2], [[3, -4], [3, -4]])
    np.testing.assert_allclose(m3[:, 2:], m[:, 2:])


def test_botsort_upstream_switches():
    a = np.array([[0, 0, 9, 9.0]])
    b = np.array([[0, 0, 4, 9.0]])
    assert iou_matrix(a, b, plus_one=True)[0, 0] == pytest.approx(50 / 100)
    assert iou_matrix(a, b)[0, 0] == pytest.approx(36 / 81)
    cfg = load_config(CFG / "botsort.yaml", ["cmc.method=none", "output.min_box_area=0.0"])
    trk = ByteTrackLike(cfg)
    trk.update(dets([(0, 0, 40, 100)], [0.9]))  # frame 1: confirmed immediately
    r = trk.update(dets([(0, 0, 40, 100), (300, 0, 40, 100)], [0.9, 0.9]))
    assert len(r.tracks) == 2  # newborn tentative track reported (report_tentative)
    tent = next(t for t in trk.tracks if t.track_id == 2)
    before = tent.mean.copy()
    trk.update(dets([(0, 0, 40, 100)], [0.9]))
    assert tent.state.value == "removed"
    np.testing.assert_allclose(tent.mean, before)  # not predicted (predict_tentative false)


def test_tracker_registry_and_cmc_guard():
    for n in ("ocsort", "deep_ocsort", "hybrid_sort", "botsort", "bytetrack"):
        trk, info = build_tracker(load_config(CFG / f"{n}.yaml"), 30.0, None)
        assert hasattr(trk, "update")
        assert info["cmc"] == "off"
    with pytest.raises(ValueError, match="no CMC"):
        build_tracker(load_config(CFG / "ocsort.yaml", ["cmc.method=file"]), 30.0, None)


@pytest.mark.parametrize("impl", ["ocsort", "deep", "hybrid", "botsort"])
def test_recorded_upstream_parity_is_exact(impl):
    rep = json.loads((ROOT / "docs" / "results" / "parity" / f"{impl}.json").read_text())
    assert rep["frames"] == 2652
    assert rep["mismatched_frames"] == 0

"""Eval pipeline: half splits, TrackEval wrapper, stress bins (checked against TrackEval)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval.latency import summarise_latency
from eval.oracle import write_oracle
from eval.run import benchmark_for
from eval.splits import crop_cache, crop_rows, half_range, write_gt_split, write_seqmap
from eval.stress import density_edges, run_stress
from eval.trackeval_runner import evaluate
from tracker.io import CacheMeta, SequenceCache

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_half_range_full_aliases() -> None:
    for s in ("full", "train", "val"):
        assert half_range(7, s) == (1, 7)
    assert half_range(7, "train_half") == (1, 4)
    assert half_range(7, "val_half") == (5, 7)


def test_crop_rows_renumbers() -> None:
    rows = np.array([[1, 1, 0, 0, 1, 1], [5, 2, 0, 0, 1, 1], [7, 3, 0, 0, 1, 1]], float)
    out = crop_rows(rows, 5, 7)
    assert out[:, 0].tolist() == [1, 3]
    assert out[:, 1].tolist() == [2, 3]


def _cache(n: int) -> SequenceCache:
    meta = CacheMeta("S", "MOT17", "train", 30.0, n, 100, 100)
    frames = np.arange(1, n + 1, dtype=np.int32)
    tlwh = np.tile(np.array([1, 1, 10, 20], np.float32), (n, 1))
    return SequenceCache(meta, frames, tlwh, np.full(n, 0.9, np.float32))


def test_crop_cache_val_half() -> None:
    c = crop_cache(_cache(600), "val_half")
    assert c.meta.seq_length == 299
    assert c.meta.split == "val_half"
    assert c.meta.extra["frame_offset"] == 301
    assert c.frame[0] == 1
    assert c.frame[-1] == 299
    assert len(list(c.frames())) == 299
    assert crop_cache(_cache(10), "full").meta.seq_length == 10


def test_benchmark_for() -> None:
    assert benchmark_for("MOT20") == "MOT20"
    assert benchmark_for("MOT17") == "MOT17"
    assert benchmark_for("DanceTrack") == "MOT17"


def test_summarise_latency() -> None:
    s = summarise_latency(np.arange(1.0, 101.0))
    assert s["frames"] == 100
    assert s["max_ms"] == 100.0
    assert s["p50_ms"] == pytest.approx(50.5)
    assert summarise_latency(np.array([]))["frames"] == 0


# --------------------------------------------------------------- synthetic TrackEval fixture
def _make_seq(root: Path, name: str, n: int, gt_rows: list[list[float]]) -> Path:
    seq = root / "src" / name
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength={n}\n"
        "imWidth=1000\nimHeight=1000\nimExt=.jpg\n"
    )
    np.savetxt(seq / "gt" / "gt.txt", np.array(gt_rows), delimiter=",", fmt="%g")
    return seq


def _synthetic(tmp_path: Path) -> tuple[Path, Path, list[list[float]]]:
    """Two people over 6 frames (val half = frames 5..6 of a 10-frame seq is too short, so use
    the full split); person 2 has low visibility; plus a distractor (class 7)."""
    gt = []
    for f in range(1, 7):
        gt.append([f, 1, 100, 100, 50, 100, 1, 1, 1.0])
        gt.append([f, 2, 400, 100, 50, 100, 1, 1, 0.2 if f <= 3 else 0.9])
        gt.append([f, 3, 700, 100, 50, 100, 1, 7, 1.0])  # static person: distractor
    seq = _make_seq(tmp_path, "SYN-01", 6, gt)
    gt_dir = tmp_path / "gt"
    write_gt_split(seq, gt_dir, "full")
    write_seqmap(gt_dir / "seqmap.txt", ["SYN-01"])
    trk = []
    for f in range(1, 7):
        trk.append([f, 10, 100, 100, 50, 100, 1, -1, -1, -1])  # perfect on person 1
        if f >= 2:  # person 2: missed at f1, id switch 20 -> 21 at f4
            trk.append([f, 20 if f <= 3 else 21, 402, 101, 50, 100, 1, -1, -1, -1])
        trk.append([f, 30, 700, 100, 50, 100, 1, -1, -1, -1])  # on the distractor: ignored
        if f == 5:
            trk.append([f, 40, 900, 900, 30, 60, 1, -1, -1, -1])  # one FP
    tdir = tmp_path / "trackers" / "syn" / "data"
    tdir.mkdir(parents=True)
    np.savetxt(tdir / "SYN-01.txt", np.array(trk), delimiter=",", fmt="%g")
    return gt_dir, tmp_path / "trackers", gt


def test_trackeval_and_stress_agree_on_synthetic(tmp_path: Path) -> None:
    gt_dir, trk_dir, _ = _synthetic(tmp_path)
    ref = evaluate(gt_dir, trk_dir, "MOT17", ["syn"], write_outputs=False)["syn"]
    c = ref["COMBINED"]
    assert (c["GT"], c["TP"], c["FN"], c["FP"], c["IDSW"]) == (12, 11, 1, 1, 1)
    st = run_stress(gt_dir, trk_dir, "syn", "MOT17", check_against=ref, density=[0, 2])
    vis = {r["bin"]: r for r in st["visibility"]}
    assert vis["0.00-0.25"]["gt"] == 3
    assert vis["0.00-0.25"]["fn"] == 1
    assert vis["0.75-1.00"]["gt"] == 9
    assert vis["0.75-1.00"]["idsw"] == 1
    dens = {r["bin"]: r for r in st["density"]}
    assert dens["2+"]["frames"] == 6
    assert dens["2+"]["mota"] == pytest.approx(c["MOTA"])


def test_stress_raises_when_totals_disagree(tmp_path: Path) -> None:
    gt_dir, trk_dir, _ = _synthetic(tmp_path)
    ref = evaluate(gt_dir, trk_dir, "MOT17", ["syn"], write_outputs=False)["syn"]
    ref["COMBINED"]["IDSW"] += 1
    with pytest.raises(AssertionError, match="IDSW"):
        run_stress(gt_dir, trk_dir, "syn", "MOT17", check_against=ref)


def test_oracle_is_perfect_without_low_vis_filter(tmp_path: Path) -> None:
    gt_dir, _, _ = _synthetic(tmp_path)
    out = tmp_path / "oracle"
    write_oracle(gt_dir, ["SYN-01"], out / "o0", tau=-1.0)
    write_oracle(gt_dir, ["SYN-01"], out / "o3", tau=0.3)
    res = evaluate(gt_dir, out, "MOT17", ["o0", "o3"], write_outputs=False)
    assert res["o0"]["COMBINED"]["MOTA"] == pytest.approx(100.0)
    assert res["o3"]["COMBINED"]["FN"] == 3


def test_density_edges_quartiles() -> None:
    assert density_edges(np.array([0, 4, 8, 12, 16, 20])) == [0, 8, 12, 16]
    assert density_edges(np.zeros(3)) == [0]


# --------------------------------------------------------------- regression on real labels
SEQ09 = DATA / "MOT17" / "train" / "MOT17-09-FRCNN"


@pytest.mark.data
@pytest.mark.skipif(not (SEQ09 / "det" / "det.txt").exists(), reason="MOT17 labels not in data/")
def test_regression_mot17_09_val_half_public(tmp_path: Path) -> None:
    """Pins the phase-1 smoke numbers for one sequence; any tracker change shows up here."""
    from eval.run import run

    root = tmp_path / "MOT17"
    (root / "train").mkdir(parents=True)
    (root / "train" / SEQ09.name).symlink_to(SEQ09)
    info = run("MOT17", root, "val_half", "public", ROOT / "configs/baselines/bytetrack.yaml",
               [], "bt", tmp_path / "out")  # fmt: skip
    c = info["metrics"]["COMBINED"]
    assert c["GT"] == 2879
    assert round(c["HOTA"], 1) == 52.7
    assert round(c["IDF1"], 1) == 59.5
    assert round(c["MOTA"], 1) == 51.6
    assert c["IDSW"] == 5
    assert info["stress"]["verified_against_trackeval"]

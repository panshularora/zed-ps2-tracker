"""Tests for the CPU-checkable helpers used by the DGX scripts."""

from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np
import pytest

from models.mot_to_coco import convert, half_range
from models.prepare import prepare_mot_like


def test_half_range_matches_bytetrack_convention() -> None:
    assert half_range(600, "train_half") == (1, 301)
    assert half_range(600, "val_half") == (302, 600)
    assert half_range(525, "train_half") == (1, 263)
    assert half_range(525, "val_half") == (264, 525)
    assert half_range(10, "train") == (1, 10)
    with pytest.raises(ValueError, match="bogus"):
        half_range(10, "bogus")


def _seq(root: Path, name: str, n: int, gt: np.ndarray | None) -> Path:
    seq = root / name
    (seq / "img1").mkdir(parents=True)
    for f in range(1, n + 1):
        (seq / "img1" / f"{f:06d}.jpg").touch()
    if gt is not None:
        (seq / "gt").mkdir()
        np.savetxt(seq / "gt" / "gt.txt", gt, delimiter=",", fmt="%g")
    return seq


def test_convert_keeps_only_train_half_people_and_clips(tmp_path: Path) -> None:
    gt = np.array([
        [1, 1, -10, 5, 30, 40, 1, 1, 1.0],   # clipped at x=0
        [3, 1, 10, 5, 30, 40, 1, 1, 1.0],
        [4, 2, 10, 5, 30, 40, 1, 1, 1.0],    # val half -> dropped
        [2, 3, 10, 5, 30, 40, 0, 1, 1.0],    # ignored -> dropped
        [2, 4, 10, 5, 30, 40, 1, 7, 1.0],    # static person class -> dropped
    ])  # fmt: skip
    seq = _seq(tmp_path / "train", "S-01", 4, gt)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=S-01\nseqLength=4\nimWidth=100\nimHeight=80\nframeRate=30\n"
    )
    coco = convert(tmp_path, "train_half")
    images, anns = coco["images"], coco["annotations"]
    assert isinstance(images, list)
    assert isinstance(anns, list)
    assert [im["file_name"] for im in images] == [f"train/S-01/img1/{f:06d}.jpg" for f in (1, 2, 3)]
    assert len(anns) == 2
    assert anns[0]["bbox"] == [0.0, 5.0, 20.0, 40.0]
    assert anns[1]["image_id"] == 3


def test_prepare_splits_by_gt_and_writes_seqinfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import models.prepare as prep

    real = prep._image_info

    def fake_image_info(img_dir: Path) -> tuple[int, str, int, int]:
        try:
            return real(img_dir)
        except ModuleNotFoundError:  # cv2 lives in the Stage A env, not the dev env
            return len(list(img_dir.iterdir())), ".jpg", 64, 48

    monkeypatch.setattr(prep, "_image_info", fake_image_info)
    root = tmp_path / "sompt"
    tr = _seq(root / "train", "SOMPT22-02", 3, np.array([[1, 1, 0, 0, 5, 5, 1, 1, 1.0]]))
    te = _seq(root / "test", "SOMPT22-01", 2, None)
    del tr, te
    out = prepare_mot_like(root, fps=25)
    assert out == {"train": ["SOMPT22-02"], "test": ["SOMPT22-01"]}
    cp = configparser.ConfigParser()
    cp.read(root / "_seqs" / "test" / "SOMPT22-01" / "seqinfo.ini")
    s = cp["Sequence"]
    assert (s["seqLength"], s["imWidth"], s["imHeight"], s["frameRate"]) == ("2", "64", "48", "25")
    # idempotent
    assert prepare_mot_like(root, fps=25) == out

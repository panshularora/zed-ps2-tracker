"""models.pack: bring_back/ → per-dataset tars, each carrying the manifest."""

import tarfile
from pathlib import Path

from models.pack import pack


def test_pack_groups_and_index(tmp_path: Path) -> None:
    b = tmp_path / "bring_back"
    for rel in ["MANIFEST.json", "SHA256SUMS", "weights_SHA256SUMS", "gt/MOT17/train/A/gt/gt.txt",
                "logs/run.log", "cache/MOT17/ablation/train/A.npz",
                "cache/MOT17/ablation/train/stage_a_log.jsonl",
                "cache/MOT20/x/train/B.npz", "clips/a.mp4"]:  # fmt: skip
        (b / rel).parent.mkdir(parents=True, exist_ok=True)
        (b / rel).write_text(rel)
    sizes = pack(b, tmp_path / "tars")
    assert set(sizes) == {"core.tar", "MOT17__ablation.tar", "MOT20__x.tar", "clips.tar"}
    with tarfile.open(tmp_path / "tars" / "MOT17__ablation.tar") as tf:
        assert tf.getnames() == ["MANIFEST.json", "SHA256SUMS", "cache/MOT17/ablation/train/A.npz"]
    with tarfile.open(tmp_path / "tars" / "core.tar") as tf:
        names = set(tf.getnames())
    assert "cache/MOT17/ablation/train/stage_a_log.jsonl" in names
    assert "gt/MOT17/train/A/gt/gt.txt" in names

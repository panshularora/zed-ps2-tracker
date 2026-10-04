"""Viewer exporter: clip structure, pending grouping, id-switch events."""

import gzip
import json
from pathlib import Path

import numpy as np

from eval.export_viewer import build_clip, id_switch_events, pending_entries, write_clip
from eval.gate_events import Gate
from tracker.types import FrameResult, PendingDecision, TrackOutput


def test_id_switch_events() -> None:
    gt = np.array([[f, 1, 0, 0, 10, 10, 1, 1, 1.0] for f in range(1, 6)])
    tr = np.array([[f, 5 if f < 3 else 6, 0, 0, 10, 10, 1.0] for f in range(1, 6)])
    assert id_switch_events(gt, tr) == [{"f": 3, "type": "id_switch", "gt": 1, "from": 5, "to": 6}]


def test_pending_groups_by_contested_detection() -> None:
    box_a, box_b = (0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 10.0, 10.0)
    pend = (
        PendingDecision(1, (box_a, box_b), (0.3, 0.4), 7, 12, chosen_index=1),
        PendingDecision(2, (box_a,), (0.32,), 7, 12, chosen_index=0),
    )
    res = [FrameResult(7, (TrackOutput(1, box_b, 0.9, "confirmed"),), pend)]
    assert pending_entries(res) == [{"f0": 7, "f1": 12, "candidates": [1, 2], "chosen": 2}]


def test_build_and_write_clip(tmp_path: Path) -> None:
    ys = [700.0 - 20 * i for i in range(6)]
    gt = np.array([[i + 1, 1, 100, y - 50, 20, 50, 1, 1, 1.0] for i, y in enumerate(ys)])
    rows = np.c_[gt[:, :6], np.full(len(gt), 0.9)]
    rows[:, 1] = 9
    gate = Gate("h", (0.0, 648.0), (1920.0, 648.0))
    params = {"hysteresis_px": 10, "delta_frames": 15, "tailgate_seconds": 1.0}
    clip = build_clip("MOT17-XX", "t", 30.0, 6, (1920, 1080), rows, gt, gate, params, [], False)
    assert set(clip) == {"clip", "tracker", "fps", "frames", "width", "height", "gate", "tracks",
                         "pending", "events"}  # fmt: skip
    assert clip["tracks"]["9"][0] == [1, 100.0, 650.0, 20.0, 50.0, 0.9]
    cross = [e for e in clip["events"] if e["type"] == "cross"]
    assert cross == [{"f": 5, "type": "cross", "id": 9, "dir": "out", "correct": True}]
    p = write_clip(clip, tmp_path)
    assert p.name == "MOT17-XX.t.json.gz"
    with gzip.open(p) as fh:
        assert json.loads(fh.read()) == clip


def test_media_spec() -> None:
    from eval.export_viewer import media_spec

    assert media_spec("clips/MOT17-04.mp4", 526) == {
        "type": "video", "src": "clips/MOT17-04.mp4", "frame_offset": 526,
    }  # fmt: skip
    assert media_spec("frames/MOT17-04/%06d.jpg", 0)["type"] == "frames"


def test_export_writes_media_with_split_offset(tmp_path: Path) -> None:
    from eval.export_viewer import export

    root = Path("data/MOT17")
    if not (root / "train" / "MOT17-09-FRCNN").exists():
        import pytest

        pytest.skip("needs `make data`")
    paths = export("MOT17-09-FRCNN", {}, tmp_path, None, root=root, include_gt=True,
                   media="clips/MOT17-09.mp4")  # fmt: skip
    with gzip.open(paths[0]) as fh:
        clip = json.loads(fh.read())
    # MOT17-09 has 525 frames; val-half starts at frame 525 // 2 + 2 = 264 -> offset 263
    assert clip["media"] == {"type": "video", "src": "clips/MOT17-09.mp4", "frame_offset": 263}

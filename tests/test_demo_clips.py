"""Demo assets: clip manifest (consent / styling rules), YOLOX ONNX decoding, face blurring."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

import cv2

from serve.clips import DemoClip, load_manifest
from serve.detector import decode, nms, postprocess

ROOT = Path(__file__).resolve().parent.parent


def write_video(path: Path, n: int, size: tuple[int, int] = (320, 240), fps: float = 10.0) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for f in range(n):
        img = np.zeros((size[1], size[0], 3), np.uint8)
        cv2.rectangle(img, (40 + 4 * f, 40), (80 + 4 * f, 160), (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    return path


def test_real_manifest_is_neutral_and_valid() -> None:
    clips = load_manifest(ROOT / "configs" / "demo" / "clips.yaml")
    assert {c.id for c in clips} >= {"subway_turnstile", "mexico_city_entrance"}
    for c in clips:
        assert c.consent is False
        assert c.event_styling == "neutral"  # Pexels: no alerts
        assert c.page.startswith("https://www.pexels.com/video/")


def test_alert_styling_requires_consent() -> None:
    with pytest.raises(ValueError, match="consent"):
        DemoClip(id="x", title="x", file="x.mp4", page="x", consent=False, event_styling="alert")
    with pytest.raises(ValueError, match="gate"):
        DemoClip(id="x", title="x", file="x.mp4", page="x", gate=(0.0, 0.5, 1.2, 0.5))


def test_yolox_decode_and_postprocess() -> None:
    size = 64  # grids 8x8, 4x4, 2x2 -> 84 anchors
    raw = np.zeros((84, 85), np.float32)
    raw[:, 4] = 0.0
    # anchor 9 = stride-8 grid cell (1, 1): centre (1+0.5)*8 = 12 when the offset is 0.5
    raw[9, :4] = [0.5, 0.5, np.log(2.0), np.log(3.0)]  # w = 16, h = 24
    raw[9, 4], raw[9, 5] = 0.9, 0.8  # obj × person = 0.72
    raw[10, :4] = [0.5, 0.5, np.log(2.0), np.log(3.0)]  # cell (2, 1): another box
    raw[10, 4], raw[10, 5] = 0.9, 0.1  # 0.09 < floor
    d = decode(raw, size)
    assert d.shape == (84, 85)
    np.testing.assert_allclose(d[9, :4], [12.0, 12.0, 16.0, 24.0], rtol=1e-6)
    tlwh, score = postprocess(
        raw, size, ratio=0.5, img_wh=(128, 128), score_floor=0.1, nms_iou=0.45
    )
    assert len(score) == 1
    assert score[0] == pytest.approx(0.72)
    np.testing.assert_allclose(tlwh[0], [8.0, 0.0, 32.0, 48.0], atol=1e-5)  # /0.5, clipped at 0


def test_nms() -> None:
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], float)
    keep = nms(b, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]


def test_head_boxes_and_blur_regions() -> None:
    from serve.blur import blur_regions, head_boxes

    tl = np.array([[100, 50, 40, 120], [0, 0, 10, 10]], np.float32)
    hb = head_boxes(tl, np.array([0.9, 0.9], np.float32))
    assert len(hb) == 1  # the 10 px box is too small to have a recognisable face
    x1, y1, x2, y2 = hb[0]
    assert x1 == pytest.approx(106)
    assert x2 == pytest.approx(134)
    assert y1 < 50 < y2 <= 50 + 0.22 * 120 + 1e-4
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    before = img.copy()
    blur_regions(img, np.array([[20, 20, 60, 60]], np.float32))
    assert not np.array_equal(img[20:60, 20:60], before[20:60, 20:60])
    assert img[20:60, 20:60].std() < before[20:60, 20:60].std() / 3  # detail removed
    assert np.array_equal(img[70:, 70:], before[70:, 70:])  # outside untouched


@pytest.mark.skipif(
    not (ROOT / "data/models/face_detection_yunet_2023mar.onnx").exists()
    or shutil.which("ffmpeg") is None,
    reason="needs data/models + ffmpeg",
)
def test_blur_video_keyframe_interval(tmp_path: Path) -> None:
    import subprocess

    from serve.blur import blur_video

    src = write_video(tmp_path / "in.mp4", 30)
    info = blur_video(src, tmp_path / "out.mp4", ROOT / "data/models", gop=10)
    assert info["frames"] == 30
    types = (
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v",
                "-show_entries",
                "frame=pict_type",
                "-of",
                "csv=p=0",
                str(tmp_path / "out.mp4"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.replace(",", "")
        .split()
    )
    assert len(types) == 30
    assert [i for i, t in enumerate(types) if t == "I"] == [0, 10, 20]

"""Face blurring for the demo clips (Pexels licence: no identifiable people shown in a bad light).

``python -m serve.blur --src data/demo/raw --out data/demo/blurred`` re-encodes every manifest
clip with the faces and heads blurred, at the same 1080p H.264 settings (CRF 20, no audio) plus
a **keyframe every 10 frames** (``-g 10``) so that frame-accurate seeking in the browser stays
cheap. The demo service lists and serves only these copies.

Two detectors, both CPU (OpenCV + onnxruntime), are combined so that a miss of one is covered by
the other:

* **faces**: OpenCV's YuNet (``face_detection_yunet_2023mar.onnx``, OpenCV Zoo, MIT) on the full
  frame, score ≥ 0.4, box grown by 40 %;
* **heads**: the top of every YOLOX person box (score ≥ 0.3), which also catches back-turned,
  profile and very small faces that YuNet misses.

Boxes are held for ±``hold`` frames (default 2) so that a one-frame detector miss does not
flash a face. Blurring is pixelate-then-Gaussian, strong enough that faces are not recognisable.
This is best effort, not a guarantee: check the output before publishing (see docs/licenses.md).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from serve.clips import DEFAULT_MANIFEST, ROOT, load_manifest
from serve.detector import MODELS, YUNET, YoloxDetector, model_path


def face_boxes(fd: object, bgr: np.ndarray, grow: float = 0.4) -> np.ndarray:
    """``(n, 4)`` xyxy face boxes from YuNet, grown by ``grow`` around the centre."""
    _, faces = fd.detect(bgr)  # type: ignore[attr-defined]
    if faces is None or len(faces) == 0:
        return np.zeros((0, 4), np.float32)
    x, y, w, h = (faces[:, i] for i in range(4))
    cx, cy, s = x + w / 2, y + h / 2, 1.0 + grow
    return np.stack([cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2], 1)


def head_boxes(tlwh: np.ndarray, score: np.ndarray, min_score: float = 0.3) -> np.ndarray:
    """The head region of each person box: centred, 0.7 w wide, top 0.22 h (≤ 0.9 w) tall."""
    keep = (score >= min_score) & (tlwh[:, 3] >= 24)
    t = tlwh[keep]
    if len(t) == 0:
        return np.zeros((0, 4), np.float32)
    x, y, w, h = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    hw = 0.7 * w
    hh = np.minimum(0.22 * h, 0.9 * w)
    cx = x + w / 2
    return np.stack([cx - hw / 2, y - 0.03 * h, cx + hw / 2, y + hh], 1)


def blur_regions(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Pixelate (1/10) then Gaussian-blur each box, in place; returns ``img``."""
    import cv2

    hgt, wid = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        a, b = max(0, int(x1)), max(0, int(y1))
        c, d = min(wid, int(np.ceil(x2))), min(hgt, int(np.ceil(y2)))
        if c - a < 2 or d - b < 2:
            continue
        roi = img[b:d, a:c]
        small = cv2.resize(
            roi, (max(1, (c - a) // 10), max(1, (d - b) // 10)), interpolation=cv2.INTER_AREA
        )
        big = cv2.resize(small, (c - a, d - b), interpolation=cv2.INTER_LINEAR)
        k = max(3, (min(c - a, d - b) // 4) | 1)
        img[b:d, a:c] = cv2.GaussianBlur(big, (k, k), 0)
    return img


def blur_video(
    src: Path,
    dst: Path,
    model_dir: Path,
    hold: int = 2,
    gop: int = 10,
    detector: str = "yolox_tiny",
    threads: int = 2,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, float]:
    import cv2

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise OSError(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fd = cv2.FaceDetectorYN.create(str(model_dir / YUNET[0]), "", (w, h), 0.4, 0.3, 5000)
    det = YoloxDetector(model_path(detector, model_dir), MODELS[detector][0], 0.3, threads=threads)
    tmp = dst.with_suffix(".tmp.mp4")
    enc = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-an",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        stdin=subprocess.PIPE,
    )
    assert enc.stdin is not None
    frames: deque[np.ndarray] = deque()
    boxes: deque[np.ndarray] = deque()
    n, n_faces, n_heads, t0 = 0, 0, 0, time.perf_counter()

    history: deque[np.ndarray] = deque(maxlen=hold)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        if progress and n % 5 == 0:
            try:
                progress("blur", min(n, total), total)
            except BaseException:
                enc.kill()
                cap.release()
                raise
        f = face_boxes(fd, img)
        tl, sc = det.detect(img)
        hb = head_boxes(tl, sc)
        n_faces += len(f)
        n_heads += len(hb)
        frames.append(img)
        boxes.append(np.concatenate([f, hb]).astype(np.float32))
        if len(frames) > hold:
            # window = the hold previous frames' boxes + this frame + hold following frames
            front = np.concatenate([*history, *list(boxes)[: hold + 1]]) if boxes else None
            img0 = frames.popleft()
            history.append(boxes.popleft())
            if front is not None and len(front):
                blur_regions(img0, front)
            enc.stdin.write(img0.tobytes())
    while frames:
        front = np.concatenate([*history, *list(boxes)[: hold + 1]])
        img0 = frames.popleft()
        history.append(boxes.popleft())
        if len(front):
            blur_regions(img0, front)
        enc.stdin.write(img0.tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {src}")
    tmp.replace(dst)
    secs = time.perf_counter() - t0
    return {
        "frames": n,
        "fps": fps,
        "faces": n_faces,
        "heads": n_heads,
        "seconds": round(secs, 1),
        "gop": gop,
        "hold": hold,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "demo" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "demo" / "blurred")
    ap.add_argument("--models", type=Path, default=ROOT / "data" / "models")
    ap.add_argument("--only", nargs="*", help="clip ids")
    ap.add_argument("--hold", type=int, default=2)
    ap.add_argument("--gop", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    log = a.out / "blur_log.jsonl"
    for c in load_manifest(a.manifest):
        if a.only and c.id not in a.only:
            continue
        src, dst = a.src / c.file, a.out / c.file
        if not src.exists():
            print(f"missing {src} (run `make clips`)")
            continue
        if dst.exists() and not a.force:
            print(f"have    {dst}")
            continue
        info = blur_video(src, dst, a.models, a.hold, a.gop)
        print(f"blurred {dst}: {info}")
        with log.open("a") as fh:
            fh.write(json.dumps({"clip": c.id, "file": c.file, **info}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

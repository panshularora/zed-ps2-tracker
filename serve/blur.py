"""Face blurring for the demo clips (Pexels licence: no identifiable people shown in a bad light).

``python -m serve.blur --src data/demo/raw --out data/demo/blurred`` re-encodes every manifest
clip with the faces blurred (``--mode face``, the default: a soft ellipse about 1.2x each face,
bodies left sharp; ``--mode head`` keeps the older pixelated head squares), at the same 1080p
H.264 settings (CRF 20, no audio) plus
a **keyframe every 10 frames** (``-g 10``) so that frame-accurate seeking in the browser stays
cheap. The demo service lists and serves only these copies.

Two detectors, both CPU (OpenCV + onnxruntime), are combined so that a miss of one is covered by
the other:

* **faces**: OpenCV's YuNet (``face_detection_yunet_2023mar.onnx``, OpenCV Zoo, MIT) on the full
  frame, score ≥ 0.4 (``head`` mode grows the box by 40 %);
* **person-box faces**: a face-sized box at the top of every YOLOX person box (``face`` mode,
  score ≥ 0.35; ``head`` mode uses the whole head, score ≥ 0.3), which also catches
  back-turned, profile and very small faces that YuNet misses.

Uploads (``serve.api``: the blurred copy that is tracked and served, and the first-frame still)
use the default ``face`` mode.

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


def face_from_person(tlwh: np.ndarray, score: np.ndarray, min_score: float = 0.35) -> np.ndarray:
    """Face-sized box at the top of each person box (catches profile / small faces YuNet misses).

    Only the face area: ~0.34 w wide and ~0.11 h tall, so shoulders and clothes stay sharp.
    """
    keep = (score >= min_score) & (tlwh[:, 3] >= 40)
    t = tlwh[keep]
    if len(t) == 0:
        return np.zeros((0, 4), np.float32)
    x, y, w, h = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    fw = 0.34 * w
    fh = np.minimum(0.11 * h, 0.5 * w)
    cx, cy = x + w / 2, y + 0.075 * h
    return np.stack([cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2], 1)


def blur_faces_soft(img: np.ndarray, boxes: np.ndarray, scale: float = 1.2) -> np.ndarray:
    """Soft elliptical blur over each box (grown to ``scale``x), in place; returns ``img``.

    Inside the ellipse: pixelate (1/8) + Gaussian; the edge is feathered so there is no hard
    rectangle. Everything outside the ellipses is untouched (bodies stay sharp).
    """
    import cv2

    hgt, wid = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rw, rh = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
        if rw < 1.5 or rh < 1.5:
            continue
        pad = 0.25
        a, b = max(0, int(cx - rw * (1 + pad))), max(0, int(cy - rh * (1 + pad)))
        c = min(wid, int(np.ceil(cx + rw * (1 + pad))))
        d = min(hgt, int(np.ceil(cy + rh * (1 + pad))))
        if c - a < 3 or d - b < 3:
            continue
        roi = img[b:d, a:c].astype(np.float32)
        small = cv2.resize(
            roi, (max(1, (c - a) // 8), max(1, (d - b) // 8)), interpolation=cv2.INTER_AREA
        )
        big = cv2.resize(small, (c - a, d - b), interpolation=cv2.INTER_LINEAR)
        k = max(3, (min(c - a, d - b) // 3) | 1)
        big = cv2.GaussianBlur(big, (k, k), 0)
        mask = np.zeros((d - b, c - a), np.float32)
        cv2.ellipse(
            mask, (int(cx - a), int(cy - b)), (max(1, int(rw)), max(1, int(rh))), 0, 0, 360, 1.0, -1
        )
        fk = max(3, (int(min(rw, rh) * 0.5)) | 1)
        mask = cv2.GaussianBlur(mask, (fk, fk), 0)[..., None]
        img[b:d, a:c] = (roi * (1 - mask) + big * mask).clip(0, 255).astype(np.uint8)
    return img


def frame_boxes(
    fd: object, tl: np.ndarray, sc: np.ndarray, img: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """YuNet faces and person-box faces (``face``) or heads (``head``) for one frame."""
    if mode == "face":
        return face_boxes(fd, img, grow=0.0), face_from_person(tl, sc)
    return face_boxes(fd, img), head_boxes(tl, sc)


def _no_blur(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    return img


def blur_with(mode: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """``face``: soft ovals; ``head``: pixelated squares; ``none``: unchanged (tests only)."""
    if mode not in ("face", "head", "none"):
        raise ValueError(f"unknown blur mode {mode!r}")
    return blur_faces_soft if mode == "face" else blur_regions if mode == "head" else _no_blur


def blur_frame(img: np.ndarray, model_dir: Path, *, detector: str = "yolox_tiny",
               threads: int = 2, mode: str = "face") -> np.ndarray:  # fmt: skip
    """Face-blur one frame exactly like :func:`blur_video` (same detectors, same ``mode``)."""
    import cv2

    h, w = img.shape[:2]
    fd = cv2.FaceDetectorYN.create(str(model_dir / YUNET[0]), "", (w, h), 0.4, 0.3, 5000)
    det = YoloxDetector(model_path(detector, model_dir), MODELS[detector][0], 0.3, threads=threads)
    tl, sc = det.detect(img)
    boxes = np.concatenate(frame_boxes(fd, tl, sc, img, mode)).astype(np.float32)
    out = img.copy()
    if len(boxes):
        blur_with(mode)(out, boxes)
    return out


def blur_video(
    src: Path,
    dst: Path,
    model_dir: Path,
    hold: int = 2,
    gop: int = 10,
    detector: str = "yolox_tiny",
    threads: int = 2,
    progress: Callable[[str, int, int], None] | None = None,
    mode: str = "face",
) -> dict[str, float | str]:
    """``mode="face"`` (default, used for uploads): soft ellipses over faces only, bodies sharp.
    ``mode="head"``: the older, heavier pixelated head rectangles."""
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
    blur = blur_with(mode)
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
        tl, sc = det.detect(img)
        f, hb = frame_boxes(fd, tl, sc, img, mode)
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
                blur(img0, front)
            enc.stdin.write(img0.tobytes())
    while frames:
        front = np.concatenate([*history, *list(boxes)[: hold + 1]])
        img0 = frames.popleft()
        history.append(boxes.popleft())
        if len(front):
            blur(img0, front)
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
        "mode": mode,
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
    ap.add_argument("--mode", choices=["face", "head"], default="face")
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
        info = blur_video(src, dst, a.models, a.hold, a.gop, mode=a.mode)
        print(f"blurred {dst}: {info}")
        with log.open("a") as fh:
            fh.write(json.dumps({"clip": c.id, "file": c.file, **info}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

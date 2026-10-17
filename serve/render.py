"""Annotated clip for a finished job: the face-blurred video with the tracker's boxes, numbers and
the gate line burnt in (``GET /api/jobs/{id}/annotated.mp4``). Rendered once, then cached in the
job directory. Only ever reads the blurred copy."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

PALETTE = [
    (66, 135, 245),
    (245, 165, 36),
    (80, 200, 120),
    (230, 90, 90),
    (170, 110, 230),
    (60, 200, 210),
    (240, 120, 200),
    (150, 150, 60),
]


def colour(tid: int) -> tuple[int, int, int]:
    r, g, b = PALETTE[tid % len(PALETTE)]
    return (b, g, r)


def render_annotated(video: Path, result: dict[str, Any], dst: Path) -> Path:
    import cv2

    stride = int(result["video"]["stride"])
    by_frame: dict[int, list[tuple[int, list[float]]]] = {}
    for tid, rows in result["tracks"].items():
        for r in rows:
            by_frame.setdefault(int(r[0]), []).append((int(tid), r[1:5]))
    g = result["gate"]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
            "23",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        stdin=subprocess.PIPE,
    )
    assert enc.stdin is not None
    last: list[tuple[int, list[float]]] = []
    idx = 0
    th = max(1, round(h / 360))
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                last = by_frame.get(idx // stride + 1, [])
            a = tuple(int(v) for v in g["a"])
            b = tuple(int(v) for v in g["b"])
            cv2.line(img, a, b, (36, 165, 245), 2 * th)
            for tid, (x, y, bw, bh) in last:
                c = colour(tid)
                p1, p2 = (int(x), int(y)), (int(x + bw), int(y + bh))
                cv2.rectangle(img, p1, p2, c, 2 * th)
                cv2.putText(
                    img,
                    f"#{tid}",
                    (int(x), max(12, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * th,
                    c,
                    th + 1,
                    cv2.LINE_AA,
                )
            enc.stdin.write(np.ascontiguousarray(img).tobytes())
            idx += 1
    finally:
        cap.release()
        enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed while rendering the annotated clip")
    tmp.replace(dst)
    return dst

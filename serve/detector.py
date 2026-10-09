"""CPU person detector for the demo service: YOLOX (Megvii, Apache-2.0) ONNX models on onnxruntime.

The official ONNX exports of YOLOX-Nano/Tiny/S (release ``0.1.1rc0``) output raw head maps
``(1, N, 85)``; this module reproduces YOLOX's ``demo/ONNXRuntime/onnx_inference.py``
pre/post-processing: letterbox to the model input with pad value 114 (BGR, 0-255, no mean/std),
grid decoding with strides 8/16/32, score = objectness × class probability, class-wise NMS.
Only COCO class 0 (person) is kept.

This is the *demo* detector (COCO weights, CPU). The benchmark numbers in ``docs/results`` use
MOTChallenge public detections; the GPU Stage A uses YOLOX-X MOT weights (``models/``).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

# name -> (input size, release URL, sha256). URLs are the official YOLOX release assets.
MODELS: dict[str, tuple[int, str, str]] = {
    "yolox_nano": (
        416,
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
        "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
    ),
    "yolox_tiny": (
        416,
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
        "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
    ),
    "yolox_s": (
        640,
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
        "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    ),
}
# OpenCV Zoo YuNet face detector (MIT), used by serve/blur.py
YUNET = (
    "face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)
STRIDES = (8, 16, 32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(2**20):
            h.update(chunk)
    return h.hexdigest()


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """``(1, 3, size, size)`` float32 input and the resize ratio (YOLOX ``preproc``)."""
    import cv2  # lazy: the event/benchmark endpoints do not need OpenCV

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(h * r), int(w * r)
    out = np.full((size, size, 3), 114, np.uint8)
    out[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(out.transpose(2, 0, 1)[None], dtype=np.float32), r


def decode(raw: np.ndarray, size: int) -> np.ndarray:
    """Grid-decode raw YOLOX outputs ``(N, 85)`` in input pixels (``demo_postprocess``)."""
    grids, strides = [], []
    for s in STRIDES:
        n = size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    g = np.concatenate(grids).astype(np.float32)
    st = np.concatenate(strides).astype(np.float32)
    out = raw.astype(np.float32).copy()
    out[:, :2] = (out[:, :2] + g) * st
    out[:, 2:4] = np.exp(out[:, 2:4]) * st
    return out


def nms(xyxy: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS; indices kept, in descending score order."""
    order = np.argsort(-scores, kind="stable")
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ov = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        order = rest[ov <= iou]
    return np.asarray(keep, np.int64)


def postprocess(
    raw: np.ndarray,
    size: int,
    ratio: float,
    img_wh: tuple[int, int],
    score_floor: float,
    nms_iou: float,
) -> tuple[FloatArray, FloatArray]:
    """Person boxes ``tlwh`` in image pixels and scores, from one raw output ``(N, 85)``."""
    d = decode(raw, size)
    score = d[:, 4] * d[:, 5]  # objectness × P(person)
    m = score >= score_floor
    d, score = d[m], score[m]
    xyxy = (
        np.stack(
            [
                d[:, 0] - d[:, 2] / 2,
                d[:, 1] - d[:, 3] / 2,
                d[:, 0] + d[:, 2] / 2,
                d[:, 1] + d[:, 3] / 2,
            ],
            1,
        )
        / ratio
    )
    w, h = img_wh
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)
    ok = (xyxy[:, 2] - xyxy[:, 0] > 1) & (xyxy[:, 3] - xyxy[:, 1] > 1)
    xyxy, score = xyxy[ok], score[ok]
    keep = nms(xyxy, score, nms_iou) if len(score) else np.zeros(0, np.int64)
    xyxy, score = xyxy[keep], score[keep]
    tlwh = np.concatenate([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], 1)
    return tlwh.astype(np.float32).reshape(-1, 4), score.astype(np.float32)


@dataclass
class YoloxDetector:
    """``detect(bgr) -> (tlwh, scores)``; persons only, every box ≥ ``score_floor``."""

    model_path: Path
    size: int
    score_floor: float = 0.1
    nms_iou: float = 0.45
    threads: int = 2

    def __post_init__(self) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, bgr: np.ndarray) -> tuple[FloatArray, FloatArray]:
        x, r = letterbox(bgr, self.size)
        raw = self.session.run(None, {self.input_name: x})[0][0]
        h, w = bgr.shape[:2]
        return postprocess(raw, self.size, r, (w, h), self.score_floor, self.nms_iou)


def model_path(name: str, model_dir: Path | None = None) -> Path:
    """Path of a known model in ``ZED_MODEL_DIR`` (download: ``python -m serve.detector fetch``)."""
    if name not in MODELS:
        raise ValueError(f"unknown detector {name!r}; known: {sorted(MODELS)}")
    root = model_dir or Path(os.environ.get("ZED_MODEL_DIR", "data/models"))
    return root / f"{name}.onnx"


def fetch_models(out: Path, names: list[str] | None = None) -> list[Path]:
    """Download the demo models into ``out`` and check their sha256 (idempotent)."""
    import urllib.request

    out.mkdir(parents=True, exist_ok=True)
    want = {n: (MODELS[n][1], MODELS[n][2]) for n in (names or list(MODELS))}
    want[YUNET[0].removesuffix(".onnx")] = (YUNET[1], YUNET[2])
    done = []
    for name, (url, digest) in want.items():
        dst = out / f"{name}.onnx"
        if not (dst.exists() and sha256(dst) == digest):
            print(f"fetch {url}")
            tmp = dst.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            if sha256(tmp) != digest:
                tmp.unlink()
                raise RuntimeError(f"{name}: sha256 mismatch")
            tmp.replace(dst)
        print(f"ok    {dst}")
        done.append(dst)
    return done


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="demo detector models")
    ap.add_argument("cmd", choices=["fetch"])
    ap.add_argument(
        "--out", type=Path, default=Path(os.environ.get("ZED_MODEL_DIR", "data/models"))
    )
    ap.add_argument("--only", nargs="*", choices=sorted(MODELS))
    a = ap.parse_args(argv)
    fetch_models(a.out, a.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

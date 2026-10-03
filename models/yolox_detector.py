"""YOLOX person detector using ByteTrack's ``yolox`` package and checkpoints.

Requires torch + torchvision and ByteTrack's source tree on ``PYTHONPATH`` (``scripts/dgx``
clones it at a pinned commit). Pre/post-processing reproduces ByteTrack's ``preproc`` and
``postprocess`` exactly (letterbox to ``test_size`` with 114 padding, BGR→RGB, ImageNet
normalisation; score = objectness × class confidence; class-wise NMS at ``nms_thresh``).

The only difference from ByteTrack's evaluation pipeline is the confidence floor: ByteTrack
keeps boxes ≥ 0.001 in the detector and the tracker discards < 0.1; we keep ≥ ``conf_floor``
(default 0.05) so the cache stays small while still leaving room to ablate τ_low below 0.1.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models.protocols import BGRImage

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

#: (depth, width) multipliers of the YOLOX variants ByteTrack ships
VARIANTS: dict[str, tuple[float, float]] = {
    "yolox_x": (1.33, 1.25),
    "yolox_l": (1.0, 1.0),
    "yolox_m": (0.67, 0.75),
    "yolox_s": (0.33, 0.50),
}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def letterbox(image: BGRImage, test_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """ByteTrack ``preproc``: returns a CHW float32 tensor-ready array and the resize ratio."""
    import cv2

    th, tw = test_size
    h, w = image.shape[:2]
    r = min(th / h, tw / w)
    nh, nw = int(h * r), int(w * r)
    padded = np.full((th, tw, 3), 114.0, dtype=np.float32)
    padded[:nh, :nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(
        np.float32
    )
    rgb = padded[:, :, ::-1] / np.float32(255.0)
    norm = (rgb - np.array(RGB_MEAN, np.float32)) / np.array(RGB_STD, np.float32)
    return np.ascontiguousarray(norm.transpose(2, 0, 1), dtype=np.float32), r


@dataclass
class YoloxConfig:
    weights: str | None
    """ByteTrack checkpoint (``*.pth.tar``); None = random init (CPU smoke tests only)."""
    variant: str = "yolox_x"
    test_size: tuple[int, int] = (800, 1440)
    conf_floor: float = 0.05
    nms_thresh: float = 0.7
    device: str = "cuda"
    fp16: bool = True
    fuse: bool = True
    trained_on: str = ""
    """Human-readable training data, stored in the cache meta (leakage audit)."""


class YoloxDetector:
    name = "yolox"

    def __init__(self, cfg: YoloxConfig) -> None:
        import torch
        from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

        self.cfg = cfg
        depth, width = VARIANTS[cfg.variant]
        in_ch = [256, 512, 1024]
        model = YOLOX(YOLOPAFPN(depth, width, in_channels=in_ch),
                      YOLOXHead(1, width, in_channels=in_ch))  # fmt: skip
        for m in model.modules():  # ByteTrack's init_yolo
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03
        self.weights_sha256: str | None = None
        if cfg.weights:
            ckpt = torch.load(cfg.weights, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt)
            # some re-releases (e.g. SparseTrack, detectron2 checkpointer) prefix the keys
            for pre in ("module.", "model."):
                if all(k.startswith(pre) for k in state):
                    state = {k[len(pre) :]: v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"checkpoint mismatch: missing={missing} unexpected={unexpected}"
                )
            self.weights_sha256 = sha256_file(cfg.weights)
        model.eval()
        if cfg.fuse:
            from yolox.utils import fuse_model

            model = fuse_model(model)
        self.device = torch.device(cfg.device)
        model = model.to(self.device)
        self.half = cfg.fp16 and self.device.type == "cuda"
        if self.half:
            model = model.half()
        self.model = model
        self._torch = torch

    def info(self) -> dict[str, object]:
        c = self.cfg
        return {
            "name": f"{c.variant} (ByteTrack)",
            "weights": Path(c.weights).name if c.weights else None,
            "weights_sha256": self.weights_sha256,
            "trained_on": c.trained_on,
            "test_size": list(c.test_size),
            "conf_floor": c.conf_floor,
            "nms_thresh": c.nms_thresh,
            "fp16": self.half,
            "score": "obj_conf * cls_conf",
        }

    def __call__(self, images: list[BGRImage]) -> list[tuple[np.ndarray, np.ndarray]]:
        from yolox.utils import postprocess

        torch = self._torch
        batch, ratios = zip(*(letterbox(im, self.cfg.test_size) for im in images), strict=True)
        x = torch.from_numpy(np.stack(batch)).to(self.device)
        if self.half:
            x = x.half()
        with torch.inference_mode():
            raw = self.model(x)
            outs = postprocess(raw.float(), 1, self.cfg.conf_floor, self.cfg.nms_thresh)
        result = []
        for im, r, o in zip(images, ratios, outs, strict=True):
            if o is None or o.shape[0] == 0:
                result.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))
                continue
            o = o.cpu().numpy()
            xyxy = o[:, :4] / r
            h, w = im.shape[:2]
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
            tlwh = np.concatenate([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], axis=1)
            score = o[:, 4] * o[:, 5]
            # only degenerate (zero-area after clipping) boxes are dropped; never by score
            ok = (tlwh[:, 2] > 0) & (tlwh[:, 3] > 0)
            order = np.argsort(-score[ok], kind="stable")
            result.append((tlwh[ok][order].astype(np.float32), score[ok][order].astype(np.float32)))
        return result

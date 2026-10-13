"""YOLOX person detector using ByteTrack's ``yolox`` package and checkpoints.

Requires torch + torchvision and ByteTrack's source tree on ``PYTHONPATH`` (``scripts/dgx``
clones it at a pinned commit). Pre/post-processing reproduces ByteTrack's ``preproc`` and
``postprocess`` exactly (letterbox to ``test_size`` with 114 padding, BGR→RGB, ImageNet
normalisation; score = objectness × class confidence; class-wise NMS at ``nms_thresh``).

Differences from ByteTrack's MOT17 ablation evaluation (``tools/track.py -f
exps/example/mot/yolox_x_ablation.py --fp16 --fuse``; audited 13 Oct 2026 at the pinned commit):

* confidence floor: ByteTrack's ``--conf`` default is 0.01; we keep ≥ ``conf_floor`` (0.05) so the
  cache stays small. The tracker discards < 0.1 (τ_low) anyway, so tracking is unaffected.
* box clipping: ByteTrack never clips boxes to the image. Before 13 Oct 2026 we clipped them
  (shrinking edge boxes, changing IoU and the w/h > 1.6 filter); ``clip_boxes`` now defaults to
  False to match. Zero-area boxes are still dropped.
* precision: ByteTrack's reported numbers use FP16. ``fp16=False`` (``--no-fp16``, ``FP16=0``)
  runs FP32 to rule out half-precision problems on a laptop GPU.
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
    clip_boxes: bool = False
    """Clip boxes to the image (ByteTrack does not; kept as an option for old caches)."""
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
            "clip_boxes": c.clip_boxes,
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
        return [raw_to_tlwh(None if o is None else o.cpu().numpy(), r, im.shape[:2],
                            self.cfg.clip_boxes)
                for im, r, o in zip(images, ratios, outs, strict=True)]  # fmt: skip


def raw_to_tlwh(
    o: np.ndarray | None, ratio: float, hw: tuple[int, int], clip: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """``postprocess`` rows ``(x1, y1, x2, y2, obj, cls, cls_id)`` in letterbox pixels → image
    ``tlwh`` + score ``obj × cls``, sorted by score. Like ByteTrack: divide by the resize ratio and
    do not clip unless ``clip``; only zero/negative-area boxes are dropped, never by score."""
    if o is None or o.shape[0] == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    xyxy = o[:, :4].astype(np.float64) / ratio
    if clip:
        h, w = hw
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
    tlwh = np.concatenate([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], axis=1)
    score = o[:, 4] * o[:, 5]
    ok = (tlwh[:, 2] > 0) & (tlwh[:, 3] > 0)
    order = np.argsort(-score[ok], kind="stable")
    return tlwh[ok][order].astype(np.float32), score[ok][order].astype(np.float32)

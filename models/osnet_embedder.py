"""OSNet re-ID embedder (torchreid, MIT). Default: ``osnet_ain_x1_0`` trained on MSMT17.

MSMT17 has no MOT17/MOT20/DanceTrack/SOMPT22 footage, so the embedder cannot leak evaluation
identities. Crops are taken from the original frame (clipped to the image), resized to
256×128 (H×W), RGB, ImageNet-normalised, embedded in batches, then L2-normalised and stored as
float16 (512-d).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from models.protocols import BGRImage
from models.yolox_detector import RGB_MEAN, RGB_STD, sha256_file


@dataclass
class OsnetConfig:
    weights: str | None
    """torchreid checkpoint; None = random init (CPU smoke tests only)."""
    arch: str = "osnet_ain_x1_0"
    input_hw: tuple[int, int] = (256, 128)
    batch_size: int = 256
    device: str = "cuda"
    fp16: bool = True
    trained_on: str = "MSMT17 (combineall)"


def crop_batch(image: BGRImage, tlwh: np.ndarray, input_hw: tuple[int, int]) -> np.ndarray:
    """Return ``(N, 3, H, W)`` float32 normalised RGB crops. Boxes are clipped to the image."""
    import cv2

    h_img, w_img = image.shape[:2]
    out = np.empty((len(tlwh), 3, *input_hw), np.float32)
    mean, std = np.array(RGB_MEAN, np.float32), np.array(RGB_STD, np.float32)
    for i, (x, y, w, h) in enumerate(np.asarray(tlwh, np.float64)):
        x1 = int(np.clip(np.floor(x), 0, w_img - 1))
        y1 = int(np.clip(np.floor(y), 0, h_img - 1))
        x2 = int(np.clip(np.ceil(x + w), x1 + 1, w_img))
        y2 = int(np.clip(np.ceil(y + h), y1 + 1, h_img))
        crop = cv2.resize(image[y1:y2, x1:x2], (input_hw[1], input_hw[0]),
                          interpolation=cv2.INTER_LINEAR)  # fmt: skip
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        out[i] = ((rgb - mean) / std).transpose(2, 0, 1)
    return out


class OsnetEmbedder:
    name = "osnet"
    dim = 512

    def __init__(self, cfg: OsnetConfig) -> None:
        import torch
        import torchreid

        self.cfg = cfg
        model = torchreid.models.build_model(cfg.arch, num_classes=1000, pretrained=False)
        self.weights_sha256: str | None = None
        if cfg.weights:
            from torchreid.utils import load_pretrained_weights

            load_pretrained_weights(model, cfg.weights)  # drops the classifier, logs mismatches
            self.weights_sha256 = sha256_file(cfg.weights)
        model.eval()
        self.device = torch.device(cfg.device)
        self.half = cfg.fp16 and self.device.type == "cuda"
        model = model.to(self.device)
        if self.half:
            model = model.half()
        self.model = model
        self._torch = torch

    def info(self) -> dict[str, object]:
        c = self.cfg
        return {
            "name": c.arch,
            "weights": Path(c.weights).name if c.weights else None,
            "weights_sha256": self.weights_sha256,
            "trained_on": c.trained_on,
            "input_hw": list(c.input_hw),
            "dim": self.dim,
            "normalised": "l2",
        }

    def __call__(self, image: BGRImage, tlwh: np.ndarray) -> npt.NDArray[np.float16]:
        torch = self._torch
        n = len(tlwh)
        if n == 0:
            return np.zeros((0, self.dim), np.float16)
        feats = []
        for s in range(0, n, self.cfg.batch_size):
            x = torch.from_numpy(crop_batch(image, tlwh[s : s + self.cfg.batch_size],
                                            self.cfg.input_hw)).to(self.device)  # fmt: skip
            if self.half:
                x = x.half()
            with torch.inference_mode():
                f = self.model(x).float()
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        emb = np.concatenate(feats).astype(np.float32)
        if emb.shape[1] != self.dim:
            raise RuntimeError(f"expected {self.dim}-d features, got {emb.shape[1]}")
        # renormalise after the float16 cast so the stored vectors are unit length
        emb16 = emb.astype(np.float16)
        norms = np.linalg.norm(emb16.astype(np.float32), axis=1, keepdims=True)
        out: npt.NDArray[np.float16] = (emb16.astype(np.float32) / np.maximum(norms, 1e-12)).astype(
            np.float16
        )
        return out

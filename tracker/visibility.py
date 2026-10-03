"""Per-detection visibility estimation at inference time, from boxes only.

* :class:`ConstantVisibility`: every box fully visible. It reproduces the baselines exactly.
* :class:`DepthOrderVisibility`: the fraction of each box that is not covered by a box that is
  *closer to the camera*, and not outside the image. "Closer" uses the bottom edge: the larger
  ``y2`` is in front. That is MOTChallenge's own rule for generating ``gt.txt`` visibility ("the
  object with the lowest y-value of the bounding box is closest to the camera", IJCV App. C.1.5),
  and SparseTrack's pseudo-depth ``H − y2`` (arXiv:2306.05238 §3.2). Implementation brief §2.

The covered fraction is computed on a ``grid × grid`` lattice of cell centres per box, which is
separable, so all pairs are vectorised:
``hidden[i, a, b] = OR_j (x-inside[i, j, a] ∧ y-inside[i, j, b] ∧ closer[i, j])``.
On random boxes, ``grid = 32`` is within 0.01 MAE of pixel rasterisation (asserted in
``tests/test_gallery_visibility.py``); the all-classes gt oracle built on it (grid 16) reaches
MAE 0.023 against MOT17 gt visibility (``docs/results/visibility``).

``score_fusion = "product"`` multiplies the geometric estimate by the clipped detector score. A
weak detection is very often a partly hidden person, so the score carries occlusion evidence that
geometry misses (static occluders, people the detector did not find). The fusion was chosen on
MOT17 train-half (Spearman 0.495 → 0.560, AP for v<0.5 0.165 → 0.246) before it was frozen.

Only detections with ``score ≥ occluder_min_score`` act as occluders: a false positive should not
"hide" a real person. Static occluders (cars, pillars) are invisible to a person detector, which
sets a floor on the error. ``eval/visibility_eval.py`` reports it as the pedestrians-only gt-box
oracle.

Coordinates: MOT files are 1-based, so the image spans ``[origin, origin + W)`` with
``origin = 1`` (pitfall §6.18).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from tracker.types import FloatArray


class VisibilityEstimator(Protocol):
    def __call__(self, xyxy: FloatArray, scores: FloatArray | None = None) -> FloatArray:
        """Return an ``(N,)`` visibility in ``[0, 1]`` for ``(N, 4)`` boxes of one frame."""
        ...


class ConstantVisibility:
    def __init__(self, value: float = 1.0) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("visibility must be in [0, 1]")
        self.value = value

    def __call__(self, xyxy: FloatArray, scores: FloatArray | None = None) -> FloatArray:
        return np.full(np.asarray(xyxy).reshape(-1, 4).shape[0], self.value, dtype=np.float64)


class DepthOrderVisibility:
    def __init__(
        self,
        image_size: tuple[int, int] | None = None,
        *,
        occluder_min_score: float = 0.0,
        grid: int = 16,
        border: bool = True,
        origin: float = 1.0,
        score_fusion: str = "none",
    ) -> None:
        """``image_size = (width, height)``; ``None`` disables the image-border term."""
        if grid < 1:
            raise ValueError("grid must be >= 1")
        self.image_size = image_size
        self.occluder_min_score = occluder_min_score
        self.grid = grid
        self.border = border and image_size is not None
        self.origin = origin
        if score_fusion not in ("none", "product"):
            raise ValueError(f"unknown score_fusion {score_fusion!r}")
        self.score_fusion = score_fusion

    def __call__(
        self,
        xyxy: FloatArray,
        scores: FloatArray | None = None,
        occluder_mask: np.ndarray | None = None,
    ) -> FloatArray:
        """Visibility of every box. ``occluder_mask`` (if given) overrides the score rule."""
        b = np.asarray(xyxy, np.float64).reshape(-1, 4)
        n = b.shape[0]
        if n == 0:
            return np.zeros(0)
        g = self.grid
        frac = (np.arange(g) + 0.5) / g
        w = np.maximum(b[:, 2] - b[:, 0], 0.0)
        h = np.maximum(b[:, 3] - b[:, 1], 0.0)
        xs = b[:, 0:1] + frac[None, :] * w[:, None]  # (N, G)
        ys = b[:, 1:2] + frac[None, :] * h[:, None]
        if occluder_mask is None:
            occ = (
                np.ones(n, bool)
                if scores is None
                else np.asarray(scores).reshape(-1) >= self.occluder_min_score
            )
        else:
            occ = np.asarray(occluder_mask, bool).reshape(-1)
        hidden = np.zeros((n, g, g), bool)
        if occ.any() and n > 1:
            o = b[occ]
            closer = o[None, :, 3] > b[:, None, 3]  # (N, M): occluder bottom edge lower
            idx = np.nonzero(occ)[0]
            closer[idx, np.arange(len(idx))] = False  # a box never occludes itself
            inx = (xs[:, None, :] >= o[None, :, 0, None]) & (xs[:, None, :] < o[None, :, 2, None])
            iny = (ys[:, None, :] >= o[None, :, 1, None]) & (ys[:, None, :] < o[None, :, 3, None])
            inx &= closer[:, :, None]
            cnt = np.einsum("ija,ijb->iab", inx.astype(np.float32), iny.astype(np.float32))
            hidden = cnt > 0.5
        if self.border and self.image_size is not None:
            W, H = self.image_size
            ox = (xs < self.origin) | (xs >= self.origin + W)
            oy = (ys < self.origin) | (ys >= self.origin + H)
            hidden |= ox[:, :, None] | oy[:, None, :]
        # hidden[i, a, b]: a indexes x, b indexes y
        vis = 1.0 - hidden.reshape(n, -1).mean(axis=1)
        vis[(w <= 0) | (h <= 0)] = 0.0
        if self.score_fusion == "product" and scores is not None:
            vis = vis * np.clip(np.asarray(scores, np.float64).reshape(-1), 0.0, 1.0)
        return np.asarray(vis, np.float64)


def build_visibility(cfg: Any, meta: Any | None = None) -> VisibilityEstimator:
    """Build from a :class:`tracker.config.VisibilityConfig` (or a bare estimator name).

    ``meta`` (a :class:`tracker.io.CacheMeta`) supplies the image size for the border term.
    """
    name = cfg if isinstance(cfg, str) else cfg.estimator
    if name == "constant":
        return ConstantVisibility()
    if name == "depth_order":
        size = None
        w = int(getattr(meta, "im_width", 0) or 0)
        h = int(getattr(meta, "im_height", 0) or 0)
        if w > 0 and h > 0:
            size = (w, h)
        if isinstance(cfg, str):
            return DepthOrderVisibility(size)
        return DepthOrderVisibility(
            size,
            occluder_min_score=cfg.occluder_min_score,
            grid=cfg.grid,
            border=cfg.border,
            score_fusion=cfg.score_fusion,
        )
    raise ValueError(f"unknown visibility estimator {name!r}")

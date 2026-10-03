"""Cost matrices. Convention: rows = tracks, columns = detections, lower = better, in ``[0, 1]``.

A cost of exactly ``1.0`` means "no evidence for this pair" and, with the thresholds used by the
association stages (all < 1), can never be matched.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tracker.types import FloatArray


def iou_matrix(a_xyxy: FloatArray, b_xyxy: FloatArray, plus_one: bool = False) -> FloatArray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` ``xyxy`` boxes; returns ``(N, M)``.

    Degenerate (zero-area) boxes have IoU 0 with everything. ``plus_one`` uses the pixel-inclusive
    convention of ``cython_bbox.bbox_overlaps`` (``w = x2 − x1 + 1``), which ByteTrack and
    BoT-SORT use; it is applied to the intersection and both areas exactly as there.
    """
    a = np.asarray(a_xyxy, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b_xyxy, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    if plus_one:
        a = a.copy()
        b = b.copy()
        a[:, 2:] += 1.0
        b[:, 2:] += 1.0
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, inter / union, 0.0)
    return np.asarray(iou, dtype=np.float64)


def iou_cost(tracks_xyxy: FloatArray, dets_xyxy: FloatArray, plus_one: bool = False) -> FloatArray:
    """``1 − IoU``."""
    return 1.0 - iou_matrix(tracks_xyxy, dets_xyxy, plus_one)


def fuse_score(cost: FloatArray, det_scores: FloatArray) -> FloatArray:
    """ByteTrack's score fusion: ``1 − (1 − cost) · score``.

    Makes a confident detection preferable to a weak one with the same overlap.
    """
    if cost.size == 0:
        return cost.copy()
    sim = (1.0 - cost) * np.asarray(det_scores, dtype=np.float64)[None, :]
    return 1.0 - sim


def cosine_cost(
    track_emb: npt.NDArray[np.floating], det_emb: npt.NDArray[np.floating]
) -> FloatArray:
    """Cosine distance ``1 − cos`` between ``(N, D)`` and ``(M, D)`` embeddings, clipped to [0, 2].

    Inputs need not be normalised; zero vectors get distance 1 (no evidence).
    """
    t = np.asarray(track_emb, dtype=np.float64)
    d = np.asarray(det_emb, dtype=np.float64)
    if t.shape[0] == 0 or d.shape[0] == 0:
        return np.ones((t.shape[0], d.shape[0]))
    if t.shape[1] != d.shape[1]:
        raise ValueError(f"embedding dims differ: {t.shape[1]} vs {d.shape[1]}")
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    dn = np.linalg.norm(d, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = (t / np.where(tn > 0, tn, 1.0)) @ (d / np.where(dn > 0, dn, 1.0)).T
    cos[(tn[:, 0] == 0)[:, None] | (dn[:, 0] == 0)[None, :]] = 0.0
    return np.clip(1.0 - cos, 0.0, 2.0)


def fuse_iou_appearance(
    iou_c: FloatArray,
    appearance_c: FloatArray,
    *,
    proximity_thresh: float,
    appearance_thresh: float,
    motion_cost: FloatArray | None = None,
) -> FloatArray:
    """BoT-SORT's IoU–re-ID fusion: ``min(c_iou, ĉ_app)``.

    ``ĉ_app = c_app / 2`` (cosine distance rescaled to [0, 1]); it is set to 1 (no evidence) when
    it exceeds ``appearance_thresh`` or when the boxes are too far apart
    (``c_iou > proximity_thresh``), so appearance can rescue a weak-IoU match but never create a
    match between distant boxes. ``motion_cost`` (default ``iou_c``) is the cost the appearance
    term is min-fused with, e.g. the score-fused IoU cost; the proximity gate always uses raw IoU.
    """
    if iou_c.shape != appearance_c.shape:
        raise ValueError(f"shape mismatch {iou_c.shape} vs {appearance_c.shape}")
    app = appearance_c / 2.0
    app = np.where(app > appearance_thresh, 1.0, app)
    app = np.where(iou_c > proximity_thresh, 1.0, app)
    return np.minimum(iou_c if motion_cost is None else motion_cost, app)


def gate_cost(cost: FloatArray, mask: npt.NDArray[np.bool_], value: float = np.inf) -> FloatArray:
    """Return a copy of ``cost`` with ``mask`` entries set to ``value`` (forbidden pairs)."""
    out = np.array(cost, dtype=np.float64, copy=True)
    out[mask] = value
    return out

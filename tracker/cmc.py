"""Camera-motion compensation (CMC): a per-frame 2×3 affine from frame ``t-1`` to frame ``t``.

Ported from BoT-SORT ``tracker/gmc.py`` (``GMC.applyFile`` and ``GMC.applySparseOptFlow``,
commit 2519854, MIT, Copyright (c) 2022 Nir Aharon).

* :class:`FileCMC` reads BoT-SORT's pre-computed affines (``third_party/cmc_files``). It runs on
  CPU with no video, so it is what we use in phase 2. Unlike BoT-SORT's sequential
  ``readline()``, it indexes by frame number (pitfall §6.10 of the implementation brief).
* :class:`SparseOptFlowCMC` is the image-based method, kept for when frames are available. It
  needs OpenCV (an optional dependency, imported lazily) and is **not exercised** in phase 2:
  with no frames there is nothing to compute it on, so ``method: auto`` falls back to "off".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CMC_DIR = ROOT / "third_party" / "cmc_files"


class CMC(Protocol):
    """``affine(frame_id)`` → 2×3 affine (frame ``t-1`` → ``t``), or ``None`` for "no CMC"."""

    def affine(self, frame_id: int) -> FloatArray | None: ...


class NoCMC:
    def affine(self, frame_id: int) -> FloatArray | None:
        return None


def parse_cmc_file(path: Path) -> FloatArray:
    """Return row-ordered affines while validating contiguous explicit frame indices.

    Files may use zero-based indices or absolute MOT frame numbers. FileCMC maps renumbered
    cache frames to rows; the first file index is not a cache-frame offset.
    """
    rows = []
    first_index: int | None = None
    for k, line in enumerate(path.read_text().splitlines()):
        tok = line.split()
        if not tok:
            raise ValueError(f"{path}: line {k} is empty")
        if len(tok) != 7:
            raise ValueError(f"{path}: line {k} has {len(tok)} fields, expected 7")
        try:
            raw_index = float(tok[0])
            index = int(raw_index)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{path}: line {k} has invalid index {tok[0]!r}") from exc
        if not np.isfinite(raw_index) or raw_index != index or index < 0:
            raise ValueError(f"{path}: line {k} has invalid frame index {tok[0]!r}")
        if first_index is None:
            first_index = index
        if index != first_index + len(rows):
            raise ValueError(f"{path}: line {k} has index {tok[0]}")
        try:
            affine = [float(v) for v in tok[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}: line {k} has a nonnumeric affine value") from exc
        if not np.isfinite(affine).all():
            raise ValueError(f"{path}: line {k} has a non-finite affine value")
        rows.append(affine)
    return np.asarray(rows, np.float64).reshape(-1, 2, 3)


class FileCMC:
    """Pre-computed affines; ``frame_id`` is 1-based and maps to line ``frame_id - 1 + offset``."""

    def __init__(self, path: Path, offset: int = 0, expected_frames: int | None = None) -> None:
        self.path = Path(path)
        self.offset = offset
        self.table = parse_cmc_file(self.path)
        if expected_frames is not None and self.table.shape[0] < offset + expected_frames:
            raise ValueError(
                f"{self.path}: {self.table.shape[0]} lines < offset {offset} + {expected_frames}"
            )

    def affine(self, frame_id: int) -> FloatArray | None:
        i = frame_id - 1 + self.offset
        if not 0 <= i < self.table.shape[0]:
            raise IndexError(f"{self.path.name}: no affine for frame {frame_id} (line {i})")
        return np.asarray(self.table[i], np.float64).copy()


def _seq_key(sequence: str) -> str:
    """``MOT17-02-FRCNN`` → ``MOT17-02``; ``dancetrack0004`` → ``dancetrack-0004``."""
    s = sequence
    for suffix in ("-FRCNN", "-SDP", "-DPM"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    if s.startswith("dancetrack") and "-" not in s:
        s = "dancetrack-" + s[len("dancetrack") :]
    return s


def find_cmc_file(
    sequence: str, split: str, frame_offset: int = 0, root: Path = DEFAULT_CMC_DIR
) -> tuple[Path, int] | None:
    """Return ``(file, line offset)`` for a sequence/split, or ``None`` if we have no affines.

    The val half uses the ``*_ablation`` files (line 0 = first frame of the half), as BoT-SORT's
    and Deep OC-SORT's ablations do; anything else indexes the full-sequence ``MOTChallenge``
    file at the split's frame offset.
    """
    key = _seq_key(sequence)
    cands: list[tuple[Path, int]] = []
    if split == "val_half":
        ds = key.split("-")[0]
        cands.append((root / f"{ds}_ablation" / f"GMC-{key}.txt", 0))
    cands.append((root / "MOTChallenge" / f"GMC-{key}.txt", frame_offset))
    cands.append((root / "DanceTrack" / f"GMC-{key}.txt", frame_offset))
    for path, off in cands:
        if path.exists():
            return path, off
    return None


def build_cmc(method: str, meta: Any | None, root: Path = DEFAULT_CMC_DIR) -> tuple[CMC, str]:
    """Build the CMC provider for ``method`` (``none`` | ``file`` | ``auto``) and a description.

    ``meta`` is a :class:`tracker.io.CacheMeta` (sequence, split, ``extra.frame_offset``).
    ``file`` raises if no affine file matches; ``auto`` silently falls back to no CMC.
    """
    if method == "none" or meta is None:
        return NoCMC(), "off"
    if method not in ("file", "auto"):
        raise ValueError(f"unknown cmc method {method!r} (sparse optical flow needs frames)")
    off = int(getattr(meta, "extra", {}).get("frame_offset", 0))
    hit = find_cmc_file(meta.sequence, meta.split, off, root)
    if hit is None:
        if method == "file":
            raise FileNotFoundError(f"no CMC file for {meta.sequence} ({meta.split})")
        return NoCMC(), "off (no affine file for this sequence)"
    path, line_off = hit
    return FileCMC(path, line_off, meta.seq_length), f"file:{path.relative_to(root)}@{line_off}"


class SparseOptFlowCMC:  # pragma: no cover - needs OpenCV and frames
    """Simplified port of BoT-SORT ``GMC.applySparseOptFlow``; not exercised or verified."""

    def __init__(self, downscale: int = 2) -> None:
        import cv2  # optional dependency

        self.cv2 = cv2
        self.downscale = max(1, downscale)
        self.prev: Any = None
        self.prev_kp: Any = None

    def apply(self, frame_bgr: npt.NDArray[np.uint8]) -> FloatArray:
        cv2 = self.cv2
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)
        if self.downscale > 1:
            gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))
        kp = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=1000, qualityLevel=0.01,
                                     minDistance=1, blockSize=3, useHarrisDetector=False,
                                     k=0.04)  # fmt: skip
        if self.prev is None:
            self.prev, self.prev_kp = gray.copy(), kp
            return H
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.prev_kp, None)
        ok = status.reshape(-1).astype(bool)
        p0, p1 = self.prev_kp[ok], nxt[ok]
        if p0.shape[0] > 4:
            H, _ = cv2.estimateAffinePartial2D(p0, p1, cv2.RANSAC)
            if self.downscale > 1:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        self.prev, self.prev_kp = gray.copy(), kp
        return np.asarray(H, np.float64)

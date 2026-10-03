"""I/O: the Stage A detection cache, MOTChallenge files and per-frame JSON export.

Detection cache, schema v1 (one ``.npz`` per sequence, written by ``models/stage_a.py``)
--------------------------------------------------------------------------------------
=========  ===========  ========  ==================================================
key        dtype        shape     meaning
=========  ===========  ========  ==================================================
frame      int32        (N,)      1-based frame index, non-decreasing
tlwh       float32      (N, 4)    box in original-image pixels (top-left x, y, w, h)
score      float32      (N,)      detector confidence (obj × cls for YOLOX)
emb        float16      (N, D)    optional; L2-normalised re-ID embedding per box
meta       str (0-d)    ()        JSON, see :class:`CacheMeta`
=========  ===========  ========  ==================================================

Every detection the detector emits above its (low) confidence floor is stored; nothing is
filtered by score on the GPU side, so trackers choose their own thresholds on CPU.

Tracker outputs
---------------
* MOTChallenge text (``frame,id,x,y,w,h,score,-1,-1,-1``), the input of TrackEval and of the
  planned ``eval/gate_events.py``.
* JSON Lines, one object per frame (``FRAME_JSON_SCHEMA``), the input of the planned demo viewer.
  ``pending`` stays empty until delayed assignment exists.
"""

from __future__ import annotations

import configparser
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tracker.types import Detections, FrameResult

CACHE_SCHEMA_VERSION = 1
FRAME_JSON_SCHEMA = "zed-ps2-frames/v1"


class CacheFormatError(ValueError):
    pass


@dataclass(frozen=True)
class CacheMeta:
    sequence: str
    dataset: str
    split: str
    frame_rate: float
    seq_length: int
    im_width: int
    im_height: int
    detector: dict[str, Any] = field(default_factory=dict)
    """name, exp, weights, weights_sha256, test_size, conf_floor, nms_thresh, ..."""
    reid: dict[str, Any] | None = None
    """name, weights, weights_sha256, input_hw, dim; None if no embeddings."""
    detector_saw_eval_half: bool = False
    """True when the detector's training data includes the frames this cache is scored on."""
    schema_version: int = CACHE_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)
    """Provenance: git commit, host, GPU, timestamps, runtimes."""


@dataclass(frozen=True)
class SequenceCache:
    meta: CacheMeta
    frame: npt.NDArray[np.int32]
    tlwh: npt.NDArray[np.float32]
    score: npt.NDArray[np.float32]
    emb: npt.NDArray[np.float16] | None = None

    def __len__(self) -> int:
        return int(self.frame.shape[0])

    @property
    def embedding_dim(self) -> int | None:
        return None if self.emb is None else int(self.emb.shape[1])

    def frames(self) -> Iterator[tuple[int, Detections]]:
        """Yield ``(frame_id, Detections)`` for every frame ``1..seq_length``, empty ones too."""
        n = self.meta.seq_length
        bounds = np.searchsorted(self.frame, np.arange(1, n + 2))
        for f in range(1, n + 1):
            lo, hi = int(bounds[f - 1]), int(bounds[f])
            emb = None if self.emb is None else self.emb[lo:hi].astype(np.float32)
            yield f, Detections(self.tlwh[lo:hi].astype(np.float64),
                                self.score[lo:hi].astype(np.float64), emb)  # fmt: skip


def validate(cache: SequenceCache) -> None:
    """Raise :class:`CacheFormatError` if the cache violates schema v1."""
    m = cache.meta
    if m.schema_version != CACHE_SCHEMA_VERSION:
        raise CacheFormatError(f"schema_version {m.schema_version} != {CACHE_SCHEMA_VERSION}")
    n = len(cache)
    if cache.frame.dtype != np.int32 or cache.frame.shape != (n,):
        raise CacheFormatError(
            f"frame must be int32 (N,), got {cache.frame.dtype} {cache.frame.shape}"
        )
    if cache.tlwh.dtype != np.float32 or cache.tlwh.shape != (n, 4):
        raise CacheFormatError(
            f"tlwh must be float32 (N, 4), got {cache.tlwh.dtype} {cache.tlwh.shape}"
        )
    if cache.score.dtype != np.float32 or cache.score.shape != (n,):
        raise CacheFormatError(f"score must be float32 (N,), got {cache.score.dtype}")
    if n and (np.any(np.diff(cache.frame) < 0)):
        raise CacheFormatError("frame must be sorted (non-decreasing)")
    if n and (cache.frame.min() < 1 or cache.frame.max() > m.seq_length):
        raise CacheFormatError(f"frame ids outside 1..{m.seq_length}")
    if not np.all(np.isfinite(cache.tlwh)) or not np.all(np.isfinite(cache.score)):
        raise CacheFormatError("non-finite boxes or scores")
    if n and np.any(cache.tlwh[:, 2:] <= 0):
        raise CacheFormatError("non-positive box width/height")
    if cache.emb is not None:
        if cache.emb.dtype != np.float16 or cache.emb.ndim != 2 or cache.emb.shape[0] != n:
            raise CacheFormatError(
                f"emb must be float16 (N, D), got {cache.emb.dtype} {cache.emb.shape}"
            )
        if n:
            norms = np.linalg.norm(cache.emb.astype(np.float32), axis=1)
            if np.any(np.abs(norms - 1.0) > 1e-2):
                raise CacheFormatError("embeddings must be L2-normalised")
        if m.reid is None:
            raise CacheFormatError("emb present but meta.reid is null")


def save_cache(path: str | Path, cache: SequenceCache) -> Path:
    """Validate and write a cache atomically (``.tmp`` + rename, so partial files never exist)."""
    validate(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "frame": cache.frame,
        "tlwh": cache.tlwh,
        "score": cache.score,
        "meta": np.array(json.dumps(asdict(cache.meta), sort_keys=True)),
    }
    if cache.emb is not None:
        arrays["emb"] = cache.emb
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp.replace(path)
    return path


def load_cache(path: str | Path) -> SequenceCache:
    with np.load(Path(path), allow_pickle=False) as z:
        missing = {"frame", "tlwh", "score", "meta"} - set(z.files)
        if missing:
            raise CacheFormatError(f"{path}: missing keys {sorted(missing)}")
        meta_dict = json.loads(str(z["meta"][()]))
        known = set(CacheMeta.__dataclass_fields__)
        unknown = set(meta_dict) - known
        if unknown:
            raise CacheFormatError(f"{path}: unknown meta keys {sorted(unknown)}")
        cache = SequenceCache(
            meta=CacheMeta(**meta_dict),
            frame=z["frame"],
            tlwh=z["tlwh"],
            score=z["score"],
            emb=z["emb"] if "emb" in z.files else None,
        )
    validate(cache)
    return cache


# ---------------------------------------------------------------------- MOTChallenge files
@dataclass(frozen=True)
class SeqInfo:
    name: str
    frame_rate: float
    seq_length: int
    im_width: int
    im_height: int


def read_seqinfo(seq_dir: str | Path) -> SeqInfo:
    cp = configparser.ConfigParser()
    cp.read(Path(seq_dir) / "seqinfo.ini")
    s = cp["Sequence"]
    return SeqInfo(
        name=s["name"],
        frame_rate=float(s["frameRate"]),
        seq_length=int(s["seqLength"]),
        im_width=int(s["imWidth"]),
        im_height=int(s["imHeight"]),
    )


def load_mot_det_txt(
    seq_dir: str | Path, dataset: str = "MOT17", split: str = "train"
) -> SequenceCache:
    """Load MOTChallenge public detections (``det/det.txt``) as an embedding-free cache."""
    seq_dir = Path(seq_dir)
    info = read_seqinfo(seq_dir)
    raw = np.loadtxt(seq_dir / "det" / "det.txt", delimiter=",", ndmin=2)
    if raw.size == 0:
        raw = np.zeros((0, 7))
    order = np.argsort(raw[:, 0], kind="stable")
    raw = raw[order]
    keep = (raw[:, 4] > 0) & (raw[:, 5] > 0)
    raw = raw[keep]
    meta = CacheMeta(
        sequence=info.name,
        dataset=dataset,
        split=split,
        frame_rate=info.frame_rate,
        seq_length=info.seq_length,
        im_width=info.im_width,
        im_height=info.im_height,
        detector={"name": "public", "source": "det/det.txt"},
    )
    return SequenceCache(
        meta=meta,
        frame=raw[:, 0].astype(np.int32),
        tlwh=raw[:, 2:6].astype(np.float32),
        score=raw[:, 6].astype(np.float32),
    )


def format_mot_rows(results: Iterable[FrameResult]) -> list[str]:
    rows = []
    for r in results:
        for t in r.tracks:
            x, y, w, h = t.tlwh
            rows.append(
                f"{r.frame_id},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{t.score:.4f},-1,-1,-1"
            )
    return rows


def write_mot_results(path: str | Path, results: Iterable[FrameResult]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = format_mot_rows(results)
    path.write_text("\n".join(rows) + ("\n" if rows else ""))
    return path


def frame_to_json(r: FrameResult) -> dict[str, Any]:
    return {
        "frame": r.frame_id,
        "tracks": [
            {
                "id": t.track_id,
                "tlwh": [round(v, 2) for v in t.tlwh],
                "score": round(t.score, 4),
                "state": t.state,
                "visibility": round(t.visibility, 3),
            }
            for t in r.tracks
        ],
        "pending": [
            {
                "track_id": p.track_id,
                "candidates_tlwh": [list(b) for b in p.candidate_boxes_tlwh],
                "costs": list(p.candidate_costs),
                "opened_frame": p.opened_frame,
                "deadline_frame": p.deadline_frame,
            }
            for p in r.pending
        ],
    }


def write_frames_jsonl(
    path: str | Path, results: Iterable[FrameResult], header: dict[str, Any] | None = None
) -> Path:
    """Write per-frame JSON Lines. First line is a header: ``{"schema": ..., **header}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(json.dumps({"schema": FRAME_JSON_SCHEMA, **(header or {})}) + "\n")
        for r in results:
            fh.write(json.dumps(frame_to_json(r), separators=(",", ":")) + "\n")
    return path

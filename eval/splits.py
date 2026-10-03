"""Half splits used by every MOT17/MOT20 paper we compare against (ByteTrack, OC-SORT, BoT-SORT,
SparseTrack, Hybrid-SORT all share ``tools/convert_mot17_to_coco.py``).

For a sequence with ``N`` frames (0-indexed image ``i`` is frame ``i + 1``):

* ``train_half``: frames ``1 .. N//2 + 1``
* ``val_half``:   frames ``N//2 + 2 .. N``, renumbered to start at 1

Ground truth and detections are cropped and renumbered the same way, so TrackEval sees a short
sequence whose ``seqLength`` is the number of val frames. This reproduces
``/workspace/zed-ps2/scripts/make_half_split.py`` (Research Lab), whose output is the reference in
``tests/test_eval_splits.py``. MOT17 uses the 7 ``*-FRCNN`` train sequences (all three detector
copies share the same images and gt).
"""

from __future__ import annotations

import configparser
import dataclasses
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from tracker.io import SequenceCache

SPLITS = ("full", "train_half", "val_half")
"""``full`` = the whole sequence (DanceTrack val, SOMPT22, or the full MOT train set)."""


def half_range(n: int, split: str) -> tuple[int, int]:
    """Inclusive 1-based frame range of ``split`` for a sequence of ``n`` frames."""
    if split == "train_half":
        return 1, n // 2 + 1
    if split == "val_half":
        return n // 2 + 2, n
    if split in ("full", "train", "val"):
        return 1, n
    raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")


def crop_rows(rows: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Keep MOT rows with frame in ``[lo, hi]`` and renumber frames to start at 1."""
    rows = np.asarray(rows)
    if rows.size == 0:
        return rows.reshape(0, rows.shape[1] if rows.ndim == 2 else 9)
    keep = (rows[:, 0] >= lo) & (rows[:, 0] <= hi)
    out = rows[keep].copy()
    out[:, 0] -= lo - 1
    return out


def crop_cache(cache: SequenceCache, split: str) -> SequenceCache:
    """Restrict a (full-sequence) cache to ``split``, renumbering frames from 1."""
    if split in ("full", "train", "val"):
        return cache
    if cache.meta.split not in ("train", ""):
        raise ValueError(f"{cache.meta.sequence}: cache split is {cache.meta.split!r}, not 'train'")
    lo, hi = half_range(cache.meta.seq_length, split)
    keep = (cache.frame >= lo) & (cache.frame <= hi)
    extra = {**cache.meta.extra, "frame_offset": lo - 1, "source_split": cache.meta.split}
    meta = dataclasses.replace(cache.meta, split=split, seq_length=hi - lo + 1, extra=extra)
    return SequenceCache(
        meta=meta,
        frame=(cache.frame[keep] - (lo - 1)).astype(np.int32),
        tlwh=cache.tlwh[keep],
        score=cache.score[keep],
        emb=None if cache.emb is None else cache.emb[keep],
    )


def _fmt_row(r: np.ndarray) -> str:
    head = ",".join(f"{v:g}" for v in r[:8])
    return f"{head},{r[8]:.6f}" if r.shape[0] > 8 else head


def write_gt_split(seq_dir: Path, out_root: Path, split: str) -> int:
    """Write ``<out_root>/<seq>/gt/gt.txt`` + ``seqinfo.ini`` for ``split``; return its length."""
    cp = configparser.ConfigParser()
    cp.optionxform = str  # type: ignore[assignment,method-assign]
    cp.read(seq_dir / "seqinfo.ini")
    n = int(cp["Sequence"]["seqLength"])
    lo, hi = half_range(n, split)
    gt = np.loadtxt(seq_dir / "gt" / "gt.txt", delimiter=",", ndmin=2)
    rows = crop_rows(gt, lo, hi)
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))] if rows.size else rows
    dst = out_root / seq_dir.name
    (dst / "gt").mkdir(parents=True, exist_ok=True)
    (dst / "gt" / "gt.txt").write_text("".join(_fmt_row(r) + "\n" for r in rows))
    cp["Sequence"]["seqLength"] = str(hi - lo + 1)
    with (dst / "seqinfo.ini").open("w") as fh:
        cp.write(fh, space_around_delimiters=False)
    return hi - lo + 1


def write_seqmap(path: Path, names: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("name\n" + "".join(f"{n}\n" for n in names))


def find_sequences(root: Path, dataset: str) -> list[Path]:
    """Sequences with gt: ``<root>/train/<seq>`` (``<root>/val`` for DanceTrack)."""
    sub = "val" if dataset.lower() == "dancetrack" else "train"
    train = root / sub if (root / sub).is_dir() else root
    pattern = "*-FRCNN" if dataset.upper() == "MOT17" else "*"
    seqs = sorted(d for d in train.glob(pattern) if (d / "gt" / "gt.txt").exists())
    if not seqs:
        raise FileNotFoundError(f"no {dataset} sequences with gt under {train}")
    return seqs


def prepare_gt(root: Path, dataset: str, split: str, out: Path) -> list[str]:
    """Write a TrackEval GT folder (``out/<seq>/...`` + ``out/seqmap.txt``); return seq names."""
    names = []
    for seq in find_sequences(root, dataset):
        write_gt_split(seq, out, split)
        names.append(seq.name)
    write_seqmap(out / "seqmap.txt", names)
    return names

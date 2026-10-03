"""Stage A: detector + re-ID over whole sequences → one cache ``.npz`` per sequence.

Usage (see docs/dgx_guide.md; normally called by scripts/dgx/run.sh)::

    python -m models.stage_a dataset --root $DATA/MOT17/train --glob '*-FRCNN' \
        --dataset MOT17 --split train --out $CACHE/MOT17/ablation/train \
        --det-weights $W/bytetrack_ablation.pth.tar --trained-on "CrowdHuman + MOT17 train-half" \
        --reid-weights $W/osnet_ain_x1_0_msmt17.pth

    python -m models.stage_a video --video gate.mp4 --dataset video --out $CACHE/video/mot17full ...

Resumable: a sequence whose cache already exists and validates is skipped; caches are written
atomically, so an interrupted run never leaves a half-written file. One JSON line per sequence
is appended to ``<out>/stage_a_log.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import platform
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from models.protocols import BGRImage, Detector, Embedder, FrameSource
from tracker.io import CacheFormatError, CacheMeta, SequenceCache, load_cache, save_cache


def _prefetch(it: Iterable[tuple[int, BGRImage]], depth: int) -> Iterator[tuple[int, BGRImage]]:
    """Decode frames on a background thread so the GPU is not waiting on JPEG decoding."""
    q: queue.Queue[Any] = queue.Queue(maxsize=depth)
    sentinel = object()
    err: list[BaseException] = []

    def work() -> None:
        try:
            for item in it:
                q.put(item)
        except BaseException as e:  # re-raised in the consumer
            err.append(e)
        finally:
            q.put(sentinel)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not sentinel:
        yield item
    if err:
        raise err[0]


def _batches(it: Iterable[tuple[int, BGRImage]], n: int) -> Iterator[list[tuple[int, BGRImage]]]:
    buf: list[tuple[int, BGRImage]] = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def process_sequence(
    source: FrameSource,
    detector: Detector,
    embedder: Embedder | None,
    *,
    dataset: str,
    split: str,
    detector_saw_eval_half: bool = False,
    batch_size: int = 8,
    extra: dict[str, Any] | None = None,
) -> SequenceCache:
    """Run detector (+ embedder) on every frame of ``source``; no score filtering here."""
    frames, boxes, scores, embs = [], [], [], []
    t0 = time.perf_counter()
    n_frames = 0
    for batch in _batches(_prefetch(source, 4 * batch_size), batch_size):
        ids, images = zip(*batch, strict=True)
        dets = detector(list(images))
        for fid, img, (tlwh, sc) in zip(ids, images, dets, strict=True):
            n_frames += 1
            if fid != n_frames:
                raise RuntimeError(
                    f"{source.name}: non-consecutive frame {fid} (expected {n_frames})"
                )
            frames.append(np.full(len(sc), fid, np.int32))
            boxes.append(np.asarray(tlwh, np.float32).reshape(-1, 4))
            scores.append(np.asarray(sc, np.float32).reshape(-1))
            if embedder is not None:
                embs.append(embedder(img, boxes[-1]))
    if n_frames != source.seq_length:
        raise RuntimeError(
            f"{source.name}: read {n_frames} frames, seqinfo says {source.seq_length}"
        )
    elapsed = time.perf_counter() - t0
    dim = embedder.dim if embedder is not None else 0
    meta = CacheMeta(
        sequence=source.name,
        dataset=dataset,
        split=split,
        frame_rate=source.frame_rate,
        seq_length=source.seq_length,
        im_width=source.width,
        im_height=source.height,
        detector=detector.info(),
        reid=embedder.info() if embedder is not None else None,
        detector_saw_eval_half=detector_saw_eval_half,
        extra={
            **(extra or {}),
            "stage_a_seconds": round(elapsed, 2),
            "stage_a_fps": round(n_frames / elapsed, 2) if elapsed > 0 else None,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    cat = np.concatenate
    return SequenceCache(
        meta=meta,
        frame=cat(frames) if frames else np.zeros(0, np.int32),
        tlwh=cat(boxes) if boxes else np.zeros((0, 4), np.float32),
        score=cat(scores) if scores else np.zeros(0, np.float32),
        emb=(cat(embs) if embs else np.zeros((0, dim), np.float16)) if embedder else None,
    )


def is_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        load_cache(path)
    except (CacheFormatError, OSError, ValueError, KeyError):
        return False
    return True


def provenance() -> dict[str, Any]:
    info: dict[str, Any] = {"host": platform.node(), "python": sys.version.split()[0]}
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True,
            stderr=subprocess.DEVNULL).strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def run_many(
    sources: Iterable[FrameSource],
    detector: Detector,
    embedder: Embedder | None,
    out_dir: Path,
    *,
    dataset: str,
    split: str,
    detector_saw_eval_half: bool,
    batch_size: int,
    force: bool = False,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "stage_a_log.jsonl"
    prov = provenance()
    written = []
    for src in sources:
        path = out_dir / f"{src.name}.npz"
        if not force and is_done(path):
            print(f"[skip] {path} (already complete)", flush=True)
            written.append(path)
            continue
        print(f"[run ] {src.name}: {src.seq_length} frames", flush=True)
        cache = process_sequence(src, detector, embedder, dataset=dataset, split=split,
                                 detector_saw_eval_half=detector_saw_eval_half,
                                 batch_size=batch_size, extra=prov)  # fmt: skip
        save_cache(path, cache)
        rec = {
            "sequence": src.name,
            "path": str(path),
            "frames": src.seq_length,
            "detections": len(cache),
            "dets_per_frame": round(len(cache) / max(src.seq_length, 1), 2),
            "bytes": path.stat().st_size,
            **{k: cache.meta.extra[k] for k in ("stage_a_seconds", "stage_a_fps")},
        }
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[done] {json.dumps(rec)}", flush=True)
        written.append(path)
    return written


def _build_models(a: argparse.Namespace) -> tuple[Detector, Embedder | None]:
    from models.osnet_embedder import OsnetConfig, OsnetEmbedder
    from models.yolox_detector import YoloxConfig, YoloxDetector

    det = YoloxDetector(YoloxConfig(
        weights=a.det_weights, variant=a.variant, test_size=(a.test_size[0], a.test_size[1]),
        conf_floor=a.conf_floor, nms_thresh=a.nms, device=a.device, fp16=not a.no_fp16,
        trained_on=a.trained_on))  # fmt: skip
    emb = None
    if not a.no_reid:
        emb = OsnetEmbedder(OsnetConfig(weights=a.reid_weights, arch=a.reid_arch,
                                        device=a.device, fp16=not a.no_fp16))  # fmt: skip
    return det, emb


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m models.stage_a", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = p.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", required=True)
    common.add_argument("--split", default="train")
    common.add_argument("--out", required=True, type=Path, help="output directory")
    common.add_argument("--det-weights", default=None)
    common.add_argument("--variant", default="yolox_x")
    common.add_argument("--test-size", nargs=2, type=int, default=[800, 1440], metavar=("H", "W"))
    common.add_argument("--conf-floor", type=float, default=0.05)
    common.add_argument("--nms", type=float, default=0.7)
    common.add_argument("--trained-on", default="", help="detector training data (audit trail)")
    common.add_argument(
        "--saw-eval-half",
        action="store_true",
        help="detector was trained on frames this cache is evaluated on",
    )
    common.add_argument("--reid-weights", default=None)
    common.add_argument("--reid-arch", default="osnet_ain_x1_0")
    common.add_argument("--no-reid", action="store_true")
    common.add_argument("--device", default="cuda")
    common.add_argument("--no-fp16", action="store_true")
    common.add_argument("--batch-size", type=int, default=8)
    common.add_argument("--force", action="store_true", help="recompute even if complete")
    d = sub.add_parser("dataset", parents=[common], help="all sequences under a split folder")
    d.add_argument("--root", required=True, type=Path)
    d.add_argument("--glob", default="*")
    v = sub.add_parser("video", parents=[common], help="a single video file")
    v.add_argument("--video", required=True, type=Path)
    v.add_argument("--name", default=None)
    a = p.parse_args(argv)

    from models.frames import ImageFolderSource, VideoSource

    sources: list[FrameSource]
    if a.mode == "dataset":
        dirs = sorted(x for x in a.root.glob(a.glob) if (x / "seqinfo.ini").exists())
        if not dirs:
            p.error(f"no sequences with seqinfo.ini under {a.root} matching {a.glob!r}")
        sources = [ImageFolderSource(x) for x in dirs]
    else:
        sources = [VideoSource(a.video, a.name)]
    det, emb = _build_models(a)
    run_many(sources, det, emb, a.out, dataset=a.dataset, split=a.split,
             detector_saw_eval_half=a.saw_eval_half, batch_size=a.batch_size,
             force=a.force)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

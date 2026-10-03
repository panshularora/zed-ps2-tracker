"""Stage B command line: run a tracker config over caches and write MOT results.

    python -m tracker.cli --config configs/baselines/bytetrack.yaml \
        --caches cache/SOMPT22/mot17full/test --out results/SOMPT22/bytetrack_like/test \
        [--jsonl] [--zip results.zip] [-o association.match_thresh=0.7]

Inputs are Stage A ``.npz`` caches (or, with ``--mot-root``, MOTChallenge ``det/det.txt`` public
detections). Writes ``<out>/<sequence>.txt`` in MOTChallenge format, optional per-frame JSONL
(``<out>/<sequence>.frames.jsonl``) and ``<out>/latency.json``.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from tracker.config import config_to_dict, load_config
from tracker.io import (
    SequenceCache,
    load_cache,
    load_mot_det_txt,
    write_frames_jsonl,
    write_mot_results,
)
from tracker.runner import run_sequence


def latency_summary(ms: np.ndarray) -> dict[str, float]:
    if ms.size == 0:
        return {"frames": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0}
    return {
        "frames": int(ms.size),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "max_ms": float(ms.max()),
        "mean_ms": float(ms.mean()),
    }


def iter_inputs(args: argparse.Namespace) -> list[SequenceCache]:
    if args.caches is not None:
        paths = sorted(Path(args.caches).glob("*.npz"))
        if not paths:
            raise SystemExit(f"no .npz caches in {args.caches}")
        return [load_cache(p) for p in paths]
    root = Path(args.mot_root)
    seqs = sorted(d for d in root.glob(args.glob) if (d / "det" / "det.txt").exists())
    if not seqs:
        raise SystemExit(f"no sequences with det/det.txt under {root} matching {args.glob!r}")
    return [load_mot_det_txt(d) for d in seqs]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tracker.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--caches", type=Path, help="directory of Stage A .npz caches")
    src.add_argument("--mot-root", type=Path, help="MOT split dir with <seq>/det/det.txt")
    p.add_argument("--glob", default="*", help="sequence glob for --mot-root (e.g. '*-FRCNN')")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("-o", "--override", action="append", default=[], help="key=value")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jsonl", action="store_true", help="also write per-frame JSON Lines")
    p.add_argument("--zip", type=Path, default=None, help="zip <out>/*.txt as results/<seq>.txt")
    a = p.parse_args(argv)

    cfg = load_config(a.config, a.override)
    a.out.mkdir(parents=True, exist_ok=True)
    lat_all, per_seq = [], {}
    for cache in iter_inputs(a):
        name = cache.meta.sequence
        run = run_sequence(cache, cfg)
        write_mot_results(a.out / f"{name}.txt", run.results)
        if a.jsonl:
            hdr = {"sequence": name, "tracker": cfg.name, "frame_rate": cache.meta.frame_rate,
                   "im_width": cache.meta.im_width, "im_height": cache.meta.im_height}  # fmt: skip
            write_frames_jsonl(a.out / f"{name}.frames.jsonl", run.results, hdr)
        per_seq[name] = latency_summary(run.latency_ms)
        lat_all.append(run.latency_ms)
        s = per_seq[name]
        print(f"{name}: {s['frames']} frames, p50 {s['p50_ms']:.2f} ms, p95 {s['p95_ms']:.2f} ms, "
              f"max {s['max_ms']:.2f} ms")  # fmt: skip
    summary = {
        "config": config_to_dict(cfg),
        "combined": latency_summary(np.concatenate(lat_all)),
        "per_sequence": per_seq,
    }
    (a.out / "latency.json").write_text(json.dumps(summary, indent=2))
    if a.zip is not None:
        with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for txt in sorted(a.out.glob("*.txt")):
                zf.write(txt, f"results/{txt.name}")
        print(f"wrote {a.zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

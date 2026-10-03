"""Tracker latency on CPU: wall-clock time of each ``update()`` call, single-threaded.

    python -m eval.latency --dataset MOT17 --root data/MOT17 --split val_half --dets public \
        --config configs/baselines/bytetrack.yaml --out results/latency/bytetrack.json

Protocol (docs/evaluation.md): ``OMP_NUM_THREADS=1`` (and MKL/OpenBLAS equivalents) are set
before NumPy is imported, one untimed warm-up pass, then ``--repeats`` timed passes over every
sequence. We report p50 / p95 / p99 / max over all timed frames, per sequence and combined, with
the CPU model. Only the tracker is timed (detector and re-ID run in Stage A; file I/O is excluded).
"""

from __future__ import annotations

import os
import sys

_NUMPY_PRELOADED = "numpy" in sys.modules
"""If NumPy was imported before this module, its thread pools may already be sized."""
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def summarise_latency(ms: npt.NDArray[np.float64]) -> dict[str, float]:
    if ms.size == 0:
        return {"frames": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0,
                "mean_ms": 0.0, "frames_over_10ms": 0}  # fmt: skip
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    return {
        "frames": int(ms.size), "p50_ms": float(p50), "p95_ms": float(p95), "p99_ms": float(p99),
        "max_ms": float(ms.max()), "mean_ms": float(ms.mean()),
        "frames_over_10ms": int((ms > 10.0).sum()),
    }  # fmt: skip


def measure(dataset: str, root: Path, split: str, dets: str, config: Path,
            overrides: list[str], repeats: int = 3) -> dict[str, Any]:  # fmt: skip
    from eval.run import load_inputs
    from eval.splits import crop_cache, find_sequences
    from tracker.config import config_to_dict, load_config
    from tracker.runner import run_sequence

    cfg = load_config(config, overrides)
    caches = [
        crop_cache(load_inputs(s, dataset, dets), split) for s in find_sequences(root, dataset)
    ]
    run_sequence(caches[0], cfg)  # warm-up (imports, allocator, caches)
    per_seq: dict[str, Any] = {}
    all_ms, all_nd = [], []
    for c in caches:
        runs = [run_sequence(c, cfg) for _ in range(repeats)]
        ms = np.concatenate([r.latency_ms for r in runs])
        nd = np.concatenate([r.n_dets for r in runs])
        per_seq[c.meta.sequence] = {**summarise_latency(ms), "dets_per_frame": float(nd.mean())}
        all_ms.append(ms)
        all_nd.append(nd)
    ms, nd = np.concatenate(all_ms), np.concatenate(all_nd)
    edges = [0, 10, 20, 40, 80, 160, 10**9]
    by_dets = {}
    for lo, hi in itertools.pairwise(edges):
        m = (nd >= lo) & (nd < hi)
        if m.any():
            label = f"{lo}+" if hi == 10**9 else f"{lo}-{hi - 1}"
            by_dets[label] = summarise_latency(ms[m])
    return {
        "dataset": dataset, "split": split, "dets": dets, "config_file": str(config),
        "overrides": overrides, "config": config_to_dict(cfg), "repeats": repeats,
        "combined": summarise_latency(ms), "per_sequence": per_seq, "by_input_dets": by_dets,
        "cpu": cpu_model(), "logical_cpus": os.cpu_count(), "python": platform.python_version(),
        "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "numpy_imported_before_env": _NUMPY_PRELOADED,
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.latency", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--dataset", required=True, choices=["MOT17", "MOT20", "DanceTrack"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", default="val_half", choices=["val_half", "train_half", "full"])
    p.add_argument("--dets", default="public")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    res = measure(a.dataset, a.root, a.split, a.dets, a.config, a.override, a.repeats)
    c = res["combined"]
    print(f"{res['cpu']} | OMP_NUM_THREADS={res['omp_num_threads']} | {c['frames']} timed frames")
    for seq, s in res["per_sequence"].items():
        print(
            f"  {seq}: {s['dets_per_frame']:.0f} dets/frame  p50 {s['p50_ms']:.2f}  "
            f"p95 {s['p95_ms']:.2f}  p99 {s['p99_ms']:.2f}  max {s['max_ms']:.2f} ms"
        )
    print(f"  combined: p50 {c['p50_ms']:.2f}  p95 {c['p95_ms']:.2f}  p99 {c['p99_ms']:.2f}  "
          f"max {c['max_ms']:.2f} ms")  # fmt: skip
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

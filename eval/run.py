"""Run a tracker config on a dataset split and score it: tracking → TrackEval → stress bins.

    python -m eval.run --dataset MOT17 --root data/MOT17 --split val_half \
        --dets public --config configs/baselines/bytetrack.yaml --name bytetrack_public \
        --out results/mot17_val_half

``--dets public`` uses MOTChallenge ``det/det.txt`` (no embeddings); otherwise pass a directory
of Stage A caches (``<seq>.npz``, full sequences; the split is applied here). Output layout::

    OUT/gt/<seq>/{gt/gt.txt,seqinfo.ini}, OUT/gt/seqmap.txt     TrackEval GT for the split
    OUT/trackers/<name>/data/<seq>.txt                           MOTChallenge results
    OUT/trackers/<name>/run.json                                 config, inputs, latency, provenance
    OUT/metrics/<name>/metrics.{json,md}                         TrackEval HOTA / CLEAR / Identity
    OUT/stress/<name>/stress.{json,md}                           visibility and density bins

Strictly online: each frame's output is produced by ``tracker.update(dets_t)`` before frame
``t + 1`` is read, and nothing is post-processed (no interpolation, no offline smoothing).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from eval.latency import cpu_model, summarise_latency
from eval.splits import crop_cache, find_sequences, prepare_gt
from tracker.config import config_to_dict, load_config
from tracker.io import SequenceCache, load_cache, load_mot_det_txt, write_mot_results
from tracker.runner import run_sequence

ROOT = Path(__file__).resolve().parents[1]


def benchmark_for(dataset: str) -> str:
    """TrackEval BENCHMARK: MOT20 adds class 6 (non_mot_vehicle) to the distractor classes."""
    return "MOT20" if dataset.upper() == "MOT20" else "MOT17"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)  # fmt: skip
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "tracker",
                                "eval", "configs"], capture_output=True, text=True)  # fmt: skip
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_inputs(seq_dir: Path, dataset: str, dets: str) -> SequenceCache:
    if dets == "public":
        return load_mot_det_txt(seq_dir, dataset=dataset, split="train")
    path = Path(dets) / f"{seq_dir.name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"no cache {path} for {seq_dir.name}")
    return load_cache(path)


def run(
    dataset: str, root: Path, split: str, dets: str, config: Path, overrides: list[str],
    name: str, out: Path, evaluate: bool = True, stress: bool = True,
) -> dict[str, Any]:  # fmt: skip
    cfg = load_config(config, overrides)
    gt_dir = out / "gt"
    names = prepare_gt(root, dataset, split, gt_dir)
    trk_dir = out / "trackers" / name
    (trk_dir / "data").mkdir(parents=True, exist_ok=True)
    lat_all, per_seq, saw = [], {}, []
    for seq_dir in find_sequences(root, dataset):
        cache = crop_cache(load_inputs(seq_dir, dataset, dets), split)
        saw.append(bool(cache.meta.detector_saw_eval_half))
        res = run_sequence(cache, cfg)
        write_mot_results(trk_dir / "data" / f"{seq_dir.name}.txt", res.results)
        per_seq[seq_dir.name] = summarise_latency(res.latency_ms)
        per_seq[seq_dir.name]["dets_per_frame"] = (
            float(np.mean(res.n_dets)) if res.n_dets.size else 0.0
        )
        lat_all.append(res.latency_ms)
    info: dict[str, Any] = {
        "dataset": dataset, "split": split, "sequences": names, "dets": dets,
        "detector_saw_eval_half": any(saw), "config_file": str(config), "overrides": overrides,
        "config": config_to_dict(cfg), "git_commit": git_commit(),
        "latency_note": "tracker update() only (no detector, no I/O); not a pinned-thread "
                        "measurement; use `python -m eval.latency` for the reported numbers",
        "latency": {"combined": summarise_latency(np.concatenate(lat_all)),
                    "per_sequence": per_seq, "cpu": cpu_model(),
                    "python": platform.python_version()},
    }  # fmt: skip
    (trk_dir / "run.json").write_text(json.dumps(info, indent=2))
    if evaluate:
        from eval.trackeval_runner import evaluate as te_eval
        from eval.trackeval_runner import format_table

        bench = benchmark_for(dataset)
        summ = te_eval(gt_dir, out / "trackers", bench, [name], write_outputs=False)[name]
        mdir = out / "metrics" / name
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "metrics.json").write_text(json.dumps(summ, indent=2))
        (mdir / "metrics.md").write_text(format_table(summ) + "\n")
        info["metrics"] = summ
        print(f"\n{name} | {dataset} {split} | dets={dets} | BENCHMARK={bench}")
        print(format_table(summ))
        if stress:
            from eval.stress import run_stress, stress_markdown

            st = run_stress(gt_dir, out / "trackers", name, bench, check_against=summ)
            sdir = out / "stress" / name
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / "stress.json").write_text(json.dumps(st, indent=2))
            (sdir / "stress.md").write_text(stress_markdown(st) + "\n")
            info["stress"] = st
            print("\n" + stress_markdown(st))
    return info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--dataset", required=True, choices=["MOT17", "MOT20", "DanceTrack"])
    p.add_argument("--root", type=Path, required=True, help="dataset root containing train/")
    p.add_argument("--split", default="val_half", choices=["val_half", "train_half", "full"])
    p.add_argument("--dets", default="public", help="'public' or a directory of Stage A caches")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("-o", "--override", action="append", default=[], help="dotted key=value")
    p.add_argument("--name", required=True, help="tracker name (output sub-folder)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--no-stress", action="store_true")
    a = p.parse_args(argv)
    run(a.dataset, a.root, a.split, a.dets, a.config, a.override, a.name, a.out,
        evaluate=not a.no_eval, stress=not a.no_stress)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

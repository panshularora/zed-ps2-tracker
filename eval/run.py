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
    det_variant: str = "FRCNN", score_map: str = "none",
    seq_overrides: dict[str, list[str]] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """``score_map = logistic`` maps raw detector scores with ``1 / (1 + e^-s)``. That is needed
    for MOT17 DPM, whose public scores are unbounded (about -0.5 to 4.8)."""
    cfg = load_config(config, overrides)
    gt_dir = out / "gt"
    names = prepare_gt(root, dataset, split, gt_dir, det_variant)
    trk_dir = out / "trackers" / name
    (trk_dir / "data").mkdir(parents=True, exist_ok=True)
    lat_all: list[Any] = []
    per_seq: dict[str, dict[str, Any]] = {}
    saw: list[bool] = []
    for seq_dir in find_sequences(root, dataset, det_variant):
        cache = crop_cache(load_inputs(seq_dir, dataset, dets), split)
        if score_map == "logistic":
            sc = (1.0 / (1.0 + np.exp(-cache.score.astype(np.float64)))).astype(np.float32)
            cache = SequenceCache(cache.meta, cache.frame, cache.tlwh, sc, cache.emb)
        elif score_map != "none":
            raise ValueError(f"unknown score_map {score_map!r}")
        saw.append(bool(cache.meta.detector_saw_eval_half))
        extra = [o for pre, ov in (seq_overrides or {}).items()
                 if seq_dir.name.startswith(pre) for o in ov]  # fmt: skip
        res = run_sequence(cache, load_config(config, [*overrides, *extra]) if extra else cfg)
        write_mot_results(trk_dir / "data" / f"{seq_dir.name}.txt", res.results)
        per_seq[seq_dir.name] = {**summarise_latency(res.latency_ms), "tracker_info": res.info}
        per_seq[seq_dir.name]["dets_per_frame"] = (
            float(np.mean(res.n_dets)) if res.n_dets.size else 0.0
        )
        lat_all.append(res.latency_ms)
    info: dict[str, Any] = {
        "dataset": dataset, "split": split, "sequences": names, "dets": dets,
        "detector_saw_eval_half": any(saw), "det_variant": det_variant,
        "score_map": score_map, "seq_overrides": seq_overrides or {},
        "config_file": str(config), "overrides": overrides,
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
    p.add_argument(
        "--det-variant",
        default="FRCNN",
        choices=["FRCNN", "SDP", "DPM"],
        help="MOT17 public detector copy",
    )
    p.add_argument(
        "--score-map",
        default="none",
        choices=["none", "logistic"],
        help="logistic: map unbounded scores (MOT17 DPM) into (0, 1)",
    )
    p.add_argument("--seq-override", action="append", default=[], metavar="SEQ_PREFIX:key=value",
                   help="override for sequences whose name starts with SEQ_PREFIX "
                        "(ByteTrack's evaluator: MOT17-05:lifecycle.track_buffer=14)")  # fmt: skip
    a = p.parse_args(argv)
    run(a.dataset, a.root, a.split, a.dets, a.config, a.override, a.name, a.out,
        evaluate=not a.no_eval, stress=not a.no_stress, det_variant=a.det_variant,
        score_map=a.score_map, seq_overrides=parse_seq_overrides(a.seq_override))  # fmt: skip
    return 0


def parse_seq_overrides(items: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for it in items:
        pre, sep, ov = it.partition(":")
        if not sep or not pre or "=" not in ov:
            raise SystemExit(f"--seq-override wants SEQ_PREFIX:key=value, got {it!r}")
        out.setdefault(pre, []).append(ov)
    return out


if __name__ == "__main__":
    raise SystemExit(main())

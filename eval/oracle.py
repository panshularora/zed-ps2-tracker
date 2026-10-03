"""Visibility oracle: the ceiling for a tracker that never outputs a box for an invisible person.

MOT17/MOT20 gt keeps pedestrians with visibility 0 (fully occluded) as boxes to be found, and
TrackEval does not filter by visibility. This tool writes the *gt itself* as a tracker result,
keeping only boxes with visibility > ``tau``, and scores it. With ``tau = 0`` that is a perfect
detector + perfect identities that simply does not coast through full occlusion.

    python -m eval.oracle --dataset MOT17 --root data/MOT17 --split val_half --tau 0 --out OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.run import benchmark_for
from eval.splits import prepare_gt
from eval.trackeval_runner import evaluate, format_table


def write_oracle(gt_dir: Path, names: list[str], trk_dir: Path, tau: float) -> None:
    (trk_dir / "data").mkdir(parents=True, exist_ok=True)
    for seq in names:
        g = np.loadtxt(gt_dir / seq / "gt" / "gt.txt", delimiter=",", ndmin=2)
        g = g[(g[:, 6] == 1) & (g[:, 7] == 1) & (g[:, 8] > tau)]
        lines = [f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},1,-1,-1,-1\n"
                 for r in g]  # fmt: skip
        (trk_dir / "data" / f"{seq}.txt").write_text("".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.oracle", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--dataset", required=True, choices=["MOT17", "MOT20"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", default="val_half", choices=["val_half", "train_half", "full"])
    p.add_argument("--tau", type=float, nargs="+", default=[0.0])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    names = prepare_gt(a.root, a.dataset, a.split, a.out / "gt")
    res = {}
    for tau in a.tau:
        name = f"oracle_vis_gt_{tau:g}"
        write_oracle(a.out / "gt", names, a.out / "trackers" / name, tau)
        summ = evaluate(a.out / "gt", a.out / "trackers", benchmark_for(a.dataset), [name],
                        write_outputs=False)[name]  # fmt: skip
        res[name] = summ["COMBINED"]
        print(f"\n{name} ({a.dataset} {a.split})\n{format_table(summ)}")
    (a.out / "oracle.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

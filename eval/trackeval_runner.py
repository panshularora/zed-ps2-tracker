"""Official metrics (HOTA, CLEAR/MOTA, Identity/IDF1) via TrackEval.

TrackEval is vendored unmodified at ``third_party/trackeval`` (commit 12c8791, MIT, see its
LICENSE). That commit still uses ``np.float``/``np.int``/``np.bool``, which NumPy ≥ 1.24 removed,
so :func:`import_trackeval` restores those aliases before importing it (the same shim as
Research Lab's ``scripts/run_trackeval.py``). The aliases map to the builtins, which is exactly
what they were in NumPy < 1.24, so results are unchanged.

``BENCHMARK`` matters: with ``MOT20`` TrackEval also treats class 6 (``non_mot_vehicle``) as a
distractor, so tracker boxes matched to it are not counted as false positives.

    python -m eval.trackeval_runner --gt OUT/gt --trackers OUT/trackers --benchmark MOT17 \
        --out OUT/metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "third_party" / "trackeval"


def import_trackeval() -> ModuleType:
    for name, typ in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, name):
            setattr(np, name, typ)
    if str(VENDORED) not in sys.path:
        sys.path.insert(0, str(VENDORED))
    import trackeval

    return trackeval  # type: ignore[no-any-return]


def dataset_config(gt: Path, trackers: Path, benchmark: str, names: list[str] | None = None
                   ) -> dict[str, Any]:  # fmt: skip
    te = import_trackeval()
    cfg: dict[str, Any] = te.datasets.MotChallenge2DBox.get_default_dataset_config()
    cfg.update({
        "GT_FOLDER": str(gt),
        "TRACKERS_FOLDER": str(trackers),
        "SEQMAP_FILE": str(gt / "seqmap.txt"),
        "SKIP_SPLIT_FOL": True,
        "BENCHMARK": benchmark,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "trackeval",
        "PRINT_CONFIG": False,
    })  # fmt: skip
    if names is not None:
        cfg["TRACKERS_TO_EVAL"] = names
    return cfg


SUMMARY_FIELDS = {
    "HOTA": ("HOTA", "HOTA"), "DetA": ("HOTA", "DetA"), "AssA": ("HOTA", "AssA"),
    "MOTA": ("CLEAR", "MOTA"), "IDF1": ("Identity", "IDF1"),
    "IDSW": ("CLEAR", "IDSW"), "TP": ("CLEAR", "CLR_TP"), "FP": ("CLEAR", "CLR_FP"),
    "FN": ("CLEAR", "CLR_FN"), "IDTP": ("Identity", "IDTP"), "IDFP": ("Identity", "IDFP"),
    "IDFN": ("Identity", "IDFN"), "Frag": ("CLEAR", "Frag"), "MT": ("CLEAR", "MT"),
    "ML": ("CLEAR", "ML"),
}  # fmt: skip
PERCENT = {"HOTA", "DetA", "AssA", "MOTA", "IDF1"}


def summarise(res: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, (metric, field) in SUMMARY_FIELDS.items():
        v = res[metric][field]
        v = float(np.mean(v)) if np.ndim(v) else float(v)
        out[key] = 100.0 * v if key in PERCENT else v
    out["GT"] = out["TP"] + out["FN"]
    return out


def evaluate(gt: Path, trackers: Path, benchmark: str, names: list[str] | None = None,
             write_outputs: bool = True) -> dict[str, dict[str, dict[str, float]]]:  # fmt: skip
    """Return ``{tracker: {sequence | 'COMBINED': summary}}`` (percentages for the scores)."""
    te = import_trackeval()
    ev_cfg = te.Evaluator.get_default_eval_config()
    ev_cfg.update({
        "USE_PARALLEL": False, "PRINT_CONFIG": False, "PRINT_RESULTS": False,
        "OUTPUT_SUMMARY": write_outputs, "OUTPUT_DETAILED": write_outputs,
        "PLOT_CURVES": False, "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_EMPTY_CLASSES": False,
    })  # fmt: skip
    ds = te.datasets.MotChallenge2DBox(dataset_config(gt, trackers, benchmark, names))
    quiet = {"PRINT_CONFIG": False}
    metrics = [te.metrics.HOTA(), te.metrics.CLEAR(quiet), te.metrics.Identity(quiet)]
    res, msg = te.Evaluator(ev_cfg).evaluate([ds], metrics)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for trk, per_seq in res["MotChallenge2DBox"].items():
        if msg["MotChallenge2DBox"][trk] != "Success":
            raise RuntimeError(f"TrackEval failed for {trk}: {msg['MotChallenge2DBox'][trk]}")
        out[trk] = {
            ("COMBINED" if seq == "COMBINED_SEQ" else seq): summarise(r["pedestrian"])
            for seq, r in per_seq.items()
        }
    return out


def format_table(summary: dict[str, dict[str, float]]) -> str:
    cols = ["HOTA", "IDF1", "MOTA", "DetA", "AssA", "IDSW", "FP", "FN", "GT"]
    lines = ["| Sequence | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for seq in sorted(summary, key=lambda s: (s == "COMBINED", s)):
        s = summary[seq]
        vals = [f"{s[c]:.1f}" if c in PERCENT else f"{round(s[c])}" for c in cols]
        name = f"**{seq}**" if seq == "COMBINED" else seq
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.trackeval_runner", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--gt", type=Path, required=True, help="GT folder with seqmap.txt")
    p.add_argument("--trackers", type=Path, required=True, help="<dir>/<tracker>/data/<seq>.txt")
    p.add_argument("--benchmark", default="MOT17", choices=["MOT17", "MOT20", "MOT16"])
    p.add_argument("--tracker-name", action="append", default=None)
    p.add_argument("--out", type=Path, default=None, help="write metrics.json + metrics.md here")
    a = p.parse_args(argv)
    res = evaluate(a.gt, a.trackers, a.benchmark, a.tracker_name)
    for trk, summ in res.items():
        print(f"\n{trk} ({a.benchmark})\n{format_table(summ)}")
        if a.out is not None:
            d = a.out / trk
            d.mkdir(parents=True, exist_ok=True)
            (d / "metrics.json").write_text(json.dumps(summ, indent=2))
            (d / "metrics.md").write_text(format_table(summ) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Quick ByteTrack reproduction check on one (or a few) MOT17 val-half sequences.

    python -m eval.repro_check --cache-dir $ZED_ROOT/stagea/cache/MOT17/ablation/train \\
        --root $ZED_ROOT/stagea/gt/MOT17 --seq MOT17-09 --out /tmp/repro

1. **Detector audit** (no GPU): reads each cache's detector metadata and lists every setting that
   differs from ByteTrack's official MOT17 ablation run (``tools/track.py -f
   exps/example/mot/yolox_x_ablation.py -c bytetrack_ablation.pth.tar --fp16 --fuse``).
2. **Tracker**: runs ``bytetrack.yaml`` (our baseline) and ``bytetrack_official.yaml`` (+ the
   evaluator's per-sequence track_buffer) on the chosen sequences' val half and prints
   MOTA / IDF1 / IDSW / FP / FN side by side.

The paper reports only the 7-sequence total (MOTA 76.6, IDF1 79.3, IDs 159, FP 3358, FN 9081), so a
one-sequence run is a smoke test: compare it with a previous run of the same sequence, and run all
seven (``--seq all``) for the real comparison. Exit code 1 if the audit finds a mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VAL_HALF_SEQS = ("MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13")
#: ByteTrack's evaluator changes track_buffer for these sequences (mot_evaluator.py, l.139-144)
BYTETRACK_SEQ_OVERRIDES = {"MOT17-05": ["lifecycle.track_buffer=14"],
                           "MOT17-13": ["lifecycle.track_buffer=25"]}  # fmt: skip
#: what ByteTrack's official MOT17 ablation run uses; ``None`` = not checked
OFFICIAL: dict[str, Any] = {
    "weights": "bytetrack_ablation.pth.tar",
    "test_size": [800, 1440],
    "nms_thresh": 0.7,
    "clip_boxes": False,
    "fp16": True,
}
REFERENCE = {"MOTA": 76.6, "IDF1": 79.3, "IDSW": 159, "FP": 3358, "FN": 9081}


def audit_detector(det: dict[str, Any], fp32_ok: bool = True) -> list[str]:
    """Human-readable mismatches between a cache's detector metadata and ByteTrack's setup."""
    issues = []
    for k, want in OFFICIAL.items():
        got = det.get(k)
        if k == "clip_boxes" and got is None:
            issues.append(
                "clip_boxes: not recorded (cache made before 13 Oct 2026, boxes were "
                "clipped to the image; ByteTrack does not clip) -> re-run Stage A"
            )
            continue
        if k == "fp16" and fp32_ok and got is False:
            continue  # FP32 is a deliberate diagnostic, not a mismatch
        if got != want:
            issues.append(f"{k}: cache has {got!r}, ByteTrack uses {want!r}")
    cf = det.get("conf_floor")
    if cf is not None and cf > 0.1:
        issues.append(f"conf_floor {cf} > 0.1 drops ByteTrack's low-score (BYTE) detections")
    return issues


def _metrics(summ: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}

    def walk(d: Any) -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                if k in REFERENCE and isinstance(v, int | float):
                    flat.setdefault(k, float(v))
                else:
                    walk(v)

    walk(summ)
    return flat


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.repro_check", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--cache-dir", type=Path, required=True, help="MOT17/ablation/train caches")
    p.add_argument("--root", type=Path, required=True, help="MOT17 root with train/<seq>/gt")
    p.add_argument(
        "--seq",
        action="append",
        default=[],
        help="MOT17-09 (repeatable) or 'all' (the 7 val-half sequences); default MOT17-09",
    )
    p.add_argument("--out", type=Path, default=Path("outputs/repro_check"))
    p.add_argument("--audit-only", action="store_true", help="no tracking, metadata audit only")
    a = p.parse_args(argv)
    seqs = a.seq or ["MOT17-09"]
    if "all" in seqs:
        seqs = list(VAL_HALF_SEQS)

    from tracker.io import load_cache

    bad = 0
    for s in seqs:
        path = a.cache_dir / f"{s}-FRCNN.npz"
        if not path.exists():
            print(f"{s}: no cache {path}", file=sys.stderr)
            return 2
        issues = audit_detector(load_cache(path).meta.detector)
        bad += len(issues)
        print(f"[audit] {s}: " + ("matches ByteTrack's detector setup" if not issues else ""))
        for i in issues:
            print(f"        - {i}")
    if a.audit_only:
        return 1 if bad else 0

    from eval.run import run

    sub = a.out / "root" / "train"
    sub.mkdir(parents=True, exist_ok=True)
    for s in seqs:
        link = sub / f"{s}-FRCNN"
        if not link.exists():
            link.symlink_to((a.root / "train" / f"{s}-FRCNN").resolve())
    rows = {}
    for name, cfg, sov in (("ours_bytetrack", "bytetrack.yaml", None),
                           ("bytetrack_official", "bytetrack_official.yaml",
                            BYTETRACK_SEQ_OVERRIDES)):  # fmt: skip
        info = run("MOT17", a.out / "root", "val_half", str(a.cache_dir),
                   ROOT / "configs" / "baselines" / cfg, [], name, a.out / name,
                   stress=False, seq_overrides=sov)  # fmt: skip
        rows[name] = _metrics(info.get("metrics", {}))
    print(f"\nval half of {', '.join(seqs)}")
    print("| run | " + " | ".join(REFERENCE) + " |")
    print("|---|" + "---|" * len(REFERENCE))
    for name, m in rows.items():
        print(
            f"| {name} | " + " | ".join(f"{m.get(k, float('nan')):.2f}" for k in REFERENCE) + " |"
        )
    if len(seqs) == len(VAL_HALF_SEQS):
        print("| ByteTrack paper | " + " | ".join(str(v) for v in REFERENCE.values()) + " |")
    else:
        print("(the paper has no per-sequence numbers: use --seq all for the comparison)")
    res = {"sequences": seqs, "rows": rows, "audit_issues": bad}
    (a.out / "repro_check.json").write_text(json.dumps(res, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

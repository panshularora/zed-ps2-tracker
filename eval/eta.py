"""Stage A throughput and time-to-finish estimate, from ``stage_a_log.jsonl``.

``run.sh all`` calls this after the first MOT17 sequence is cached:

    python -m eval.eta --log CACHE/MOT17/ablation/train/stage_a_log.jsonl \\
        --seqs DATA/MOT17/train --glob '*-FRCNN' --later MOT20=8931x2

It prints the measured frames per second (detector + re-ID, end to end) and the estimated time
for the remaining sequences of this dataset, plus later passes given as ``NAME=FRAMES[xPASSES]``
(``NAME=?`` for an unknown frame count). Later datasets use a different input size and crowd
density, so their estimate is labelled as a rough extrapolation of this fps.
"""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
from typing import Any


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line being written right now
    return out


def seq_lengths(root: Path, glob: str = "*") -> dict[str, int]:
    out = {}
    for ini in sorted(root.glob(f"{glob}/seqinfo.ini")):
        cp = configparser.ConfigParser()
        cp.read(ini)
        out[ini.parent.name] = int(cp["Sequence"]["seqLength"])
    return out


def fmt_dur(seconds: float) -> str:
    m = round(seconds / 60)
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def parse_later(items: list[str]) -> list[tuple[str, int | None]]:
    out: list[tuple[str, int | None]] = []
    for it in items:
        name, _, spec = it.partition("=")
        if not spec or spec == "?":
            out.append((name, None))
            continue
        frames, _, passes = spec.partition("x")
        out.append((name, int(frames) * (int(passes) if passes else 1)))
    return out


def estimate(log: list[dict[str, Any]], lengths: dict[str, int],
             later: list[tuple[str, int | None]]) -> dict[str, Any]:  # fmt: skip
    done = {r["sequence"] for r in log}
    frames = sum(int(r["frames"]) for r in log)
    secs = sum(float(r["stage_a_seconds"]) for r in log)
    fps = frames / secs if secs > 0 else None
    remaining = sum(n for s, n in lengths.items() if s not in done)
    res: dict[str, Any] = {"sequences_done": len(done), "frames_done": frames, "fps": fps,
                           "remaining_frames": remaining, "later": []}  # fmt: skip
    if fps:
        res["remaining_seconds"] = remaining / fps
        for name, n in later:
            res["later"].append({"name": name, "frames": n,
                                 "seconds": n / fps if n is not None else None})  # fmt: skip
    return res


def message(e: dict[str, Any]) -> str:
    if not e["fps"]:
        return "Stage A: no finished sequence yet, no estimate"
    parts = [f"Stage A measured {e['fps']:.1f} frames/s (detector + re-ID) over "
             f"{e['frames_done']} frames; rest of this dataset ({e['remaining_frames']} frames): "
             f"~{fmt_dur(e['remaining_seconds'])}"]  # fmt: skip
    rough = [f"{x['name']} ~{fmt_dur(x['seconds'])}" if x["seconds"] is not None
             else f"{x['name']} unknown" for x in e["later"]]  # fmt: skip
    if rough:
        parts.append("rough extrapolation at the same fps (larger inputs run slower): "
                     + ", ".join(rough))  # fmt: skip
    return "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.eta", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--seqs", type=Path, required=True, help="folder of sequence folders")
    p.add_argument("--glob", default="*")
    p.add_argument("--later", nargs="*", default=[], metavar="NAME=FRAMES[xPASSES]")
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    e = estimate(read_log(a.log), seq_lengths(a.seqs, a.glob), parse_later(a.later))
    if a.json:
        a.json.write_text(json.dumps(e, indent=1))
    print(message(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

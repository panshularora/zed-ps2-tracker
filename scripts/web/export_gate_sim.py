"""Record one seeded run of the synthetic gate scene with the real trackers.

    python scripts/web/export_gate_sim.py [--seed 7] [--seconds 30] [--play crowd|keep]

Writes
* web/public/offline/gate-sim.ndjson (crowd) or gate-keep.ndjson (keep play);
* docs/results/synthetic_gate/seed<seed>.json for crowd runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from serve.gate_sim import SimParams, simulate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--play", choices=("crowd", "keep"), default="crowd")
    a = ap.parse_args()
    p = SimParams(seed=a.seed, seconds=12.0 if a.play == "keep" else a.seconds, play=a.play)
    name = "gate-keep.ndjson" if a.play == "keep" else "gate-sim.ndjson"
    stream = ROOT / "web" / "public" / "offline" / name
    stream.parent.mkdir(parents=True, exist_ok=True)
    end = None
    with stream.open("w") as fh:
        for x in simulate(p):
            fh.write(json.dumps(x, separators=(",", ":")) + "\n")
            end = x
    if a.play == "keep":
        print(f"wrote {stream.relative_to(ROOT)}: {end['summary']}")
        return 0
    full = list(simulate(p, ("bytetrack", "botsort", "ours")))
    end = full[-1]
    out = ROOT / "docs" / "results" / "synthetic_gate" / f"seed{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "source": "scripts/web/export_gate_sim.py (serve/gate_sim.py, real trackers)",
                "params": full[0]["params"],
                "trackers": full[0]["trackers"],
                "frames": end["frames"],
                "true_count": end["true_count"],
                "summary": end["summary"],
                "note": (
                    "Synthetic scene with exact ground truth; one seed, one run. "
                    "Static camera, so BoT-SORT runs without CMC."
                ),
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {stream.relative_to(ROOT)} and {out.relative_to(ROOT)}: {end['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Record one seeded run of the synthetic gate scene with the real trackers.

    python scripts/web/export_gate_sim.py [--seed 7] [--seconds 30]

Writes
* web/public/offline/gate-sim.ndjson: the exact stream of /api/gate-sim/stream for the default
  sliders, so the page can replay it when the backend is offline (sliders need the backend);
* docs/results/synthetic_gate/seed<seed>.json: the run's end summary (ByteTrack, BoT-SORT, ours),
  read by the web app's all-datasets table.
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
    a = ap.parse_args()
    p = SimParams(seed=a.seed, seconds=a.seconds)
    stream = ROOT / "web" / "public" / "offline" / "gate-sim.ndjson"
    stream.parent.mkdir(parents=True, exist_ok=True)
    with stream.open("w") as fh:
        for x in simulate(p):
            fh.write(json.dumps(x, separators=(",", ":")) + "\n")
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

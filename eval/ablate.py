"""``make ablate``: run every entry of an ablation manifest and write one comparison table.

    python -m eval.ablate --manifest configs/ablations/manifest.yaml --out outputs/ablate \
        --report docs/results/ablation --jobs 6

Each entry runs ``eval.run`` (tracking → TrackEval → stress bins) into its own folder, so parallel
runs never share the TrackEval ``gt`` folder. The report holds:

* ``ablation.md`` / ``ablation.csv``: HOTA, IDF1, MOTA, IDSW, AssA, DetA, and compute latency
  p50 / p95 / max of ``update()``, measured afterwards by ``eval.latency``. That runs one entry at
  a time, with ``OMP_NUM_THREADS=1``, one warm-up and ``--latency-repeats`` timed passes.
* ``stress_visibility.png`` / ``stress_density.png``: recall, ID recall and IDSW per 1k gt boxes
  by gt-visibility bin, and IDF1 / IDSW per 1k gt by crowd-density bin, for every entry.

``--no-latency`` falls back to the ``update()`` times recorded during the parallel tracking runs,
which are noisier because other processes share the CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

COLS = ("HOTA", "IDF1", "MOTA", "IDSW", "AssA", "DetA")


def load_manifest(path: Path) -> dict[str, Any]:
    m: dict[str, Any] = yaml.safe_load(path.read_text())
    if not isinstance(m, dict) or "entries" not in m:
        raise ValueError(f"{path}: expected a mapping with 'entries'")
    names = [e["name"] for e in m["entries"]]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicate entry names")
    for e in m["entries"]:
        unknown = set(e) - {"name", "label", "group", "config", "overrides"}
        if unknown:
            raise ValueError(f"{path}: entry {e['name']}: unknown keys {sorted(unknown)}")
    return m


def run_entry(m: dict[str, Any], e: dict[str, Any], out: Path, force: bool,
              timeout_s: float | None = None) -> Path:  # fmt: skip
    name = str(e["name"])
    od = out / name
    if not force and (od / "stress" / name / "stress.json").exists():
        return od
    cmd = [sys.executable, "-m", "eval.run", "--dataset", m["dataset"], "--root", m["root"],
           "--split", m["split"], "--dets", m.get("dets", "public"), "--config", e["config"],
           "--name", name, "--out", str(od)]  # fmt: skip
    for o in e.get("overrides", []):
        cmd += ["-o", o]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if r.returncode:
        raise RuntimeError(f"{name} failed:\n{r.stderr[-2000:]}")
    return od


def latency_entry(m: dict[str, Any], e: dict[str, Any], out: Path, repeats: int,
                  force: bool) -> Path:  # fmt: skip
    path = out / str(e["name"]) / "latency.json"
    if path.exists() and not force:
        return path
    cmd = [sys.executable, "-m", "eval.latency", "--dataset", m["dataset"], "--root", m["root"],
           "--split", m["split"], "--dets", m.get("dets", "public"), "--config", e["config"],
           "--repeats", str(repeats), "--out", str(path)]  # fmt: skip
    for o in e.get("overrides", []):
        cmd += ["-o", o]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f"latency {e['name']} failed:\n{r.stderr[-2000:]}")
    return path


def collect(m: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    rows = []
    for e in m["entries"]:
        n = str(e["name"])
        od = out / n
        met = json.loads((od / "metrics" / n / "metrics.json").read_text())["COMBINED"]
        run = json.loads((od / "trackers" / n / "run.json").read_text())
        lat_file = od / "latency.json"
        lat_src = "eval.latency" if lat_file.exists() else "run.json"
        lat = (json.loads(lat_file.read_text()) if lat_file.exists() else run["latency"])[
            "combined"
        ]
        st = json.loads((od / "stress" / n / "stress.json").read_text())
        rows.append({"name": n, "label": e.get("label", n), "group": e.get("group", ""),
                     **{k: float(met[k]) for k in COLS},
                     "latency_source": lat_src, "p50_ms": lat["p50_ms"],
                     "p95_ms": lat["p95_ms"], "max_ms": lat["max_ms"],
                     "stress": st})  # fmt: skip
    return rows


def markdown(m: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    title = m.get("title") or f"{m['dataset']} {m['split']}, {m.get('dets', 'public')} detections"
    lines = [f"### Ablation: {title}", "",
             "| Group | Tracker | HOTA | IDF1 | MOTA | IDSW | AssA | DetA | p50 / p95 / max ms |",
             "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        lines.append(f"| {r['group']} | {r['label']} | {r['HOTA']:.1f} | {r['IDF1']:.1f} | "
                     f"{r['MOTA']:.1f} | {r['IDSW']:.0f} | {r['AssA']:.1f} | {r['DetA']:.1f} | "
                     f"{r['p50_ms']:.2f} / {r['p95_ms']:.2f} / {r['max_ms']:.1f} |")  # fmt: skip
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = ["group", "name", "label", *COLS, "p50_ms", "p95_ms", "max_ms"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_stress(rows: list[dict[str, Any]], out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    specs = [("visibility", [("recall", "recall"), ("id_recall", "ID recall"),
                             ("idsw_per_1k_gt", "IDSW per 1k gt")], "gt visibility bin"),
             ("density", [("idf1", "IDF1"), ("recall", "recall"),
                          ("idsw_per_1k_gt", "IDSW per 1k gt")], "gt boxes per frame")]  # fmt: skip
    for key, metrics, xlabel in specs:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        cmap = plt.get_cmap("tab20")
        for i, r in enumerate(rows):
            # "=0 (subset)" style bins overlap the others; they stay in the JSON, not on the axis
            bins = [b for b in r["stress"][key] if not str(b["bin"]).startswith("=")]
            xs = [b["bin"] for b in bins]
            for ax, (mk, _) in zip(axes, metrics, strict=True):
                ax.plot(xs, [b[mk] for b in bins], marker="o", label=r["label"], color=cmap(i))
        for ax, (_, ml) in zip(axes, metrics, strict=True):
            ax.set(xlabel=xlabel, ylabel=ml)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        p = out / f"stress_{key}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def run_manifest(m: dict[str, Any], out: Path, report: Path, jobs: int = 6, force: bool = False,
                 latency: bool = True, latency_repeats: int = 1) -> str:  # fmt: skip
    """Run every entry, measure latency, write the report folder; return the markdown table."""
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(lambda e: run_entry(m, e, out, force), m["entries"]))
    if latency:
        for e in m["entries"]:  # sequential: nothing else competes for the CPU
            latency_entry(m, e, out, latency_repeats, force)
    rows = collect(m, out)
    report.mkdir(parents=True, exist_ok=True)
    md = markdown(m, rows)
    (report / "ablation.md").write_text(md + "\n")
    write_csv(rows, report / "ablation.csv")
    (report / "ablation.json").write_text(json.dumps(rows, indent=1))
    plot_stress(rows, report)
    return md


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.ablate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--manifest", type=Path, default=Path("configs/ablations/manifest.yaml"))
    p.add_argument("--out", type=Path, default=Path("outputs/ablate"))
    p.add_argument("--report", type=Path, default=Path("docs/results/ablation"))
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--force", action="store_true", help="re-run entries that already have output")
    p.add_argument("--no-latency", action="store_true")
    p.add_argument("--latency-repeats", type=int, default=1)
    a = p.parse_args(argv)
    m = load_manifest(a.manifest)
    md = run_manifest(m, a.out, a.report, a.jobs, a.force, not a.no_latency, a.latency_repeats)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

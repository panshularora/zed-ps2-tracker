"""Run metadata for ``run.sh all``: which GPU, detector, precision, input size and profile.

Every results file of a ``run.sh all`` run carries this record, so a table can never be read
without knowing which detector produced it (``scripts/dgx/run_all.sh`` calls ``stamp`` after
``make phase3``; the final bundle manifest embeds the same record).

    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \\
        --format=csv,noheader,nounits | python -m eval.run_info gpu > gpu.json
    python -m eval.run_info stamp --report REPORT --index STAGEA/index.json \\
        --profile laptop --gpu-json gpu.json --out run_info.json

Stamping is idempotent: Markdown files get one ``<!-- run-info -->`` header block (replaced on
re-run), JSON objects a ``run_info`` key (JSON lists: one per row), CSV files six ``run_*``
columns.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

MARK_BEGIN = "<!-- run-info -->"
MARK_END = "<!-- /run-info -->"
CSV_COLS = ("run_profile", "run_gpu", "run_gpu_mem_mib", "run_detector", "run_precision",
            "run_input_size")  # fmt: skip


def _num(x: str) -> float | None:
    m = re.search(r"[\d.]+", x)
    try:
        return float(m.group()) if m else None
    except ValueError:
        return None


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap
    --format=csv,noheader[,nounits]``. Robust to a header line, units (``6144 MiB``),
    ``[N/A]`` / ``[Not Supported]`` fields, missing trailing fields and blank lines."""
    gpus: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("name"):
            continue
        parts = [p.strip() for p in line.split(",")]
        name = parts[0]
        mem = _num(parts[1]) if len(parts) > 1 else None
        if mem is not None and len(parts) > 1 and "gib" in parts[1].lower():
            mem *= 1024
        driver = parts[2] if len(parts) > 2 and not parts[2].startswith("[") else None
        cap = parts[3] if len(parts) > 3 and not parts[3].startswith("[") else None
        gpus.append({"index": len(gpus), "name": name,
                     "memory_mib": int(mem) if mem is not None else None,
                     "driver": driver, "compute_cap": cap})  # fmt: skip
    return gpus


def detector_of(cache_dir: Path) -> dict[str, Any]:
    """Detector record of one cache folder, read from the first cache's own metadata."""
    from tracker.io import load_cache  # lazy: `gpu` works with a bare python3 (no numpy)

    first = next(iter(sorted(Path(cache_dir).glob("*.npz"))), None)
    if first is None:
        return {}
    m = load_cache(first).meta
    d = m.detector
    ts = d.get("test_size")
    return {
        "detector": d.get("name"),
        "weights": d.get("weights"),
        "trained_on": d.get("trained_on"),
        "precision": "fp16" if d.get("fp16") else "fp32",
        "input_size": f"{ts[0]}x{ts[1]}" if isinstance(ts, list | tuple) and len(ts) == 2 else None,
        "saw_eval_half": m.detector_saw_eval_half,
        "gpu_in_cache": m.extra.get("gpu"),
        "reid": (m.reid or {}).get("name"),
    }


def collect(
    index: dict[str, Any], profile: str, gpu: dict[str, Any] | None,
    host: str | None = None, tagged: list[str] | None = None,
) -> dict[str, Any]:  # fmt: skip
    caches = {k: detector_of(Path(g["dir"])) for k, g in sorted(index.get("caches", {}).items())}
    return {"profile": profile, "host": host, "gpu": gpu or {}, "detectors": caches,
            "laptop_detector_caches": sorted(tagged or [])}  # fmt: skip


def _summary_line(info: dict[str, Any], key: str | None) -> dict[str, str]:
    g = info.get("gpu") or {}
    d = info["detectors"].get(key, {}) if key else {}
    if not d and key is None:
        dets = sorted({str(v.get("weights")) for v in info["detectors"].values()})
        d = {"detector": "; ".join(dets) if dets else None}
    return {
        "run_profile": str(info.get("profile")),
        "run_gpu": str(g.get("name")),
        "run_gpu_mem_mib": str(g.get("memory_mib")),
        "run_detector": f"{d.get('detector')} [{d.get('weights')}]"
        if "weights" in d
        else str(d.get("detector")),
        "run_precision": str(d.get("precision", "see detectors")),
        "run_input_size": str(d.get("input_size", "see detectors")),
    }


def _md_block(info: dict[str, Any], key: str | None) -> str:
    s = _summary_line(info, key)
    lines = [MARK_BEGIN,
             f"> Run: profile **{s['run_profile']}**, GPU {s['run_gpu']} "
             f"({s['run_gpu_mem_mib']} MiB); detector {s['run_detector']}, "
             f"{s['run_precision']}, input {s['run_input_size']}"
             + (f"; cache `{key}`" if key else "") + ".",
             MARK_END, ""]  # fmt: skip
    return "\n".join(lines)


def _stamp_md(path: Path, info: dict[str, Any], key: str | None) -> None:
    text = path.read_text()
    text = re.sub(re.escape(MARK_BEGIN) + r".*?" + re.escape(MARK_END) + r"\n\n?", "", text,
                  flags=re.S)  # fmt: skip
    path.write_text(_md_block(info, key) + "\n" + text)


def _stamp_json(path: Path, info: dict[str, Any], key: str | None) -> None:
    data = json.loads(path.read_text())
    rec = {**_summary_line(info, key), "cache": key}
    if isinstance(data, dict):
        data["run_info"] = rec
    elif isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                row["run_info"] = rec
    else:
        return
    path.write_text(json.dumps(data, indent=1))


def _stamp_csv(path: Path, info: dict[str, Any], key: str | None) -> None:
    rows = list(csv.reader(io.StringIO(path.read_text())))
    if not rows:
        return
    head = rows[0]
    keep = [i for i, h in enumerate(head) if h not in CSV_COLS]
    s = _summary_line(info, key)
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow([head[i] for i in keep] + list(CSV_COLS))
    for r in rows[1:]:
        w.writerow([r[i] for i in keep if i < len(r)] + [s[c] for c in CSV_COLS])
    path.write_text(out.getvalue())


def stamp_report(report: Path, info: dict[str, Any]) -> list[Path]:
    """Stamp every .md / .json / .csv under ``report``. A sub-folder named after a phase-3
    condition id is attributed to that condition's cache (from ``summary.json``)."""
    sj = report / "summary.json"
    summary = json.loads(sj.read_text()) if sj.exists() else {}
    by_id = {r["id"]: r["cache"] for r in summary.get("ran", [])}
    top_key = summary.get("tune_cache")
    done = []
    for p in sorted(report.rglob("*")):
        if not p.is_file() or p.suffix not in (".md", ".json", ".csv"):
            continue
        rel = p.relative_to(report)
        key = by_id.get(rel.parts[0]) if len(rel.parts) > 1 else None
        if key is None and rel.parts[0] in ("go_no_go_v2", "tune.json", "reproduction.md",
                                            "reproduction.json", "go_no_go_v2.md",
                                            "go_no_go_v2.json", "gate_events"):  # fmt: skip
            key = top_key
        {".md": _stamp_md, ".json": _stamp_json, ".csv": _stamp_csv}[p.suffix](p, info, key)
        done.append(p)
    (report / "run_info.json").write_text(json.dumps(info, indent=1))
    return done


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.run_info", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gpu", help="nvidia-smi csv on stdin -> JSON of GPU --index")
    g.add_argument("--index", type=int, default=0)
    s = sub.add_parser("stamp")
    s.add_argument("--report", type=Path, required=True)
    s.add_argument("--index", type=Path, required=True, help="stagea index.json")
    s.add_argument("--profile", required=True)
    s.add_argument("--gpu-json", type=Path, default=None)
    s.add_argument("--host", default=None)
    s.add_argument("--tagged", nargs="*", default=[], help="laptop-detector cache keys")
    s.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    if a.cmd == "gpu":
        gpus = parse_nvidia_smi(sys.stdin.read())
        if not gpus:
            print("no GPU in nvidia-smi output", file=sys.stderr)
            return 1
        pick = next((x for x in gpus if x["index"] == a.index), gpus[0])
        print(json.dumps(pick))
        return 0
    gpu = json.loads(a.gpu_json.read_text()) if a.gpu_json and a.gpu_json.exists() else None
    info = collect(json.loads(a.index.read_text()), a.profile, gpu, a.host, a.tagged)
    files = stamp_report(a.report, info)
    if a.out:
        a.out.write_text(json.dumps(info, indent=1))
    print(f"stamped {len(files)} files under {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

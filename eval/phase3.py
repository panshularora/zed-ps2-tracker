"""``make phase3``: every post-GPU experiment from the ingested Stage A caches, in one command.

    python -m eval.phase3 --index data/stagea/index.json --plan configs/phase3/plan.yaml \
        --out outputs/phase3 --report docs/results/phase3

The plan (``configs/phase3/plan.yaml``) was frozen before any GPU data existed. Steps:

1. **Tune** ``tau_update`` (``appearance.min_visibility``) and ``tau_cost``
   (``appearance.cost_min_visibility``) on MOT17 *train_half* only, one at a time, with 0 (off) as
   a candidate. The chosen values are written to ``tune.json`` and used unchanged everywhere else.
2. **Conditions**: MOT17 val-half (with single-threaded latency), MOT20 val-half once per
   ingested detector variant (``SAW_EVAL_HALF`` variants in their own labelled table), and
   DanceTrack val. Each is an ``eval.ablate`` manifest: tracking → TrackEval → stress bins →
   ``ablation.{md,csv,json}`` + ``stress_{visibility,density}.png``.
3. **Reproduction check**: the IoU-only ByteTrackLike row on MOT17 val-half against ByteTrack's
   own half-val numbers (MOTA 76.6 / IDF1 79.3 / IDs 159; arXiv 2110.06864v3 Table 1, IoU|IoU row,
   and Table 11, no interpolation) with the tolerance fixed in the plan.

Conditions whose caches are not ingested yet are listed as waiting, so the command can be re-run
after each ``make ingest``; finished entries are skipped (``--force`` re-runs them).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any

import yaml

from eval import ablate


def load_plan(path: Path) -> dict[str, Any]:
    plan: dict[str, Any] = yaml.safe_load(path.read_text())
    for c in plan["conditions"]:
        for e in c["entries"]:
            unknown = set(e) - {"name", "label", "group", "config", "overrides"}
            if unknown:
                raise ValueError(f"{c['id']}/{e['name']}: unknown keys {sorted(unknown)}")
    return plan


def substitute(entries: list[dict[str, Any]], params: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for e in entries:
        ov = [str(o).format(**params) for o in e.get("overrides", [])]
        out.append({**e, "overrides": ov})
    return out


def _manifest(index: dict[str, Any], key: str, dataset: str, split: str,
              entries: list[dict[str, Any]], title: str) -> dict[str, Any]:  # fmt: skip
    g = index["caches"][key]
    if not g.get("gt_root"):
        raise FileNotFoundError(f"{key}: no gt labels ingested (ingest core.tar)")
    return {"dataset": dataset, "root": g["gt_root"], "split": split, "dets": g["dir"],
            "entries": entries, "title": title}  # fmt: skip


def tune(plan: dict[str, Any], index: dict[str, Any], out: Path, jobs: int,
         force: bool) -> dict[str, Any]:  # fmt: skip
    t = plan["tune"]
    if t["cache"] not in index["caches"]:
        raise FileNotFoundError(f"tuning needs {t['cache']} (ingest MOT17 first)")
    chosen: dict[str, float] = {}
    table: dict[str, list[dict[str, Any]]] = {}
    for pname, spec in t["params"].items():
        entries = []
        for v in spec["values"]:
            entries.append({"name": f"tune_{pname}_{v}", "label": f"{spec['key']}={v}",
                            "group": "tune", "config": t["config"],
                            "overrides": [*t["base_overrides"], f"{spec['key']}={v}"]})  # fmt: skip
        m = _manifest(index, t["cache"], t["dataset"], t["split"], entries,
                      f"tune {pname} on {t['dataset']} {t['split']}")  # fmt: skip
        ablate.run_manifest(m, out / "tune", out / "tune" / f"report_{pname}", jobs, force,
                            latency=False)  # fmt: skip
        rows = ablate.collect(m, out / "tune")
        crit = t["criterion"]
        best = max(range(len(rows)), key=lambda i: tuple(rows[i][c] for c in crit))
        chosen[pname] = float(spec["values"][best])
        table[pname] = [{"value": v, **{c: r[c] for c in ("HOTA", "IDF1", "IDSW")}}
                        for v, r in zip(spec["values"], rows, strict=True)]  # fmt: skip
    res = {"split": t["split"], "criterion": t["criterion"], "chosen": chosen, "sweeps": table}
    return res


def reproduction(plan: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path,
                 metrics_dir: Path) -> dict[str, Any]:  # fmt: skip
    r = plan["reproduction"]
    met = json.loads((metrics_dir / r["entry"] / "metrics" / r["entry"] / "metrics.json")
                     .read_text())["COMBINED"]  # fmt: skip
    ref, tol = r["reference"], r["tolerance"]
    ours = {k: float(met[k]) for k in ref}
    ok = {k: abs(ours[k] - ref[k]) <= v for k, v in tol.items()}
    verdict = "REPRODUCED" if all(ok.values()) else "NOT REPRODUCED"
    lines = ["# ByteTrack reproduction check (MOT17 val-half, YOLOX-X bytetrack_ablation)", "",
             f"Reference: {r['source']}. Tolerance fixed in `configs/phase3/plan.yaml` before "
             "any GPU data existed.", "",
             "| Metric | ByteTrack paper | ours (ByteTrackLike, IoU) | Δ | tolerance |",
             "|---|---|---|---|---|"]  # fmt: skip
    for k in ref:
        t = f"±{tol[k]}" if k in tol else "(reported only)"
        lines.append(f"| {k} | {ref[k]} | {ours[k]:.1f} | {ours[k] - ref[k]:+.1f} | {t} |")
    lines += ["", f"**Verdict: {verdict}.**", "",
              "Known differences that can move the numbers: xywh Kalman filter instead of xyah; "
              "TrackEval CLEAR instead of py-motmetrics; the common output filter (area > 100, "
              "w/h ≤ 1.6, as ByteTrack's evaluator); no per-sequence thresholds (ByteTrack's "
              "val-half run has none either)."]  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reproduction.md").write_text("\n".join(lines) + "\n")
    res = {"reference": ref, "ours": ours, "within_tolerance": ok, "verdict": verdict}
    (out_dir / "reproduction.json").write_text(json.dumps(res, indent=1))
    return res


def run(
    plan_path: Path, index_path: Path, out: Path, report: Path, jobs: int = 6,
    force: bool = False, latency_repeats: int = 1, latency: bool = True,
) -> dict[str, Any]:  # fmt: skip
    plan = load_plan(plan_path)
    index = json.loads(index_path.read_text())
    keys = sorted(index["caches"])
    summary: dict[str, Any] = {"plan": str(plan_path), "ran": [], "waiting": []}
    report.mkdir(parents=True, exist_ok=True)
    try:
        tuned = tune(plan, index, out, jobs, force)
    except FileNotFoundError as e:
        summary["waiting"].append(f"tune: {e}")
        (report / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"waiting: {e}")
        return summary
    (report / "tune.json").write_text(json.dumps(tuned, indent=1))
    params = tuned["chosen"]
    summary["tuned"] = params
    for c in plan["conditions"]:
        hits = [k for k in keys if fnmatch.fnmatch(k, c["cache"])]
        if not hits:
            summary["waiting"].append(f"{c['id']}: no cache matching {c['cache']}")
            continue
        for key in hits:
            g = index["caches"][key]
            variant = key.split("/")[1]
            cid = c["id"] if len(hits) == 1 and "*" not in c["cache"] else f"{c['id']}__{variant}"
            title = c["title"].format(variant=variant)
            if g["saw_eval_half"]:
                title += " (DETECTOR SAW THE EVAL HALF: not a clean result)"
            if not g["has_embeddings"]:
                summary["waiting"].append(f"{cid}: {key} has no embeddings")
                continue
            entries = substitute(c["entries"], params)
            try:
                m = _manifest(index, key, c["dataset"], c["split"], entries, title)
            except FileNotFoundError as e:
                summary["waiting"].append(f"{cid}: {e}")
                continue
            md = ablate.run_manifest(m, out / cid, report / cid, jobs, force,
                                     latency=latency and bool(c.get("latency")),
                                     latency_repeats=latency_repeats)  # fmt: skip
            print(md + "\n")
            summary["ran"].append({"id": cid, "cache": key, "saw_eval_half": g["saw_eval_half"]})
            if c["id"] == "mot17_val_half" and key == plan["reproduction"]["cache"]:
                rows = ablate.collect(m, out / cid)
                summary["reproduction"] = reproduction(plan, rows, report, out / cid)
    (report / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["# Phase 3 results index", "", f"Tuned on MOT17 train_half: `{params}`", ""]
    for r in summary["ran"]:
        tag = " **(SAW_EVAL_HALF)**" if r["saw_eval_half"] else ""
        lines.append(f"* [{r['id']}]({r['id']}/ablation.md){tag}")
    if "reproduction" in summary:
        lines.append(f"* [ByteTrack reproduction](reproduction.md): "
                     f"{summary['reproduction']['verdict']}")  # fmt: skip
    for w in summary["waiting"]:
        lines.append(f"* waiting: {w}")
    (report / "README.md").write_text("\n".join(lines) + "\n")
    for w in summary["waiting"]:
        print(f"waiting: {w}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.phase3", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--plan", type=Path, default=Path("configs/phase3/plan.yaml"))
    p.add_argument("--index", type=Path, default=Path("data/stagea/index.json"))
    p.add_argument("--out", type=Path, default=Path("outputs/phase3"))
    p.add_argument("--report", type=Path, default=Path("docs/results/phase3"))
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-latency", action="store_true")
    p.add_argument("--latency-repeats", type=int, default=1)
    a = p.parse_args(argv)
    if not a.index.exists():
        print(f"{a.index} not found: run `make ingest BRING_BACK=<path>` first")
        return 1
    run(a.plan, a.index, a.out, a.report, a.jobs, a.force, a.latency_repeats, not a.no_latency)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

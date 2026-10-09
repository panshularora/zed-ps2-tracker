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

**Laptop detector** (``run.sh all --profile laptop`` after a CUDA out-of-memory fallback, or an
explicit ``--detector s``): ``--remap-cache MOT17/ablation/train=MOT17/laptop_.../train`` points
the plan's fixed cache keys at the laptop detector's caches, and ``--tag-cache PATTERN`` marks
caches as laptop-detector output. Every row (baselines included) and title of such a condition
gets ``--tag`` appended, the reproduction check is written as PENDING (the paper numbers belong
to YOLOX-X ``bytetrack_ablation``), and the pre-registered go/no-go v2 is PENDING when its
primary cache is a laptop-detector one (the rule names the leak-free YOLOX-X cache). The plan
file itself is never edited; the remap and tag are recorded in ``summary.json``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from eval import ablate

LAPTOP_TAG = "laptop detector, not comparable to published numbers"


def apply_remap(plan: dict[str, Any], remap: dict[str, str]) -> dict[str, Any]:
    """Copy of ``plan`` with fixed cache keys replaced (``tune``, ``reproduction``,
    ``conditions[].cache``, ``go_no_go_v2.primary``). Wildcard keys and the go/no-go
    replication caches are left alone: the pre-registration names those exactly."""
    if not remap:
        return plan
    p: dict[str, Any] = yaml.safe_load(yaml.safe_dump(plan))
    p["tune"]["cache"] = remap.get(p["tune"]["cache"], p["tune"]["cache"])
    p["reproduction"]["cache"] = remap.get(p["reproduction"]["cache"], p["reproduction"]["cache"])
    for c in p["conditions"]:
        c["cache"] = remap.get(c["cache"], c["cache"])
    g = p.get("go_no_go_v2")
    if g:
        g["primary"]["cache"] = remap.get(g["primary"]["cache"], g["primary"]["cache"])
    return p


def is_tagged(key: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(key, pat) for pat in patterns)


def tag_entries(entries: list[dict[str, Any]], tag: str) -> list[dict[str, Any]]:
    return [{**e, "label": f"{e['label']} [{tag}]"} for e in entries]


def pending_reproduction(plan: dict[str, Any], key: str, tag: str, out_dir: Path) -> dict[str, Any]:
    r = plan["reproduction"]
    lines = ["# ByteTrack reproduction check: PENDING", "",
             f"This run's MOT17 cache `{key}` comes from the {tag}. The reference numbers "
             f"({r['source']}) belong to YOLOX-X `bytetrack_ablation`, so comparing them would "
             "be meaningless. The check runs when the YOLOX-X `MOT17/ablation/train` cache "
             "exists (the DGX run, or the laptop without the out-of-memory fallback)."]  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reproduction.md").write_text("\n".join(lines) + "\n")
    res = {"verdict": "PENDING", "reason": f"{key}: {tag}", "reference": r["reference"]}
    (out_dir / "reproduction.json").write_text(json.dumps(res, indent=1))
    return res


def pending_go_no_go(key: str, tag: str, report: Path) -> dict[str, Any]:
    lines = ["# Go/no-go v2: PENDING", "",
             f"The pre-registered rule (`docs/preregistration/go_no_go_v2.md`) is defined on the "
             f"leak-free YOLOX-X `MOT17/ablation/train` cache. This run's MOT17 cache `{key}` "
             f"comes from the {tag}, so no verdict is computed. Run on a GPU that fits YOLOX-X "
             "(fp16, batch 1 fits most 6 GB cards) to get it."]  # fmt: skip
    report.mkdir(parents=True, exist_ok=True)
    (report / "go_no_go_v2.md").write_text("\n".join(lines) + "\n")
    res = {"verdict": "PENDING", "reason": f"{key}: {tag}"}
    (report / "go_no_go_v2.json").write_text(json.dumps(res, indent=1))
    return res


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


def _flag_rate(run_dir: Path, name: str) -> float:
    run = json.loads((run_dir / "trackers" / name / "run.json").read_text())
    frames = flagged = 0
    for seq in run["latency"]["per_sequence"].values():
        d = seq["tracker_info"].get("delayed")
        if d:
            frames += d["frames"]
            flagged += d["ambiguous_frames"]
    return flagged / frames if frames else 0.0


def go_no_go_v2(plan: dict[str, Any], index: dict[str, Any], out: Path, report: Path,
                jobs: int, force: bool, latency: bool) -> dict[str, Any] | None:  # fmt: skip
    """Run the pre-registered v2 check and apply its decision rule mechanically."""
    g = plan.get("go_no_go_v2")
    if not g or g["primary"]["cache"] not in index["caches"]:
        return None
    p = g["primary"]
    entries = [{"name": f"v2_{arm}", "label": arm, "group": "sanity check", "config": g["config"],
                "overrides": ov} for arm, ov in g["arms"].items()]  # fmt: skip
    title = "go/no-go v2 (sanity check): delayed decision with re-ID, MOT17 val-half"
    m = _manifest(index, p["cache"], p["dataset"], p["split"], entries, title)
    od = out / "go_no_go_v2"
    ablate.run_manifest(m, od, report / "go_no_go_v2", jobs, force, latency=latency)
    rows = {r["name"].removeprefix("v2_"): r for r in ablate.collect(m, od)}
    for arm in rows:
        rows[arm]["flag_rate"] = _flag_rate(od / f"v2_{arm}", f"v2_{arm}")
    rule, k0, k5 = g["rule"], rows["k0"], rows["k5"]
    crit = {
        "1_idsw": k5["IDSW"] <= rule["idsw_ratio"] * k0["IDSW"],
        "2_identity": (k5["IDF1"] >= k0["IDF1"] + rule["idf1_gain"]
                       and k5["HOTA"] >= k0["HOTA"] - rule["hota_drop"]),
        "3_selectivity": k5["flag_rate"] <= rule["max_flag_rate"],
        "4_compute": k5["p95_ms"] <= rule["max_p95_ms"],
    }  # fmt: skip
    rep: dict[str, Any] = {"status": "not evaluated (no clean MOT20 cache ingested)"}
    r = g["replication"]
    key = next((k for k in r["caches"] if k in index["caches"]), None)
    if key is not None:
        rm = _manifest(index, key, r["dataset"], r["split"],
                       [e for e in entries if e["name"] in ("v2_k0", "v2_k5")],
                       "go/no-go v2 replication")  # fmt: skip
        rod = out / "go_no_go_v2_mot20"
        try:
            for e in rm["entries"]:
                ablate.run_entry(rm, e, rod, force, timeout_s=60.0 * r["timeout_min"])
            rr = {x["name"]: x for x in ablate.collect(rm, rod)}
            ok = rr["v2_k5"]["IDSW"] <= rr["v2_k0"]["IDSW"]
            rep = {"status": "pass" if ok else "fail", "cache": key,
                   "idsw_k0": rr["v2_k0"]["IDSW"], "idsw_k5": rr["v2_k5"]["IDSW"]}  # fmt: skip
        except subprocess.TimeoutExpired:
            rep = {"status": "fail (timeout)", "cache": key}
        crit["5_replication"] = rep["status"] == "pass"
    passed = all(crit.values())
    if passed and key is None:
        verdict = "GO (MOT17 only; replication not evaluated)"
    else:
        verdict = "GO" if passed else "NO-GO"
    slim = {a: {k: v for k, v in x.items() if k != "stress"} for a, x in rows.items()}
    res = {"verdict": verdict, "criteria": crit, "replication": rep, "rows": slim}
    lines = ["# Go/no-go v2 (sanity check, not a results table)", "",
             "Pre-registered in `docs/preregistration/go_no_go_v2.md`; verdict computed by "
             "`eval/phase3.py` from the frozen rule.", "",
             "| Arm | HOTA | IDF1 | IDSW | flagged frames | p95 ms |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for a, x in rows.items():
        lines.append(f"| {a} | {x['HOTA']:.1f} | {x['IDF1']:.1f} | {x['IDSW']:.0f} | "
                     f"{100 * x['flag_rate']:.1f} % | {x['p95_ms']:.2f} |")  # fmt: skip
    lines += ["", "| Criterion | pass |", "|---|---|"]
    lines += [f"| {k} | {'yes' if v else 'no'} |" for k, v in crit.items()]
    lines += ["", f"MOT20 replication: {rep}", "", f"**Verdict: {verdict}**"]
    report.mkdir(parents=True, exist_ok=True)
    (report / "go_no_go_v2.md").write_text("\n".join(lines) + "\n")
    (report / "go_no_go_v2.json").write_text(json.dumps(res, indent=1))
    return res


def run(
    plan_path: Path, index_path: Path, out: Path, report: Path, jobs: int = 6,
    force: bool = False, latency_repeats: int = 1, latency: bool = True,
    remap: dict[str, str] | None = None, tag_caches: list[str] | None = None,
    tag: str = LAPTOP_TAG,
) -> dict[str, Any]:  # fmt: skip
    remap = dict(remap or {})
    tag_patterns = [*(tag_caches or []), *remap.values()]
    plan = apply_remap(load_plan(plan_path), remap)
    index = json.loads(index_path.read_text())
    keys = sorted(index["caches"])
    summary: dict[str, Any] = {"plan": str(plan_path), "ran": [], "waiting": [],
                               "tune_cache": plan["tune"]["cache"]}  # fmt: skip
    if remap or tag_caches:
        summary["deviation"] = {"remap": remap, "tag_caches": tag_caches or [], "tag": tag}
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
            tagged = is_tagged(key, tag_patterns)
            if tagged:
                title += f" [{tag}: {key}]"
            if not g["has_embeddings"]:
                summary["waiting"].append(f"{cid}: {key} has no embeddings")
                continue
            entries = substitute(c["entries"], params)
            if tagged:
                entries = tag_entries(entries, tag)
            try:
                m = _manifest(index, key, c["dataset"], c["split"], entries, title)
            except FileNotFoundError as e:
                summary["waiting"].append(f"{cid}: {e}")
                continue
            md = ablate.run_manifest(m, out / cid, report / cid, jobs, force,
                                     latency=latency and bool(c.get("latency")),
                                     latency_repeats=latency_repeats)  # fmt: skip
            print(md + "\n")
            summary["ran"].append({"id": cid, "cache": key, "saw_eval_half": g["saw_eval_half"],
                                   "laptop_detector": tagged})  # fmt: skip
            if c["id"] == "mot17_val_half" and key == plan["reproduction"]["cache"]:
                if tagged:
                    summary["reproduction"] = pending_reproduction(plan, key, tag, report)
                else:
                    rows = ablate.collect(m, out / cid)
                    summary["reproduction"] = reproduction(plan, rows, report, out / cid)
    primary = plan.get("go_no_go_v2", {}).get("primary", {}).get("cache")
    if primary in index["caches"] and is_tagged(primary, tag_patterns):
        gng: dict[str, Any] | None = pending_go_no_go(primary, tag, report)
    else:
        gng = go_no_go_v2(plan, index, out, report, jobs, force, latency)
    if gng is None:
        summary["waiting"].append("go_no_go_v2: needs MOT17/ablation/train")
    else:
        summary["go_no_go_v2"] = gng["verdict"]
    (report / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["# Phase 3 results index", "", f"Tuned on MOT17 train_half: `{params}`", ""]
    if "deviation" in summary:
        lines += [
            f"**Rows marked [{tag}] are not comparable to published numbers.** "
            f"Plan deviation (recorded, plan file unchanged): `{summary['deviation']}`",
            "",
        ]
    for r in summary["ran"]:
        mark = " **(SAW_EVAL_HALF)**" if r["saw_eval_half"] else ""
        mark += f" **[{tag}]**" if r.get("laptop_detector") else ""
        lines.append(f"* [{r['id']}]({r['id']}/ablation.md){mark}")
    if "reproduction" in summary:
        lines.append(f"* [ByteTrack reproduction](reproduction.md): "
                     f"{summary['reproduction']['verdict']}")  # fmt: skip
    if "go_no_go_v2" in summary:
        lines.append(f"* [go/no-go v2](go_no_go_v2.md) (sanity check): {summary['go_no_go_v2']}")
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
    p.add_argument(
        "--remap-cache",
        action="append",
        default=[],
        metavar="SRC=DST",
        help="use cache DST wherever the plan names SRC (laptop detector)",
    )
    p.add_argument(
        "--tag-cache",
        action="append",
        default=[],
        metavar="PATTERN",
        help="caches (fnmatch) whose rows get --tag appended",
    )
    p.add_argument("--tag", default=LAPTOP_TAG)
    a = p.parse_args(argv)
    if not a.index.exists():
        print(f"{a.index} not found: run `make ingest BRING_BACK=<path>` first")
        return 1
    remap = {}
    for r in a.remap_cache:
        src, sep, dst = r.partition("=")
        if not sep or not src or not dst:
            p.error(f"--remap-cache wants SRC=DST, got {r!r}")
        remap[src] = dst
    run(a.plan, a.index, a.out, a.report, a.jobs, a.force, a.latency_repeats, not a.no_latency,
        remap=remap, tag_caches=a.tag_cache, tag=a.tag)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

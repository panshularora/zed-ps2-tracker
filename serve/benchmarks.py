"""The official benchmark results, read from the repo's own outputs in ``docs/results``.

Nothing here is computed or typed in by hand: every number is loaded from a file written by
``make ablate`` / ``make gates`` / ``eval.visibility_eval`` / ``eval.go_no_go``, and the response
names that file. Results that need the GPU session (``make phase3``) are reported as ``pending``
until ``docs/results/phase3/`` exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HEADLINE = "ours_vis_botsort"
PLOTS = {
    "stress_visibility": "ablation/stress_visibility.png",
    "stress_density": "ablation/stress_density.png",
    "go_no_go_curve": "go_no_go/curve.png",
}
SETTING = (
    "MOT17 val-half (second half of each of the 7 MOT17 train sequences), public FRCNN "
    "detections, IoU/motion only (no re-ID), TrackEval HOTA/CLEAR/Identity"
)

# what the GPU session adds (docs/next_gpu_session.md); shown as "pending DGX run"
PENDING_GPU = [
    {
        "id": "yolox_rows",
        "what": "MOT17 val-half ablation on YOLOX-X detections, with and without re-ID",
        "source": "docs/results/phase3/<id>/ablation.md",
    },
    {
        "id": "mot20",
        "what": "MOT20 val-half (dense crowds) ablation",
        "source": "docs/results/phase3/<id>/",
    },
    {"id": "dancetrack", "what": "DanceTrack val ablation", "source": "docs/results/phase3/<id>/"},
    {
        "id": "reproduction",
        "what": "ByteTrack reproduction check (paper: MOTA 76.6 / IDF1 79.3)",
        "source": "docs/results/phase3/<id>/reproduction.md",
    },
    {
        "id": "go_no_go_v2",
        "what": "Pre-registered go/no-go v2 (re-ID, visibility-gated appearance)",
        "source": "docs/results/phase3/<id>/go_no_go_v2.md",
    },
    {
        "id": "detector_latency",
        "what": "Detector latency on GPU (YOLOX-X fp16)",
        "source": "Stage A log (stage_a_log.jsonl)",
    },
    {
        "id": "gate_events_gpu",
        "what": "Gate events (attribution, tailgate P/R, count error) on GPU detections",
        "source": "eval.gate_events on outputs/phase3",
    },
]


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _round(x: Any, nd: int = 1) -> Any:
    return round(float(x), nd) if isinstance(x, int | float) else x


def ablation(root: Path) -> dict[str, Any] | None:
    rows = _load(root / "ablation" / "ablation.json")
    if rows is None:
        return None
    out = []
    for r in rows:
        st = r.get("stress", {})
        out.append(
            {
                "name": r["name"],
                "label": r["label"],
                "group": r["group"],
                "headline": r["name"] == HEADLINE,
                **{k: _round(r[k]) for k in ("HOTA", "IDF1", "MOTA", "AssA", "DetA")},
                "IDSW": int(r["IDSW"]),
                "p50_ms": _round(r.get("p50_ms"), 2),
                "p95_ms": _round(r.get("p95_ms"), 2),
                "max_ms": _round(r.get("max_ms"), 1),
                "stress_visibility": [
                    {
                        "bin": b["bin"],
                        "gt": b["gt"],
                        "recall": _round(b["recall"], 4),
                        "id_recall": _round(b["id_recall"], 4),
                        "idsw_per_1k_gt": _round(b["idsw_per_1k_gt"], 2),
                    }
                    for b in st.get("visibility", [])
                ],
                "stress_density": [
                    {
                        "bin": b["bin"],
                        "frames": b["frames"],
                        "gt": b["gt"],
                        "idf1": _round(b["idf1"], 1),
                        "mota": _round(b["mota"], 1),
                        "idsw_per_1k_gt": _round(b["idsw_per_1k_gt"], 2),
                    }
                    for b in st.get("density", [])
                ],
            }
        )
    return {
        "setting": SETTING,
        "source": "docs/results/ablation/ablation.json",
        "reproduce": "make ablate",
        "rows": out,
        "sequences": rows[0].get("stress", {}).get("sequences", []),
    }


def gate_events(root: Path) -> dict[str, Any] | None:
    g = _load(root / "gate_events" / "gate_events.json")
    if g is None:
        return None
    keys = (
        "gt_crossings",
        "trk_crossings",
        "matched",
        "count_error",
        "precision",
        "recall",
        "attribution_error",
        "gt_tailgates",
        "trk_tailgates",
        "tailgate_precision",
        "tailgate_recall",
    )
    rows = [
        {"label": k, "check": "check" in k, **{f: v.get(f) for f in keys}} for k, v in g.items()
    ]
    return {
        "source": "docs/results/gate_events/gate_events.json",
        "reproduce": "make gates",
        "note": "MOT17-02/04/09 val-half: 25 gt crossings and 9 tailgates, too few to rank "
        "trackers. The gt-as-tracker row is a sanity check.",
        "rows": rows,
    }


def visibility(root: Path) -> dict[str, Any] | None:
    v = _load(root / "visibility" / "visibility_MOT17_val_half.json")
    if v is None:
        return None
    names = {
        "detector_boxes_x_score": "ours (geometry × score)",
        "detector_boxes": "ours, geometry only",
        "detector_score_as_visibility": "detector score as visibility (naive)",
        "oracle_pedestrians_only": "gt-box oracle (same boxes)",
    }
    keys = ("n", "mae", "spearman", "auroc_vis_lt_0.5", "ap_vis_lt_0.5")
    rows = [
        {
            "key": k,
            "label": lbl,
            "ours": k == "detector_boxes_x_score",
            **{f: _round(v[k].get(f), 3) for f in keys},
        }
        for k, lbl in names.items()
        if k in v
    ]
    return {"source": "docs/results/visibility/visibility_MOT17_val_half.json", "rows": rows}


def go_no_go(root: Path) -> dict[str, Any] | None:
    rows = _load(root / "go_no_go" / "rows_v2.json")
    if rows is None:
        return None
    keep = ("condition", "K", "K_ms_mean", "HOTA", "IDF1", "IDSW_trackeval", "p95_ms")
    return {
        "source": "docs/results/go_no_go/rows_v2.json",
        "verdict": "no-go",
        "sanity_check": True,
        "summary": "docs/results/go_no_go.md",
        "rows": [
            {k: _round(r[k], 1) if k not in ("condition", "K") else r[k] for k in keep}
            for r in rows
        ],
    }


def phase3(root: Path) -> dict[str, Any]:
    d = root / "phase3"
    if not d.exists():
        return {"status": "pending", "label": "pending DGX run", "items": PENDING_GPU}
    files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*.md"))
    return {
        "status": "available",
        "files": [f"docs/results/phase3/{f}" for f in files],
        "items": PENDING_GPU,
    }


# ---- laptop/DGX GPU run: docs/results/phase3/ (installed by `make ingest BRING_BACK=<tar>`) ----

GPU_PLOT_FILES = ("stress_visibility", "stress_density")
# wording of docs/preregistration/go_no_go_v2.md (frozen 4 Oct 2026), shown next to each verdict
GO_NO_GO_CRITERIA = {
    "1_idsw": ("ID switches, MOT17 val-half", "IDSW(K=5) <= 0.90 x IDSW(K=0)"),
    "2_identity": (
        "Identity quality, MOT17 val-half",
        "IDF1(K=5) >= IDF1(K=0) + 0.5 and HOTA(K=5) >= HOTA(K=0) - 0.2",
    ),
    "3_selectivity": ("Selectivity, MOT17 val-half", "trigger fires on <= 10 % of frames"),
    "4_compute": ("Compute, MOT17 val-half", "p95 update() <= 20 ms single-threaded"),
    "5_replication": (
        "Replication, MOT20 val-half",
        "IDSW(K=5) <= IDSW(K=0), K=5 run finishes within 60 min",
    ),
}


def _int(v: Any) -> int | None:
    return int(v) if isinstance(v, int | float) else None


def _gpu_row(r: dict[str, Any]) -> dict[str, Any]:
    tot = (r.get("stress") or {}).get("totals") or {}
    return {
        "name": r["name"],
        "label": r["label"],
        "group": r["group"],
        **{k: _round(r.get(k)) for k in ("HOTA", "IDF1", "MOTA", "AssA", "DetA")},
        "IDSW": _int(r.get("IDSW")),
        # CLEAR totals of the same TrackEval run (ablation.json -> stress.totals)
        "FP": _int(tot.get("FP")),
        "FN": _int(tot.get("FN")),
        "p95_ms": _round(r.get("p95_ms"), 2),
    }


# the visibility rule added on top of re-ID: (tracker + re-ID, same tracker + re-ID + rule)
REID_PAIRS = (
    ("bytetrack_reid", "reid_vis_gated", "ByteTrack"),
    ("botsort_reid", "botsort_reid_vis_gated", "BoT-SORT"),
)


def _reid_rule(datasets: list[dict[str, Any]], raw: dict[str, list[dict[str, Any]]]) -> dict:
    """Per dataset: tracker + re-ID vs the same tracker + re-ID + visibility rule (raw values)."""
    pairs = []
    for base, ours, family in REID_PAIRS:
        rows = []
        for ds in datasets:
            by = {r["name"]: r for r in raw[ds["id"]]}
            if base not in by or ours not in by:
                continue
            a, b = by[base], by[ours]
            rows.append(
                {
                    "dataset": ds["id"],
                    "base": {
                        "label": a["label"],
                        "HOTA": _round(a["HOTA"]),
                        "IDSW": _int(a["IDSW"]),
                        "IDF1": _round(a["IDF1"]),
                    },
                    "ours": {
                        "label": b["label"],
                        "HOTA": _round(b["HOTA"]),
                        "IDSW": _int(b["IDSW"]),
                        "IDF1": _round(b["IDF1"]),
                    },
                    "d_HOTA": _round(float(b["HOTA"]) - float(a["HOTA"]), 2),
                    "d_IDF1": _round(float(b["IDF1"]) - float(a["IDF1"]), 2),
                    "d_IDSW": int(b["IDSW"]) - int(a["IDSW"]),
                }
            )
        pairs.append(
            {
                "family": family,
                "base": base,
                "ours": ours,
                "rows": rows,
                "fewer_idsw_everywhere": bool(rows) and all(r["d_IDSW"] < 0 for r in rows),
                "hota_never_lower": bool(rows) and all(r["d_HOTA"] >= 0 for r in rows),
            }
        )
    return {"source": "docs/results/phase3/<id>/ablation.json", "pairs": pairs}


def _stress_scale(root: Path) -> dict[str, Any] | None:
    """Size of the simulated Stress Map, counted from its JSON (docs/results/stress)."""
    sm = _load(root / "stress" / "stress_map.json")
    if not sm or not sm.get("cells"):
        return None
    per_cell = sorted({len(c.get("seeds") or []) for c in sm["cells"]})
    trackers = len(sm.get("trackers") or {})
    return {
        "source": "docs/results/stress/stress_map.json",
        "cells": len(sm["cells"]),
        "seeds_per_cell": per_cell[0] if len(per_cell) == 1 else None,
        "seeds": [min(sm.get("seeds", [0])), max(sm.get("seeds", [0]))],
        "trackers": trackers,
        "runs": sum(len(c.get("seeds") or []) for c in sm["cells"]) * trackers,
    }


def _got(cid: str, k0: dict[str, Any], k5: dict[str, Any], rep: dict[str, Any]) -> str | None:
    """The K=0 -> K=5 values a criterion compares, straight from go_no_go_v2.json."""

    def f(r: dict[str, Any], k: str, nd: int = 1) -> str:
        v = r.get(k)
        return f"{float(v):.{nd}f}" if isinstance(v, int | float) else "?"

    if cid == "1_idsw":
        return f"IDSW {f(k0, 'IDSW', 0)} -> {f(k5, 'IDSW', 0)}"
    if cid == "2_identity":
        return f"IDF1 {f(k0, 'IDF1')} -> {f(k5, 'IDF1')}, HOTA {f(k0, 'HOTA')} -> {f(k5, 'HOTA')}"
    if cid == "3_selectivity":
        fr = k5.get("flag_rate")
        if not isinstance(fr, int | float):
            return None
        return f"fires on {100 * float(fr):.1f} % of frames"
    if cid == "4_compute":
        return f"p95 {f(k0, 'p95_ms', 2)} -> {f(k5, 'p95_ms', 2)} ms"
    if cid == "5_replication" and "idsw_k0" in rep:
        return f"MOT20 IDSW {f(rep, 'idsw_k0', 0)} -> {f(rep, 'idsw_k5', 0)} ({rep.get('status')})"
    if cid == "5_replication":
        return str(rep.get("status")) if rep else None
    return None


def gpu_plot_path(root: Path, name: str) -> Path | None:
    """``gpu-<condition>-<plot>`` -> docs/results/phase3/<condition>/<plot>.png (no traversal)."""
    if not name.startswith("gpu-") or name.count("-") != 2:
        return None
    _, cond, plot = name.split("-")
    if plot not in GPU_PLOT_FILES or not cond.replace("_", "").isalnum():
        return None
    p = root / "phase3" / cond / f"{plot}.png"
    return p if p.is_file() else None


def gpu(root: Path) -> dict[str, Any] | None:
    d = root / "phase3"
    summary = _load(d / "summary.json")
    if summary is None:
        return None
    ran = summary.get("ran", [])
    run_info = _load(d / "run_info.json") or {}
    detectors = run_info.get("detectors", {})
    datasets = []
    raw: dict[str, list[dict[str, Any]]] = {}
    for c in ran:
        abl = _load(d / c["id"] / "ablation.json")
        if abl is None or c.get("saw_eval_half", False):
            continue
        raw[c["id"]] = abl
        det = detectors.get(c["cache"], {})
        st0 = abl[0].get("stress") or {}
        datasets.append(
            {
                "id": c["id"],
                "cache": c["cache"],
                "dataset": c["cache"].split("/")[0],
                "clean": not c.get("saw_eval_half", False),
                "detector": det.get("weights"),
                "trained_on": det.get("trained_on"),
                "input_size": det.get("input_size"),
                "source": f"docs/results/phase3/{c['id']}/ablation.md",
                "sequences": len(st0.get("sequences") or []),
                # frames scored = sum of the per-density-bin frame counts of the same run
                "frames": sum(int(b.get("frames", 0)) for b in st0.get("density") or []),
                "rows": [_gpu_row(r) for r in abl],
                "plots": {
                    k: f"/api/benchmarks/plots/gpu-{c['id']}-{k}.png"
                    for k in GPU_PLOT_FILES
                    if (d / c["id"] / f"{k}.png").is_file()
                },
            }
        )
    ran_ids = {c["id"] for c in ran}
    excluded = sorted(
        p.name
        for p in d.iterdir()
        if p.is_dir() and "SAW_EVAL_HALF" in p.name and p.name not in ran_ids
    )
    gng = _load(d / "go_no_go_v2.json")
    go_no_go_v2 = None
    if gng:
        rows = gng.get("rows", {})
        k0, k5 = rows.get("k0", {}), rows.get("k5", {})
        rep = gng.get("replication", {})
        go_no_go_v2 = {
            "verdict": gng.get("verdict"),
            "source": "docs/results/phase3/go_no_go_v2.json",
            "rule": "docs/preregistration/go_no_go_v2.md",
            "what": "Delayed decision (K-frame look-ahead with re-ID, "
            "configs/ours/delayed_v2_reid.yaml); GO only if all five hold for K = 5.",
            "criteria": [
                {
                    "id": cid,
                    "what": GO_NO_GO_CRITERIA.get(cid, (cid, ""))[0],
                    "rule": GO_NO_GO_CRITERIA.get(cid, (cid, ""))[1],
                    "got": _got(cid, k0, k5, rep),
                    "passed": bool(ok),
                }
                for cid, ok in gng.get("criteria", {}).items()
            ],
            "rows": [
                {
                    "arm": k,
                    **{m: _round(r.get(m)) for m in ("HOTA", "IDF1", "MOTA", "AssA")},
                    "IDSW": int(r["IDSW"]),
                    "p95_ms": _round(r.get("p95_ms"), 2),
                    "flag_pct": _round(100 * float(r.get("flag_rate", 0.0))),
                }
                for k, r in rows.items()
            ],
        }
    rp = _load(d / "reproduction.json")
    reproduction = None
    if rp:
        ref, ours = rp.get("reference", {}), rp.get("ours", {})
        tol = {"MOTA": 1.0, "IDF1": 1.5}  # configs/phase3/plan.yaml reproduction.tolerance
        reproduction = {
            "verdict": rp.get("verdict"),
            "source": "docs/results/phase3/reproduction.md",
            "reference": "ByteTrack arXiv 2110.06864v3, Table 1 (IoU|IoU) / Table 11",
            "rows": [
                {
                    "metric": m,
                    "paper": ref.get(m),
                    "ours": _round(ours.get(m), 1 if m in ("MOTA", "IDF1") else 0),
                    "delta": _round(float(ours[m]) - float(ref[m]), 1)
                    if m in ours and m in ref
                    else None,
                    "tolerance": tol.get(m),
                    "within": rp.get("within_tolerance", {}).get(m),
                }
                for m in ("MOTA", "IDF1", "IDSW", "FP", "FN")
            ],
        }
    ge = _load(d / "gate_events" / "gate_events.json")
    gcrit = (go_no_go_v2 or {}).get("criteria", [])
    scale = {
        "evaluations": len(datasets),
        "tracker_runs": sum(len(x["rows"]) for x in datasets),
        "tracker_configs": len({r["name"] for x in datasets for r in x["rows"]}),
        "sequences": sum(x["sequences"] for x in datasets),
        "frames": sum(x["frames"] for x in datasets),
        "tracked_frames": sum(x["frames"] * len(x["rows"]) for x in datasets),
        "excluded": len(excluded),
        "go_no_go": {
            "verdict": (go_no_go_v2 or {}).get("verdict"),
            "failed": sum(1 for c in gcrit if not c["passed"]),
            "total": len(gcrit),
        }
        if go_no_go_v2
        else None,
        "stress": _stress_scale(root),
        "detectors": sorted({x["detector"] for x in datasets if x["detector"]}),
        "reid": sorted({v.get("reid") for v in detectors.values() if v.get("reid")}),
        "precision": sorted({v.get("precision") for v in detectors.values() if v.get("precision")}),
    }
    return {
        "scale": scale,
        "reid_rule": _reid_rule(datasets, raw),
        "source": "docs/results/phase3",
        "analysis": "docs/results/gpu/ANALYSIS.md",
        "gpu": (run_info.get("gpu") or {}).get("name"),
        "profile": run_info.get("profile"),
        "tuned": summary.get("tuned"),
        "datasets": datasets,
        "excluded": excluded,
        "reproduction": reproduction,
        "go_no_go_v2": go_no_go_v2,
        "gate_events": {
            "source": "docs/results/phase3/gate_events/gate_events.md",
            "rows": [
                {
                    "label": k,
                    "check": "check" in k,
                    **{
                        f: v.get(f)
                        for f in (
                            "gt_crossings",
                            "trk_crossings",
                            "count_error",
                            "precision",
                            "recall",
                            "attribution_error",
                            "tailgate_precision",
                            "tailgate_recall",
                        )
                    },
                }
                for k, v in ge.items()
            ],
        }
        if ge
        else None,
    }


def benchmarks(root: Path) -> dict[str, Any]:
    abl = ablation(root)
    head = next((r for r in abl["rows"] if r["headline"]), None) if abl else None
    return {
        "headline": head,
        "ablation": abl,
        "gate_events": gate_events(root),
        "visibility": visibility(root),
        "go_no_go": go_no_go(root),
        "phase3": phase3(root),
        "gpu": gpu(root),
        "plots": {
            k: f"/api/benchmarks/plots/{k}.png" for k, v in PLOTS.items() if (root / v).exists()
        },
    }

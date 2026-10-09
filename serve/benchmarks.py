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
    files = sorted(str(p.relative_to(root.parent.parent)) for p in d.rglob("*.md"))
    return {"status": "available", "files": files, "items": PENDING_GPU}


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
        "plots": {
            k: f"/api/benchmarks/plots/{k}.png" for k, v in PLOTS.items() if (root / v).exists()
        },
    }

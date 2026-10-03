"""Score the box-only visibility estimate against MOT ``gt.txt`` visibility (brief §2.4).

    python -m eval.visibility_eval --root data/MOT17 --split val_half --dets public \
        --occluder-min-score 0.5 --out docs/results/visibility

Protocol:

1. GT set: rows with ``class == 1`` and ``flag == 1`` on the split (TrackEval's kept set).
2. Our estimate is computed on the **detector boxes** of each frame (the tracker's input: score >
   ``--det-min-score``), Hungarian-matched to GT with IoU ≥ 0.5. Each matched pair gives (v̂, v).
   Coverage (matched fraction per gt-visibility bin) is reported, because detectors miss exactly
   the heavily occluded people.
3. Metrics: MAE, RMSE, bias, Spearman ρ, Kendall τ-b, binned calibration (gt bins
   [0, .25), [.25, .5), [.5, .75), [.75, 1]) with an ECE-style weighted gap, AUROC and AP for
   "heavily occluded" (v < 0.5) and "nearly invisible" (v < 0.25).
4. Reference rows:
   * **gt-box oracle, all classes**: the estimator on gt boxes with every annotated class
     (cars, static persons, occluders) as an occluder. MOT generated its visibility almost this
     way, so this is an implementation check, not a result.
   * **gt-box oracle, pedestrians only**: occluders are class-1 gt boxes. This is the floor for any
     person-detector-based estimate, since static occluders are never detected.
   * the **detector score** as a naive visibility proxy, and a **constant** (the mean).
5. Parameters are tuned on ``train_half`` only (``--tune``); the val half is reported once.
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau, rankdata, spearmanr

from eval.splits import find_sequences, half_range
from tracker.cost import iou_matrix
from tracker.io import load_mot_det_txt, read_seqinfo
from tracker.types import tlwh_to_xyxy
from tracker.visibility import DepthOrderVisibility

MOVING_CAMERA = {
    "MOT17-05",
    "MOT17-10",
    "MOT17-11",
    "MOT17-13",
    "MOT17-06",
    "MOT17-07",
    "MOT17-12",
    "MOT17-14",
}
GT_BINS = (0.0, 0.25, 0.5, 0.75, 1.0000001)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """P(score_pos > score_neg) with ties counted ½ (Mann–Whitney)."""
    labels = labels.astype(bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    tp = labels[order].cumsum()
    prec = tp / np.arange(1, len(order) + 1)
    return float((prec * labels[order]).sum() / labels.sum())


def metrics(est: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    est = np.asarray(est, np.float64)
    gt = np.asarray(gt, np.float64)
    n = len(gt)
    out: dict[str, Any] = {"n": n}
    if n == 0:
        return out
    err = est - gt
    out.update(
        mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err**2).mean())), bias=float(err.mean())
    )
    const = np.std(est) == 0
    out["spearman"] = float("nan") if const else float(spearmanr(est, gt).statistic)
    out["kendall_tau_b"] = float("nan") if const else float(kendalltau(est, gt).statistic)
    bins = []
    ece = 0.0
    for lo, hi in itertools.pairwise(GT_BINS):
        m = (gt >= lo) & (gt < hi)
        k = int(m.sum())
        b: dict[str, Any] = {"gt_bin": [lo, min(hi, 1.0)], "n": k}
        if k:
            b["mean_est"] = float(est[m].mean())
            b["mean_gt"] = float(gt[m].mean())
            ece += k / n * abs(b["mean_est"] - b["mean_gt"])
        bins.append(b)
    out["calibration"] = bins
    out["ece"] = float(ece)
    for thr in (0.5, 0.25):
        lab = gt < thr
        out[f"auroc_vis_lt_{thr}"] = auroc(lab, -est)
        out[f"ap_vis_lt_{thr}"] = average_precision(lab, -est)
        out[f"prevalence_vis_lt_{thr}"] = float(lab.mean())
    return out


def match(det_xyxy: np.ndarray, gt_xyxy: np.ndarray, thr: float = 0.5) -> list[tuple[int, int]]:
    if len(det_xyxy) == 0 or len(gt_xyxy) == 0:
        return []
    iou = iou_matrix(det_xyxy, gt_xyxy)
    r, c = linear_sum_assignment(-iou)
    return [(int(i), int(j)) for i, j in zip(r, c, strict=True) if iou[i, j] >= thr]


def collect(
    root: Path,
    dataset: str,
    split: str,
    occluder_min_score: float,
    det_min_score: float = 0.1,
    grid: int = 16,
    border: bool = True,
    score_map: str = "none",
) -> dict[str, Any]:
    """Per-pair arrays for the detector-box estimate and per-gt arrays for the oracles."""
    rows: dict[str, list[Any]] = {k: [] for k in ("est", "gt", "score", "seq", "ndet")}
    orc: dict[str, list[Any]] = {k: [] for k in ("all", "ped", "gt", "seq", "matched")}
    for seq in find_sequences(root, dataset):
        info = read_seqinfo(seq)
        est = DepthOrderVisibility(
            (info.im_width, info.im_height),
            grid=grid,
            border=border,
            occluder_min_score=occluder_min_score,
        )
        orc_est = DepthOrderVisibility((info.im_width, info.im_height), grid=grid, border=border)
        lo, hi = half_range(info.seq_length, split)
        gt = np.loadtxt(seq / "gt" / "gt.txt", delimiter=",", ndmin=2)
        gt = gt[(gt[:, 0] >= lo) & (gt[:, 0] <= hi)]
        det = load_mot_det_txt(seq, dataset, "train")
        keep = (det.frame >= lo) & (det.frame <= hi)
        d_frame, d_tlwh, d_score = (
            det.frame[keep],
            det.tlwh[keep].astype(np.float64),
            det.score[keep].astype(np.float64),
        )
        if score_map == "logistic":
            d_score = 1.0 / (1.0 + np.exp(-d_score))
        key = seq.name.rsplit("-", 1)[0] if dataset.upper() == "MOT17" else seq.name
        for f in range(lo, hi + 1):
            g = gt[gt[:, 0] == f]
            ped = (g[:, 7] == 1) & (g[:, 6] == 1)
            gx = tlwh_to_xyxy(g[:, 2:6])
            mk = d_frame == f
            dx = tlwh_to_xyxy(d_tlwh[mk])
            ds = d_score[mk]
            sel = ds > det_min_score
            dx, ds = dx[sel], ds[sel]
            if ped.any():
                v_all = orc_est(gx)[ped]
                v_ped = orc_est(gx, occluder_mask=(g[:, 7] == 1))[ped]
                matched = np.zeros(int(ped.sum()), bool)
                pairs = match(dx, gx[ped])
                if pairs:
                    v_det = est(dx, ds)
                    for i, j in pairs:
                        rows["est"].append(v_det[i])
                        rows["gt"].append(g[ped][j, 8])
                        rows["score"].append(ds[i])
                        rows["seq"].append(key)
                        rows["ndet"].append(len(dx))
                        matched[j] = True
                orc["all"].extend(v_all)
                orc["ped"].extend(v_ped)
                orc["gt"].extend(g[ped][:, 8])
                orc["seq"].extend([key] * int(ped.sum()))
                orc["matched"].extend(matched)
    return {
        "pairs": {k: np.asarray(v) for k, v in rows.items()},
        "oracle": {k: np.asarray(v) for k, v in orc.items()},
    }


def report(data: dict[str, Any]) -> dict[str, Any]:
    p, o = data["pairs"], data["oracle"]
    mean_gt = float(p["gt"].mean()) if len(p["gt"]) else float("nan")
    out: dict[str, Any] = {
        "detector_boxes": metrics(p["est"], p["gt"]),
        "detector_boxes_x_score": metrics(p["est"] * np.clip(p["score"], 0, 1), p["gt"]),
        "detector_score_as_visibility": metrics(p["score"], p["gt"]),
        "constant_mean": metrics(np.full(len(p["gt"]), mean_gt), p["gt"]),
        "oracle_all_classes": metrics(o["all"], o["gt"]),
        "oracle_pedestrians_only": metrics(o["ped"], o["gt"]),
        "oracle_pedestrians_only_on_matched": metrics(
            o["ped"][o["matched"]], o["gt"][o["matched"]]
        ),
    }
    cov = []
    for lo, hi in itertools.pairwise(GT_BINS):
        m = (o["gt"] >= lo) & (o["gt"] < hi)
        cov.append(
            {
                "gt_bin": [lo, min(hi, 1.0)],
                "gt_boxes": int(m.sum()),
                "matched_by_detector": float(o["matched"][m].mean()) if m.any() else None,
            }
        )
    out["coverage"] = cov
    moving = np.array([s in MOVING_CAMERA for s in p["seq"]], bool)
    out["by_camera"] = {
        "static": metrics(p["est"][~moving], p["gt"][~moving]),
        "moving": metrics(p["est"][moving], p["gt"][moving]),
    }
    out["by_sequence"] = {
        s: metrics(p["est"][p["seq"] == s], p["gt"][p["seq"] == s])
        for s in sorted(set(p["seq"].tolist()))
    }
    if len(p["ndet"]):
        t1, t2 = np.quantile(p["ndet"], [1 / 3, 2 / 3])
        out["by_density"] = {
            f"dets/frame <= {t1:.0f}": metrics(p["est"][p["ndet"] <= t1], p["gt"][p["ndet"] <= t1]),
            f"{t1:.0f} < dets/frame <= {t2:.0f}": metrics(
                p["est"][(p["ndet"] > t1) & (p["ndet"] <= t2)],
                p["gt"][(p["ndet"] > t1) & (p["ndet"] <= t2)],
            ),
            f"dets/frame > {t2:.0f}": metrics(p["est"][p["ndet"] > t2], p["gt"][p["ndet"] > t2]),
        }
    return out


def _f(x: Any, nd: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def markdown(rep: dict[str, Any], title: str) -> str:
    lines = [
        f"### {title}",
        "",
        "| Estimate | n | MAE | RMSE | bias | Spearman ρ | Kendall τ-b | ECE | AUROC v<0.5 | "
        "AP v<0.5 | AUROC v<0.25 | AP v<0.25 |",
        "|---" * 12 + "|",
    ]
    names = [
        ("detector_boxes_x_score", "**ours (geometry × score), on detector boxes**"),
        ("detector_boxes", "ours, geometry only"),
        ("detector_score_as_visibility", "detector score as visibility (naive)"),
        ("constant_mean", "constant (mean gt vis of matched)"),
        ("oracle_pedestrians_only_on_matched", "gt-box oracle, pedestrians only (same boxes)"),
        ("oracle_pedestrians_only", "gt-box oracle, pedestrians only (all gt)"),
        ("oracle_all_classes", "gt-box oracle, all classes (implementation check)"),
    ]
    for k, label in names:
        m = rep[k]
        lines.append(
            f"| {label} | {m['n']} | {_f(m.get('mae'))} | {_f(m.get('rmse'))} | "
            f"{_f(m.get('bias'))} | {_f(m.get('spearman'))} | {_f(m.get('kendall_tau_b'))} | "
            f"{_f(m.get('ece'))} | {_f(m.get('auroc_vis_lt_0.5'))} | "
            f"{_f(m.get('ap_vis_lt_0.5'))} | {_f(m.get('auroc_vis_lt_0.25'))} | "
            f"{_f(m.get('ap_vis_lt_0.25'))} |"
        )
    lines += [
        "",
        "Calibration of ours, geometry only (by gt-visibility bin):",
        "",
        "| gt bin | n | mean v̂ | mean v |",
        "|---|---|---|---|",
    ]
    for b in rep["detector_boxes"].get("calibration", []):
        lines.append(
            f"| [{b['gt_bin'][0]:.2f}, {b['gt_bin'][1]:.2f}) | {b['n']} | "
            f"{_f(b.get('mean_est'))} | {_f(b.get('mean_gt'))} |"
        )
    lines += [
        "",
        "Coverage: fraction of gt pedestrians matched by a detector box (IoU ≥ 0.5):",
        "",
        "| gt bin | gt boxes | matched |",
        "|---|---|---|",
    ]
    for c in rep["coverage"]:
        lines.append(
            f"| [{c['gt_bin'][0]:.2f}, {c['gt_bin'][1]:.2f}) | {c['gt_boxes']} | "
            f"{_f(c['matched_by_detector'])} |"
        )
    lines += [
        "",
        "Stratified (ours, geometry only):",
        "",
        "| Stratum | n | MAE | Spearman ρ | AUROC v<0.5 |",
        "|---|---|---|---|---|",
    ]
    for group in ("by_camera", "by_density", "by_sequence"):
        for k, m in rep.get(group, {}).items():
            lines.append(
                f"| {k} | {m['n']} | {_f(m.get('mae'))} | {_f(m.get('spearman'))} | "
                f"{_f(m.get('auroc_vis_lt_0.5'))} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval.visibility_eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", type=Path, default=Path("data/MOT17"))
    p.add_argument("--dataset", default="MOT17")
    p.add_argument("--split", default="val_half")
    p.add_argument(
        "--occluder-min-score",
        type=float,
        default=0.1,
        help="0.1 = chosen on MOT17 train_half by AUROC v<0.5",
    )
    p.add_argument("--det-min-score", type=float, default=0.1)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument(
        "--tune",
        action="store_true",
        help="sweep occluder_min_score on --split (use train_half) and print MAE/AUROC",
    )
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    warnings.filterwarnings("ignore", message="An input array is constant")
    if a.tune:
        res = {}
        for t in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95):
            m = report(collect(a.root, a.dataset, a.split, t, a.det_min_score, a.grid))
            d = m["detector_boxes"]
            res[t] = {k: d[k] for k in ("mae", "spearman", "auroc_vis_lt_0.5", "ece")}
            fx = m["detector_boxes_x_score"]
            res[t].update({f"x_score_{k}": fx[k] for k in ("mae", "spearman", "auroc_vis_lt_0.5")})
            print(
                f"occluder_min_score={t}: " + ", ".join(f"{k}={v:.4f}" for k, v in res[t].items())
            )
        if a.out:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / f"tune_{a.dataset}_{a.split}.json").write_text(json.dumps(res, indent=1))
        return 0
    rep = report(collect(a.root, a.dataset, a.split, a.occluder_min_score, a.det_min_score, a.grid))
    rep["params"] = {
        "occluder_min_score": a.occluder_min_score,
        "det_min_score": a.det_min_score,
        "grid": a.grid,
        "split": a.split,
        "dataset": a.dataset,
    }
    md = markdown(rep, f"{a.dataset} {a.split}, public FRCNN detections")
    print(md)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"visibility_{a.dataset}_{a.split}.json").write_text(json.dumps(rep, indent=1))
        (a.out / f"visibility_{a.dataset}_{a.split}.md").write_text(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

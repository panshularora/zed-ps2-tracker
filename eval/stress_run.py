"""Run trackers on stress scenarios, score them, and explain ID changes from the tracker's trace.

    python -m eval.stress_run grid      [--seeds 20 --seed0 1000 --workers 7]
    python -m eval.stress_run scenarios [--seeds 20 --seed0 1000]
    python -m eval.stress_run noise     [--seeds 20 --seed0 1000]

Tuning must use other seeds than reporting (``--seed0 0`` for tuning, ``1000`` for reports).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from eval.stress_metrics import (
    TrackRows,
    first_disagreement,
    gate_counts,
    id_change_events,
    match_frames,
    tally,
    trackeval_metrics,
)
from eval.stress_sim import FPS, Frame, H, Scenario, W, render
from tracker.bytetrack import ByteTrackLike
from tracker.config import TrackerConfig, load_config
from tracker.io import CacheMeta
from tracker.runner import build_tracker
from tracker.types import Detections

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "stress"
TRACKERS = {
    "ours": "configs/ours/visibility_botsort.yaml",
    "bytetrack": "configs/baselines/bytetrack.yaml",
}
GRID_HIDE = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
GRID_PEOPLE = (5, 10, 15, 20, 30, 40)
SCENARIOS: dict[str, Scenario] = {
    "lookalike_cross": Scenario(kind="lookalike_cross", people=8, hide_s=1.0),
    "stop_and_go": Scenario(kind="stop_and_go", people=10, hide_s=1.0),
    "step_back": Scenario(kind="step_back", people=10, hide_s=1.0),
    "group": Scenario(kind="group", people=12, hide_s=1.0),
    "camera_shake": Scenario(kind="camera_shake", people=10, hide_s=1.0, jitter_px=4.0),
    "linger": Scenario(kind="linger", people=10, hide_s=1.0),
    "fast_walker": Scenario(kind="fast_walker", people=10, hide_s=1.0),
    "dropout_5f": Scenario(kind="dropout", people=10, dropout_f=5),
    "dropout_15f": Scenario(kind="dropout", people=10, dropout_f=15),
    "dropout_30f": Scenario(kind="dropout", people=10, dropout_f=30),
}
NOISE_LEVELS = (0.05, 0.15, 0.4)
_CFG: dict[str, TrackerConfig] = {}


def config(path: str) -> TrackerConfig:
    if path not in _CFG:
        _CFG[path] = load_config(ROOT / path)
    return _CFG[path]


def meta() -> CacheMeta:
    return CacheMeta(
        sequence="stress",
        dataset="synthetic",
        split="full",
        frame_rate=FPS,
        seq_length=0,
        im_width=W,
        im_height=H,
    )


def _tlwh(b: Any) -> tuple[float, float, float, float]:
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def run_tracker(
    cfg: TrackerConfig, frames: list[Frame], trace: bool = False
) -> tuple[TrackRows, list[dict[str, Any]]]:
    trk, _ = build_tracker(cfg, FPS, meta())
    if trace:
        assert isinstance(trk, ByteTrackLike)
        trk.trace_enabled = True
    rows: TrackRows = []
    traces: list[dict[str, Any]] = []
    for fr in frames:
        res = trk.update(Detections(fr.dets, fr.scores))
        assert res is not None
        rows.append([(o.track_id, _tlwh(o.tlwh)) for o in res.tracks])
        if trace:
            assert isinstance(trk, ByteTrackLike)
            assert trk.last_trace is not None
            traces.append(trk.last_trace)
    return rows, traces


# ---------------------------------------------------------------- explanation of one ID change
def _centre(b: list[float] | tuple[float, ...]) -> tuple[float, float]:
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def explain(
    ev: tuple[int, int, int, int],
    frames: list[Frame],
    matches: list[dict[int, int]],
    traces: list[dict[str, Any]],
    cfg: TrackerConfig,
) -> dict[str, Any]:
    """Why person ``pid``'s ID went from ``a`` to ``b`` at ``frame``: read from the trace."""
    f, pid, a, b = ev
    f_last = max((k + 1 for k in range(f - 1) if matches[k].get(pid) == a), default=0)
    # what happened to track a after it last had this person
    a_lost: dict[str, Any] | None = None
    a_expired: dict[str, Any] | None = None
    for k in range(f_last, f):
        tr = traces[k]
        info = tr["tracks"].get(a)
        if a_lost is None and info and info.get("state") == "lost":
            a_lost = {
                "frame": tr["frame"],
                **{x: info.get(x) for x in ("visibility", "buffer", "buffer_extended")},
            }
        for e in tr["expired"]:
            if e["track_id"] == a and a_expired is None:
                a_expired = {"frame": tr["frame"], **e}
    if a_lost is None:  # a track becomes lost at the end of the frame it missed: next snapshot
        for k in range(f_last, min(f + 1, len(traces))):
            info = traces[k]["tracks"].get(a)
            if info and "buffer" in info:
                a_lost = {
                    "frame": traces[k]["frame"],
                    "visibility": info.get("visibility"),
                    "buffer": info.get("buffer"),
                    "buffer_extended": info.get("buffer_extended"),
                }
                break
    # where track b came from
    born = None
    for k in range(f):
        for bb in traces[k]["births"]:
            if bb["track_id"] == b:
                born = (traces[k]["frame"], bb["det"])
    transfer = born is None or born[0] <= f_last
    if not transfer and born is not None:
        fd, det = born
    else:
        fd, det = f, None
        for st in traces[f - 1]["stages"]:
            for t_id, d in st["matches"]:
                if t_id == b:
                    det = d
    tr = traces[fd - 1]
    fr = frames[fd - 1]
    cands: list[dict[str, Any]] = []
    a_row: dict[str, Any] | None = None
    if det is not None:
        for st in tr["stages"]:
            if det not in st["det_idx"]:
                continue
            j = st["det_idx"].index(det)
            matched = dict(st["matches"])
            for i, t_id in enumerate(st["track_ids"]):
                info = tr["tracks"].get(t_id, {})
                c: tuple[float, float] = _centre(info.get("pred_tlwh", [0, 0, 0, 0]))
                dc = _centre(fr.dets[det].tolist())
                row = {
                    "stage": st["stage"],
                    "track_id": t_id,
                    "state": info.get("state"),
                    "iou": st["iou"][i][j],
                    "cost": st["cost"][i][j],
                    "thresh": st["thresh"],
                    "centre_dist_rel": round(
                        math.hypot(c[0] - dc[0], c[1] - dc[1]) / max(fr.dets[det][3], 1.0), 3
                    ),
                    "frames_lost": info.get("frames_lost"),
                    "matched_det": matched.get(t_id),
                }
                if t_id == a:
                    a_row = row
                cands.append(row)
    cands.sort(key=lambda r: r["cost"])
    top = cands[:4]
    if a_row is not None and a_row not in top:
        top.append(a_row)
    a_alive = a in tr["tracks"]
    if transfer:
        reason = "transfer"
    elif not a_alive:
        reason = "expired"
    elif det is not None and det not in tr["high_idx"]:
        reason = "low_score_box"
    elif a_row is not None and a_row["matched_det"] is not None and a_row["matched_det"] != det:
        reason = "taken_by_other_box"
    elif a_row is not None and a_row["cost"] > a_row["thresh"]:
        reason = "cost_above_threshold"
    else:
        reason = "not_a_candidate"
    det_info = None
    if det is not None:
        det_info = {
            "tlwh": [round(float(v), 1) for v in fr.dets[det]],
            "score": round(float(fr.scores[det]), 3),
            "est_visibility": round(float(tr["det_vis"][det]), 3),
            "true_pid": fr.det_pid[det],
        }
    return {
        "frame": f,
        "pid": pid,
        "from_id": a,
        "to_id": b,
        "last_seen_as_from": f_last,
        "decision_frame": fd,
        "reason": reason,
        "to_born": born[0] if born else None,
        "from_lost": a_lost,
        "from_expired": a_expired,
        "detection": det_info,
        "candidates": top,
        "occluded_thresh": cfg.visibility.occluded_thresh,
        "base_buffer": cfg.lifecycle.track_buffer,
    }


def why_text(x: dict[str, Any]) -> str:
    """One sentence built only from the trace fields above."""
    a, b = x["from_id"], x["to_id"]
    lost = x.get("from_lost") or {}
    vis = lost.get("visibility")
    buf = lost.get("buffer")
    vis_s = ""
    if vis is not None and buf is not None:
        rule = "extended" if lost.get("buffer_extended") else "not extended"
        vis_s = (
            f" When #{a} lost its box its visibility was {vis:.2f} (threshold "
            f"{x['occluded_thresh']}), so its memory was {rule}: {buf} frames."
        )
    r = x["reason"]
    if r == "expired":
        e = x.get("from_expired") or {}
        return (
            f"#{a} was deleted at frame {e.get('frame')} after {e.get('frames_lost')} frames "
            f"without a box (memory {e.get('buffer')}); the person's next box started #{b}." + vis_s
        )
    if r == "cost_above_threshold":
        c: dict[str, Any] = next((k for k in x["candidates"] if k["track_id"] == a), {})
        return (
            f"#{a} was still remembered, but its predicted box overlapped the new box by IoU "
            f"{c.get('iou', 0):.2f} (cost {c.get('cost', 1):.2f} > {c.get('thresh')}), "
            f"centre off by {c.get('centre_dist_rel', 0):.2f} box heights; a new #{b} "
            "was started." + vis_s
        )
    if r == "taken_by_other_box":
        return f"#{a} was matched to a different box that frame, so this person's box started #{b}."
    if r == "low_score_box":
        d = x.get("detection") or {}
        return (
            f"The person came back as a weak box (score {d.get('score')}); lost tracks such "
            f"as #{a} are only matched to confident boxes, so #{b} took it." + vis_s
        )
    if r == "transfer":
        return (
            f"#{b} already existed (another track) and was matched onto this person "
            f"instead of #{a}."
        )
    return f"#{a} was not a candidate for this box; #{b} took it." + vis_s


# ---------------------------------------------------------------- one seed, both trackers
def run_seed(
    sc: Scenario, seed: int, trackers: dict[str, str], explain_first: bool = True
) -> dict[str, Any]:
    frames = render(sc, seed)
    out: dict[str, Any] = {"seed": seed}
    evs: dict[str, list[tuple[int, int, int, int]]] = {}
    matches_of: dict[str, list[dict[int, int]]] = {}
    for name, path in trackers.items():
        rows, _ = run_tracker(config(path), frames)
        m = match_frames(frames, rows)
        te = trackeval_metrics(frames, rows)
        t = tally(frames, m)
        true_n, trk_n = gate_counts(frames, rows)
        ev = id_change_events(m)
        evs[name], matches_of[name] = ev, m
        out[name] = {
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in te.items()},
            "people": t.people,
            "kept": t.kept,
            "changed": t.changed,
            "id_changes": len(ev),
            "gate_true": true_n,
            "gate_counted": trk_n,
            "gate_error": trk_n - true_n,
        }
    names = list(trackers)
    if len(names) == 2:
        d = first_disagreement(evs[names[0]], evs[names[1]])
        if d is not None:
            d["changed"] = names[0] if d["changed"] == "a" else names[1]
            if explain_first:
                why = []
                for name in names:
                    for e in evs[name]:
                        if e[1] == d["pid"] and abs(e[0] - d["frame"]) <= 2 * FPS:
                            _, traces = run_tracker(config(trackers[name]), frames, trace=True)
                            x = explain(e, frames, matches_of[name], traces, config(trackers[name]))
                            x["tracker"] = name
                            x["why"] = why_text(x)
                            why.append(x)
                            break
                d["why"] = why
        out["first_disagreement"] = d
    return out


# ---------------------------------------------------------------- statistics
def bootstrap_ci(x: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def sign_test_p(diff: np.ndarray) -> float:
    """Exact two-sided sign test (ties dropped)."""
    pos, neg = int((diff > 0).sum()), int((diff < 0).sum())
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * p))


def verdict(diff: np.ndarray) -> str:
    """ID-change difference ours − ByteTrack per seed: negative = ours better."""
    lo, hi = bootstrap_ci(diff)
    p = sign_test_p(diff)
    if hi < 0 and p < 0.05:
        return "ours_better"
    if lo > 0 and p < 0.05:
        return "ours_worse"
    if np.all(diff == 0) or (lo >= -0.5 and hi <= 0.5):
        return "tie"
    return "unclear"


METRICS = (
    "people",
    "kept",
    "changed",
    "id_changes",
    "IDSW",
    "Frag",
    "IDF1",
    "AssA",
    "HOTA",
    "gate_error",
)


def summarise(sc: Scenario, runs: list[dict[str, Any]], trackers: list[str]) -> dict[str, Any]:
    cell: dict[str, Any] = {"scenario": sc.as_dict(), "seeds": [r["seed"] for r in runs]}
    for name in trackers:
        agg: dict[str, Any] = {}
        for m in METRICS:
            v = np.asarray([r[name][m] for r in runs], float)
            agg[m] = {
                "mean": round(float(v.mean()), 4),
                "min": round(float(v.min()), 4),
                "max": round(float(v.max()), 4),
            }
        agg["abs_gate_error_mean"] = round(
            float(np.mean([abs(r[name]["gate_error"]) for r in runs])), 3
        )
        agg["per_seed"] = {
            m: [r[name][m] for r in runs]
            for m in ("people", "kept", "id_changes", "IDF1", "AssA", "gate_error")
        }
        cell[name] = agg
    if len(trackers) == 2:
        a, b = trackers
        diff = np.asarray([r[a]["id_changes"] - r[b]["id_changes"] for r in runs], float)
        lo, hi = bootstrap_ci(diff)
        cell["id_change_diff"] = {
            "per_seed": diff.astype(int).tolist(),
            "mean": round(float(diff.mean()), 3),
            "ci95": [round(lo, 3), round(hi, 3)],
            "sign_test_p": round(sign_test_p(diff), 4),
            "verdict": verdict(diff),
            "definition": f"{a} minus {b}; negative = {a} better",
        }
        cell["first_disagreement"] = [
            {"seed": r["seed"], **r["first_disagreement"]}
            if r.get("first_disagreement")
            else {"seed": r["seed"], "frame": None}
            for r in runs
        ]
    return cell


def _job(args: tuple[Scenario, int, dict[str, str]]) -> dict[str, Any]:
    sc, seed, trackers = args
    return run_seed(sc, seed, trackers)


def run_cells(
    cells: list[Scenario], seeds: list[int], trackers: dict[str, str], workers: int
) -> list[dict[str, Any]]:
    jobs = [(sc, s, trackers) for sc in cells for s in seeds]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_job, jobs, chunksize=2))
    out = []
    for i, sc in enumerate(cells):
        out.append(summarise(sc, res[i * len(seeds) : (i + 1) * len(seeds)], list(trackers)))
    return out


def provenance(seeds: list[int], trackers: dict[str, str]) -> dict[str, Any]:
    import subprocess

    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    return {
        "generator": "eval/stress_sim.py + eval/stress_run.py",
        "git": rev,
        "trackers": trackers,
        "seeds": seeds,
        "fps": FPS,
        "frame": [W, H],
        "gt_rule": "boxes with true visible fraction >= 0.25 are scored; "
        "fully hidden frames are not",
        "kept_rule": "matched (IoU>=0.5) in >=80% of scored frames under one ID "
        "never matched to another person",
        "metrics": "HOTA/AssA/DetA, IDF1, CLEAR IDSW/Frag from vendored TrackEval "
        "(third_party/trackeval)",
        "id_changes": "times a person's matched ID differs from their previous matched ID "
        "(= CLEAR IDSW)",
        "gate_error": "tracker crossings minus true crossings (foot point, 8 px hysteresis)",
        "cpu_only": True,
    }


def write_csv(path: Path, cells: list[dict[str, Any]], trackers: list[str], key: str) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        hdr = ["cell", "people_n", "hide_s", "noise"]
        for t in trackers:
            hdr += [f"{t}_{m}_{s}" for m in METRICS for s in ("mean", "min", "max")]
            hdr += [f"{t}_abs_gate_error_mean"]
        hdr += ["id_change_diff_mean", "ci95_lo", "ci95_hi", "sign_p", "verdict"]
        w.writerow(hdr)
        for c in cells:
            s = c["scenario"]
            row = [c.get(key, s["label"] or s["kind"]), s["people"], s["hide_s"], s["noise"]]
            for t in trackers:
                row += [c[t][m][x] for m in METRICS for x in ("mean", "min", "max")]
                row += [c[t]["abs_gate_error_mean"]]
            d = c["id_change_diff"]
            row += [d["mean"], d["ci95"][0], d["ci95"][1], d["sign_test_p"], d["verdict"]]
            w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("grid", "scenarios", "noise"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--ours", default=TRACKERS["ours"], help="config of 'ours' (for tuning runs)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    trackers = {"ours": a.ours, "bytetrack": TRACKERS["bytetrack"]}
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.what == "grid":
        cells = [Scenario(people=n, hide_s=h) for n in GRID_PEOPLE for h in GRID_HIDE]
    elif a.what == "noise":
        cells = [
            Scenario(people=n, hide_s=h, noise=z)
            for z in NOISE_LEVELS
            for n in (10, 30)
            for h in (1.0, 3.0)
        ]
    else:
        cells = [replace(sc, label=k) for k, sc in SCENARIOS.items()]
    res = run_cells(cells, seeds, trackers, a.workers)
    doc: dict[str, Any] = {
        "schema": f"zed-ps2-stress-{a.what}/v1",
        **provenance(seeds, trackers),
        "elapsed_s": round(time.time() - t0, 1),
    }
    if a.what == "grid":
        doc["axes"] = {"hide_s": list(GRID_HIDE), "people": list(GRID_PEOPLE), "noise": 0.15}
        for c in res:
            c["cell"] = f"n{c['scenario']['people']}-h{c['scenario']['hide_s']:g}"
    elif a.what == "noise":
        for c in res:
            s = c["scenario"]
            c["cell"] = f"z{s['noise']:g}-n{s['people']}-h{s['hide_s']:g}"
    else:
        for c in res:
            c["cell"] = c["scenario"]["label"]
    doc["cells"] = res
    name = {"grid": "stress_map", "noise": "stress_noise", "scenarios": "stress_scenarios"}[a.what]
    name += a.tag
    (a.out / f"{name}.json").write_text(json.dumps(doc, separators=(",", ":")))
    write_csv(a.out / f"{name}.csv", res, list(trackers), "cell")
    print(f"wrote {a.out / name}.json ({len(res)} cells, {time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Precompute the Scenarios tab overlays: one detection pass per clip (CPU YOLOX), then the
ByteTrack baseline and ours (BoT-SORT + visibility) on the *identical* detections.

Detection reads the **unblurred** clips (``data/demo/raw``, never committed or served), so the
face blur of the shown cuts cannot change a box or a tally; only boxes and numbers are written.

Writes web/public/scenarios/<clip>.json (small: rounded boxes per tracked frame) and
web/public/scenarios/index.json. These clips have NO ground truth: the ID-switch numbers are
the heuristic ``serve.analysis.suspected_switches`` count, illustrative only, never results.

    .venv/bin/python scripts/demo/precompute_scenarios.py [--only subway_turnstile ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.demo.guide import annotate  # noqa: E402
from serve.analysis import suspected_switches  # noqa: E402
from serve.clips import load_manifest  # noqa: E402
from serve.detector import MODELS, YoloxDetector, model_path  # noqa: E402
from serve.events import track_rows  # noqa: E402
from serve.pipeline import detect_video, probe  # noqa: E402
from tracker.config import load_config  # noqa: E402
from tracker.runner import run_sequence  # noqa: E402

DETECTOR = "yolox_s"
TRACKERS = {
    "baseline": ("ByteTrack", ROOT / "configs/baselines/bytetrack.yaml"),
    "ours": ("BoT-SORT + visibility (ours)", ROOT / "configs/ours/visibility_botsort.yaml"),
}
# scenario order, problem phrase, what to watch, stride, tracked-frame cap (CPU budget)
SCENARIOS: list[tuple[str, str, str, int, int | None]] = [
    (
        "man_entering_metro",
        "What tracking is",
        "One person, one box, one ID from entry to exit.",
        1,
        None,
    ),
    (
        "subway_turnstile",
        "Counting at a gate",
        "Each person keeps one ID as they pass the turnstiles.",
        1,
        None,
    ),
    (
        "milan_turnstiles",
        "Bunching at the barrier",
        "People squeeze together at the barrier; watch whether IDs survive the overlap.",
        2,
        None,
    ),
    (
        "mexico_city_entrance",
        "Recovering after occlusion",
        "People pass behind others; watch whether the same ID comes back.",
        1,
        600,
    ),
    (
        "shibuya_crossing",
        "Lookalikes in a dense crowd",
        "Many similar-looking people cross; the hardest case for keeping IDs apart.",
        2,
        600,
    ),
    (
        "rabat_stadium",
        "Event entrance (Fast Lane)",
        "Fans stream toward the entrance in portrait phone footage.",
        1,
        None,
    ),
]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def disagreements(rb: np.ndarray, ro: np.ndarray, min_iou: float = 0.5) -> list[dict[str, Any]]:
    """Frames where the IoU-matched pairing between baseline and ours IDs changes: one tracker
    keeps an ID while the other changes it (illustrative; no ground truth says who is right)."""
    pair_o: dict[int, int] = {}  # ours id -> last matched baseline id
    pair_b: dict[int, int] = {}
    out: list[dict[str, Any]] = []
    for f in np.unique(np.concatenate([rb[:, 0], ro[:, 0]])).astype(int):
        b, o = rb[rb[:, 0] == f], ro[ro[:, 0] == f]
        for orow in o:
            best, bid = min_iou, None
            for brow in b:
                v = iou(orow[2:6], brow[2:6])
                if v > best:
                    best, bid = v, int(brow[1])
            if bid is None:
                continue
            oid = int(orow[1])
            box = [round(float(x), 1) for x in orow[2:6]]
            if oid in pair_o and pair_o[oid] != bid:
                out.append(
                    {
                        "frame": int(f),
                        "kept": "ours",
                        "ours_id": oid,
                        "baseline_from": pair_o[oid],
                        "baseline_to": bid,
                        "box": box,
                    }
                )
            elif bid in pair_b and pair_b[bid] != oid:
                out.append(
                    {
                        "frame": int(f),
                        "kept": "baseline",
                        "baseline_id": bid,
                        "ours_from": pair_b[bid],
                        "ours_to": oid,
                        "box": box,
                    }
                )
            pair_o[oid], pair_b[bid] = bid, oid
    return out


def compact(rows: np.ndarray) -> dict[str, list[list[float]]]:
    frames: dict[str, list[list[float]]] = {}
    for r in rows:
        frames.setdefault(str(int(r[0])), []).append(
            [int(r[1]), *(round(float(v)) for v in r[2:6])]
        )
    return frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=ROOT / "data/demo/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "web/public/scenarios")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--annotate",
        action="store_true",
        help="only (re)compute the guided-replay block on existing JSON (no detection)",
    )
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.annotate:
        for cid, *_ in SCENARIOS:
            dst = a.out / f"{cid}.json"
            if not dst.exists() or (a.only and cid not in a.only):
                continue
            data = json.loads(dst.read_text())
            data["guide"] = annotate(data)
            dst.write_text(json.dumps(data, separators=(",", ":")))
            g = data["guide"]
            print(cid, g["gate"]["name"], "moments", [m["f"] for m in g["moments"]], flush=True)
        return 0
    manifest = {c.id: c for c in load_manifest()}
    det = YoloxDetector(model_path(DETECTOR), MODELS[DETECTOR][0], 0.1, threads=a.threads)
    det.name = DETECTOR  # type: ignore[attr-defined]
    index = []
    for cid, problem, caption, stride, cap in SCENARIOS:
        clip = manifest[cid]
        path = a.clips / clip.file
        dst = a.out / f"{cid}.json"
        if a.only and cid not in a.only:
            if dst.exists():
                index.append(json.loads(dst.read_text())["meta"])
            continue
        info = probe(path)
        cache, _ = detect_video(path, det, stride, cap)
        m = cache.meta
        runs: dict[str, Any] = {}
        rows: dict[str, np.ndarray] = {}
        for key, (label, cfg) in TRACKERS.items():
            r = track_rows(run_sequence(cache, load_config(cfg)).results)
            rows[key] = r
            sw = suspected_switches(r, m.frame_rate, m.seq_length)
            runs[key] = {
                "label": label,
                "config": str(cfg.relative_to(ROOT)),
                "id_switches": len(sw),
                "tracks": len(np.unique(r[:, 1])) if len(r) else 0,
                "frames": compact(r),
            }
        dis = disagreements(rows["baseline"], rows["ours"])
        meta = {
            "id": cid,
            "title": clip.title,
            "problem": problem,
            "caption": caption,
            "source": clip.page,
            "licence": clip.licence,
            "detector": DETECTOR,
            "kind": "illustrative",
            "width": info.width,
            "height": info.height,
            "src_fps": info.fps,
            "stride": stride,
            "frames_tracked": m.seq_length,
            "covered_s": round(m.seq_length * stride / info.fps, 2),
            "id_switches": {k: runs[k]["id_switches"] for k in runs},
            "tracks": {k: runs[k]["tracks"] for k in runs},
            "disagreements": len(dis),
        }
        dst.write_text(
            json.dumps(
                {
                    "meta": meta,
                    "runs": runs,
                    "disagreements": dis,
                    "guide": annotate({"meta": meta, "runs": runs, "disagreements": dis}),
                },
                separators=(",", ":"),
            )
        )
        print(
            cid,
            json.dumps(meta["id_switches"]),
            "dis",
            len(dis),
            "frames",
            m.seq_length,
            flush=True,
        )
        index.append(meta)
    order = [s[0] for s in SCENARIOS]
    index.sort(key=lambda x: order.index(x["id"]))
    (a.out / "index.json").write_text(json.dumps(index, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

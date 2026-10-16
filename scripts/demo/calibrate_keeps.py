"""Calibration evidence for scripts/demo/verify_keeps.py thresholds (writes
docs/results/keep_check_calibration.json).

* appearance: on the Pexels demo cuts (our tracker's output), "same" pairs = one ID's torso
  histogram over K frames vs the same ID 15 frames later with no gap in between; "different"
  pairs = vs another ID visible at those later frames. Neither set is ground truth (a continuous
  ID can itself be wrong), so this only shows how the two distributions separate.
* motion: on MOT17 (human labels), the motion/size check for every reappearance whose kept/swap
  verdict is known from ground truth.
"""

from __future__ import annotations

import json
import random
from typing import Any

import numpy as np

from scripts.demo.verify_keeps import (
    APP_MIN,
    PUB,
    ROOT,
    SIZE_MAX,
    K,
    Track,
    Video,
    bhattacharyya,
    hsv_hist,
    load_gz,
    mean_hist,
    motion_check,
    torso,
    tracks_of,
    verify_gt,
    viewer_tracks,
)

CLIPS = [
    "subway_turnstile",
    "mexico_city_entrance",
    "rabat_stadium",
    "milan_turnstiles",
    "shibuya_crossing",
]
LAG = 15


def appearance_pairs() -> tuple[list[float], list[float]]:
    random.seed(0)
    pos: list[float] = []
    neg: list[float] = []
    for clip in CLIPS:
        scn = json.loads((PUB / "scenarios" / f"{clip}.json").read_text())
        m = scn["meta"]
        v = Video(PUB / "demo" / f"{clip}.mp4")
        sc = v.W / m["width"]
        tr = tracks_of(scn["runs"]["ours"]["frames"])

        def hist(
            t: Track, fs: list[int], v: Video = v, stride: int = m["stride"], sc: float = sc
        ) -> np.ndarray | None:
            return mean_hist(
                [hsv_hist(v.crop((f - 1) * stride, torso(t[f], sc, v.W, v.H))) for f in fs]
            )

        for tid, t in tr.items():
            fs = sorted(t)
            for i in range(0, len(fs) - 2 * K - LAG, 20):
                w, w2 = fs[i : i + K], fs[i + K + LAG : i + 2 * K + LAG]
                if not w2:
                    continue
                if w2[-1] - w[0] == 2 * K + LAG - 1:
                    a, b = hist(t, w), hist(t, w2)
                    if a is not None and b is not None:
                        pos.append(bhattacharyya(a, b))
                others = [o for o in tr if o != tid and all(f in tr[o] for f in w2)]
                if others:
                    o = random.choice(others)
                    a, b = hist(t, w), hist(tr[o], w2)
                    if a is not None and b is not None:
                        neg.append(bhattacharyya(a, b))
    return pos, neg


def motion_on_gt() -> list[dict[str, Any]]:
    vd = ROOT / "viewer" / "data"
    rows = []
    for gtf in sorted(vd.glob("*.gt.json.gz")):
        clip = gtf.name.split(".")[0]
        gt = viewer_tracks(load_gz(gtf))
        for trk in ("bytetrack", "botsort"):
            c = load_gz(vd / f"{clip}.{trk}.json.gz")
            tr = viewer_tracks(c)
            for r in verify_gt(tr, gt, trk, c["fps"]):
                e, t, ra = motion_check(tr[r.id], r.last, r.back, c["fps"])
                rows.append(
                    {
                        "clip": clip,
                        "tracker": trk,
                        "gt_verdict": r.verdict,
                        "motion_ok": e <= t and ra <= SIZE_MAX,
                    }
                )
    return rows


def main() -> None:
    pos, neg = appearance_pairs()
    p, n = np.array(pos), np.array(neg)
    mot = motion_on_gt()
    out = {
        "appearance": {
            "same_pairs": len(pos),
            "different_pairs": len(neg),
            "app_min": APP_MIN,
            "same_below_threshold": round(float((p < APP_MIN).mean()), 3),
            "different_at_or_above_threshold": round(float((n >= APP_MIN).mean()), 3),
            "same_median": round(float(np.median(p)), 3),
            "different_median": round(float(np.median(n)), 3),
        },
        "motion_vs_gt": {
            v: {
                "n": sum(r["gt_verdict"] == v for r in mot),
                "motion_ok": sum(r["gt_verdict"] == v and r["motion_ok"] for r in mot),
            }
            for v in ("kept", "swap", "unknown")
        },
    }
    (ROOT / "docs" / "results" / "keep_check_calibration.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

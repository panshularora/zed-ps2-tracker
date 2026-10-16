"""Kept-vs-swap check for "came back with the same number" events.

When a tracker ID disappears for some frames and the same ID is drawn again, the UI used to say
"kept #N". That is only true if the box came back on the *same person*. A tracker can instead hand
the number to someone else (identity transfer / false keep). This module decides, per reappearance:

* Pexels clips (no ground truth): a HEURISTIC check, not ground truth. Two independent signals:
  1. appearance: HSV colour histogram of the torso region of the box (faces are blurred in the
     demo cuts, torso colour is not), averaged over up to K visible frames before the gap and up
     to K after it; similarity = Bhattacharyya coefficient (1 = identical histograms).
  2. motion plausibility: constant-velocity extrapolation of the box bottom-centre from the last
     frames before the gap, compared with where the box reappears; error in units of the box
     height, with a tolerance that grows with the gap length; plus a box-height ratio check.
  Both must pass for "kept"; otherwise "swap" (shown as "possible swap: #N given to someone
  else", counted as uncertain).
* MOT17 / MOT20 (human labels): match the box before and after the gap to the ground-truth box
  (IoU >= 0.5); same labelled person = "kept", different = "swap", no match = "unknown".

Usage: python -m scripts.demo.verify_keeps   (writes web/public/verify/<clip>.json)
"""

from __future__ import annotations

import gzip
import json
import subprocess
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "web" / "public"
OUT = PUB / "verify"

K = 5  # frames each side used for the appearance histogram
H_BINS, S_BINS, V_BINS = 12, 4, 3
APP_MIN = 0.80  # Bhattacharyya coefficient: below this the before/after torso colours disagree
MOT_BASE = 0.6  # allowed extrapolation error, in box heights, for a 0 s gap ...
MOT_PER_S = 0.9  # ... plus this many box heights per second of gap
SIZE_MAX = 1.6  # max ratio between box heights before and after (perspective + detector jitter)
GT_IOU = 0.5

XYWH = tuple[float, float, float, float]
Track = dict[int, XYWH]


@dataclass
class Reappearance:
    id: int
    run: str
    last: int  # last tracked frame before the gap (1-based)
    back: int  # first tracked frame after the gap
    gap_frames: int
    gap_s: float
    verdict: str = "unknown"  # kept | swap | unknown
    app: float | None = None
    motion_err: float | None = None
    motion_tol: float | None = None
    size_ratio: float | None = None
    gt_before: int | None = None
    gt_after: int | None = None
    reasons: list[str] = field(default_factory=list)
    occluder: int | None = None


def tracks_of(frames: dict[str, list[list[float]]]) -> dict[int, Track]:
    out: dict[int, Track] = {}
    for f, bs in frames.items():
        for b in bs:
            out.setdefault(int(b[0]), {})[int(f)] = (
                float(b[1]),
                float(b[2]),
                float(b[3]),
                float(b[4]),
            )
    return out


def gaps_of(track: Track) -> list[tuple[int, int]]:
    fs = sorted(track)
    return [(a, b) for a, b in pairwise(fs) if b > a + 1]


# ---- motion -------------------------------------------------------------------------------
def foot(b: XYWH) -> tuple[float, float]:
    return b[0] + b[2] / 2, b[1] + b[3]


def motion_check(track: Track, last: int, back: int, fps: float) -> tuple[float, float, float]:
    """(error in box heights, tolerance, height ratio >= 1) for reappearing at `back`."""
    before = [f for f in sorted(track) if f <= last][-10:]
    pa = foot(track[last])
    if len(before) >= 2:
        f0 = before[0]
        p0 = foot(track[f0])
        v = ((pa[0] - p0[0]) / (last - f0), (pa[1] - p0[1]) / (last - f0))
    else:
        v = (0.0, 0.0)
    n = back - last
    pred = (pa[0] + v[0] * n, pa[1] + v[1] * n)
    pb = foot(track[back])
    h = max(track[last][3], 1.0)
    err = float(np.hypot(pb[0] - pred[0], pb[1] - pred[1]) / h)
    tol = MOT_BASE + MOT_PER_S * (n / fps)
    ratio = max(track[back][3], 1.0) / h
    return err, tol, float(max(ratio, 1 / ratio))


# ---- appearance ---------------------------------------------------------------------------
def torso(b: XYWH, scale: float, W: int, H: int) -> tuple[int, int, int, int]:
    x, y, w, h = (v * scale for v in b)
    x0, x1 = x + 0.2 * w, x + 0.8 * w
    y0, y1 = y + 0.18 * h, y + 0.6 * h
    return (max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1)))


def hsv_hist(rgb: np.ndarray) -> np.ndarray | None:
    """Normalised H x S x V histogram of an RGB crop (uint8, HxWx3)."""
    if rgb.size == 0 or rgb.shape[0] < 4 or rgb.shape[1] < 4:
        return None
    from PIL import Image

    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.int32)
    h = hsv[..., 0] * H_BINS // 256
    s = hsv[..., 1] * S_BINS // 256
    v = hsv[..., 2] * V_BINS // 256
    idx = (h * S_BINS + s) * V_BINS + v
    hist = np.bincount(idx.ravel(), minlength=H_BINS * S_BINS * V_BINS).astype(np.float64)
    out: np.ndarray = hist / hist.sum()
    return out


def bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.sqrt(a * b)))


def mean_hist(hs: list[np.ndarray | None]) -> np.ndarray | None:
    ok = [h for h in hs if h is not None]
    if not ok:
        return None
    m = np.mean(ok, axis=0)
    out: np.ndarray = m / m.sum()
    return out


class Video:
    """Decoded demo cut (all frames in memory; clips are short 720p cuts)."""

    def __init__(self, path: Path):
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        w, h = (int(v) for v in probe.split(",")[:2])
        # decode at half of the 720p cut to keep memory small (torso colour survives easily)
        self.W, self.H = w // 2 // 2 * 2, h // 2 // 2 * 2
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                f"scale={self.W}:{self.H}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            stdout=subprocess.PIPE,
        )
        n = self.W * self.H * 3
        frames = []
        assert proc.stdout is not None
        while chunk := proc.stdout.read(n):
            if len(chunk) < n:
                break
            frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(self.H, self.W, 3))
        proc.wait()
        self.frames = frames

    def crop(self, idx: int, box: tuple[int, int, int, int]) -> np.ndarray:
        idx = min(max(idx, 0), len(self.frames) - 1)
        x0, y0, x1, y1 = box
        return self.frames[idx][y0:y1, x0:x1]


def appearance(
    video: Video, track: Track, last: int, back: int, stride: int, scale: float
) -> float | None:
    before = [f for f in sorted(track) if f <= last][-K:]
    after = [f for f in sorted(track) if f >= back][:K]

    def hist(fs: list[int]) -> np.ndarray | None:
        return mean_hist(
            [
                hsv_hist(video.crop((f - 1) * stride, torso(track[f], scale, video.W, video.H)))
                for f in fs
            ]
        )

    a, b = hist(before), hist(after)
    return None if a is None or b is None else bhattacharyya(a, b)


def decide(app: float | None, err: float, tol: float, ratio: float) -> tuple[str, list[str]]:
    reasons = []
    if app is None:
        reasons.append("no usable crop")
    elif app < APP_MIN:
        reasons.append(f"appearance differs ({app:.2f} < {APP_MIN})")
    if err > tol:
        reasons.append(f"path implausible ({err:.2f} > {tol:.2f} box heights)")
    if ratio > SIZE_MAX:
        reasons.append(f"size jump x{ratio:.2f} > {SIZE_MAX}")
    return ("kept" if not reasons else "swap"), reasons


def occluder_of(tracks: dict[int, Track], tid: int, last: int, back: int) -> int | None:
    """The other ID whose box covers the hidden person's interpolated box most during the gap."""
    a, b = tracks[tid][last], tracks[tid][back]
    score: dict[int, float] = {}
    for f in range(last + 1, back):
        k = (f - last) / (back - last)
        g = tuple(a[i] + (b[i] - a[i]) * k for i in range(4))
        for oid, t in tracks.items():
            o = t.get(f)
            if oid == tid or o is None:
                continue
            iw = max(0.0, min(g[0] + g[2], o[0] + o[2]) - max(g[0], o[0]))
            ih = max(0.0, min(g[1] + g[3], o[1] + o[3]) - max(g[1], o[1]))
            score[oid] = score.get(oid, 0.0) + iw * ih / max(g[2] * g[3], 1.0)
    if not score:
        return None
    best = max(score, key=lambda i: score[i])
    return best if score[best] / max(back - last - 1, 1) >= 0.3 else None


def verify_pexels(scn: dict[str, Any], video: Video | None) -> list[Reappearance]:
    m = scn["meta"]
    fps = m["src_fps"] / m["stride"]
    out: list[Reappearance] = []
    for run in ("ours", "baseline"):
        tracks = tracks_of(scn["runs"][run]["frames"])
        for tid, tr in sorted(tracks.items()):
            for last, back in gaps_of(tr):
                r = Reappearance(
                    tid, run, last, back, back - last - 1, round((back - last - 1) / fps, 2)
                )
                err, tol, ratio = motion_check(tr, last, back, fps)
                app = (
                    appearance(video, tr, last, back, m["stride"], video.W / m["width"])
                    if video
                    else None
                )
                r.app = None if app is None else round(app, 3)
                r.motion_err, r.motion_tol, r.size_ratio = (
                    round(err, 3),
                    round(tol, 3),
                    round(ratio, 3),
                )
                r.verdict, r.reasons = decide(app, err, tol, ratio)
                r.occluder = occluder_of(tracks, tid, last, back)
                out.append(r)
    return out


# ---- ground truth (MOTChallenge) ----------------------------------------------------------
def iou(a: XYWH, b: XYWH) -> float:
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


def gt_id(gt: dict[int, Track], f: int, box: XYWH) -> int | None:
    best, who = GT_IOU, None
    for gid, t in gt.items():
        g = t.get(f)
        if g is not None and (v := iou(g, box)) >= best:
            best, who = v, gid
    return who


def verify_gt(
    tracks: dict[int, Track], gt: dict[int, Track], run: str, fps: float
) -> list[Reappearance]:
    out = []
    for tid, tr in sorted(tracks.items()):
        for last, back in gaps_of(tr):
            r = Reappearance(
                tid, run, last, back, back - last - 1, round((back - last - 1) / fps, 2)
            )
            r.gt_before, r.gt_after = gt_id(gt, last, tr[last]), gt_id(gt, back, tr[back])
            if r.gt_before is None or r.gt_after is None:
                r.verdict, r.reasons = (
                    "unknown",
                    ["box not matched to a labelled person (IoU < 0.5)"],
                )
            elif r.gt_before == r.gt_after:
                r.verdict = "kept"
            else:
                r.verdict, r.reasons = (
                    "swap",
                    [f"labelled person {r.gt_before} before, {r.gt_after} after"],
                )
            r.occluder = occluder_of(tracks, tid, last, back)
            out.append(r)
    return out


def per_person(
    tracks: dict[int, Track], gt: dict[int, Track], fps: float, min_s: float = 0.5
) -> dict[str, int]:
    """Same denominator for every tracker: each labelled person seen >= ``min_s``.

    For every frame of that person, the tracker box with the best IoU (>= 0.5) gives its number.
    kept = exactly one number from first to last frame (gaps allowed); changed = two or more;
    missed = never matched.
    """
    by_frame: dict[int, list[tuple[int, XYWH]]] = {}
    for tid, tr in tracks.items():
        for f, b in tr.items():
            by_frame.setdefault(f, []).append((tid, b))
    out = {"labelled": 0, "kept": 0, "changed": 0, "missed": 0}
    for _gid, g in gt.items():
        if len(g) < min_s * fps:
            continue
        out["labelled"] += 1
        ids: set[int] = set()
        for f, gb in g.items():
            best, who = GT_IOU, None
            for tid, b in by_frame.get(f, []):
                if (v := iou(gb, b)) >= best:
                    best, who = v, tid
            if who is not None:
                ids.add(who)
        out["kept" if len(ids) == 1 else "changed" if ids else "missed"] += 1
    return out


def load_gz(p: Path) -> Any:
    with gzip.open(p) as fh:
        return json.load(fh)


def viewer_tracks(c: dict[str, Any]) -> dict[int, Track]:
    return {
        int(i): {int(r[0]): (r[1], r[2], r[3], r[4]) for r in rows}
        for i, rows in c["tracks"].items()
    }


METHOD = {
    "pexels": (
        "heuristic: torso HSV histogram (Bhattacharyya) + constant-velocity path"
        " + box-height ratio; not ground truth"
    ),
    "gt": (
        "ground truth: box before and after the gap matched to the human-labelled person"
        " (IoU >= 0.5)"
    ),
}
THRESHOLDS = {
    "app_min": APP_MIN,
    "motion_base": MOT_BASE,
    "motion_per_s": MOT_PER_S,
    "size_max": SIZE_MAX,
    "k_frames": K,
    "gt_iou": GT_IOU,
}


def summarise(rs: list[Reappearance]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in rs:
        s = out.setdefault(r.run, {"kept": 0, "swap": 0, "unknown": 0})
        s[r.verdict] += 1
    return out


def write(
    clip: str, kind: str, rs: list[Reappearance], people: dict[str, Any] | None = None
) -> dict[str, Any]:
    doc = {
        "clip": clip,
        "method": kind,
        "method_text": METHOD[kind],
        "thresholds": THRESHOLDS,
        "summary": summarise(rs),
        "events": [asdict(r) for r in rs],
    }
    if people is not None:
        doc["people"] = people
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{clip}.json").write_text(json.dumps(doc, separators=(",", ":")))
    return doc


def main() -> None:
    import sys

    gt_only = "--gt-only" in sys.argv[1:]
    idx = [] if gt_only else json.loads((PUB / "scenarios" / "index.json").read_text())
    for m in idx:
        scn = json.loads((PUB / "scenarios" / f"{m['id']}.json").read_text())
        mp4 = PUB / "demo" / f"{m['id']}.mp4"
        video = Video(mp4) if mp4.exists() else None
        doc = write(m["id"], "pexels", verify_pexels(scn, video))
        print(m["id"], doc["summary"])
    vd = ROOT / "viewer" / "data"
    for gtf in sorted(vd.glob("*.gt.json.gz")):
        clip = gtf.name.split(".")[0]
        gt = viewer_tracks(load_gz(gtf))
        rs: list[Reappearance] = []
        people: dict[str, Any] = {}
        # "ours" = BoT-SORT + visibility (configs/ours/visibility_botsort.yaml) when exported
        ours = "vis_botsort" if (vd / f"{clip}.vis_botsort.json.gz").exists() else "botsort"
        for tr in ("bytetrack", ours):
            p = vd / f"{clip}.{tr}.json.gz"
            if p.exists():
                c = load_gz(p)
                run = "baseline" if tr == "bytetrack" else "ours"
                rs += verify_gt(viewer_tracks(c), gt, run, c["fps"])
                people[run] = per_person(viewer_tracks(c), gt, c["fps"])
        doc = write(clip, "gt", rs, people)
        print(clip, "people", people, "returns", doc["summary"])


if __name__ == "__main__":
    main()

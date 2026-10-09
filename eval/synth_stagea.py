"""Synthetic Stage A bundle for testing ingest → phase3 end to end without a GPU.

    python -m eval.synth_stagea --root /tmp/zed-synth        # fake ZED_ROOT + bring_back + tars

It builds a fake ``ZED_ROOT`` exactly as the DGX scripts lay it out (``data/<dataset>/...`` with
``seqinfo.ini`` + ``gt/gt.txt``, ``logs/``, ``weights/SHA256SUMS``) and writes the caches with the
**real** Stage A writer (:func:`models.stage_a.run_many`: ``process_sequence`` → ``save_cache``,
``stage_a_log.jsonl``), using a detector and an embedder that read the synthetic gt instead of
pixels. Then it runs the real ``models.bringback`` (integrity check + bundle) and
``models.pack`` (tars). Only the pixels are fake; every file format is the production one.

The scenes: people walk across the frame in both directions at different depths, so they occlude
each other; the detector drops some boxes, jitters the rest, scores them by visibility and adds
low-score false positives; the embedder gives each identity a fixed 512-d direction plus noise.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

# (dataset, detector label, split, sequences, saw_eval_half)
LAYOUT = [
    ("MOT17", "ablation", "train", ["MOT17-90-FRCNN", "MOT17-91-FRCNN"], False),
    ("MOT20", "sparsetrack_ablation", "train", ["MOT20-90"], False),
    ("MOT20", "mot17full_xdomain", "train", ["MOT20-90"], False),
    ("MOT20", "mot20full_SAW_EVAL_HALF", "train", ["MOT20-90"], True),
    ("DanceTrack", "dancetrack", "val", ["dancetrack9001"], False),
]
W, H, FPS = 960, 540, 25.0


def make_gt(seed: int, n_frames: int, n_people: int) -> npt.NDArray[np.float64]:
    """Rows ``frame, id, x, y, w, h, 1, 1, visibility`` (MOTChallenge gt)."""
    rng = np.random.default_rng(seed)
    rows = []
    people = []
    for pid in range(1, n_people + 1):
        bottom = rng.uniform(0.55, 0.98) * H
        h = 60 + 140 * (bottom / H)
        direction = 1 if pid % 2 else -1
        speed = rng.uniform(4.0, 9.0) * direction
        start = rng.integers(1, n_frames // 3)
        x0 = -h * 0.4 if direction > 0 else W
        people.append((pid, bottom, h, speed, start, x0))
    for f in range(1, n_frames + 1):
        boxes = []
        for pid, bottom, h, speed, start, x0 in people:
            if f < start:
                continue
            w = h * 0.4
            x = x0 + speed * (f - start)
            if x + w < 0 or x > W:
                continue
            boxes.append([f, pid, x, bottom - h, w, h])
        b = np.array(boxes, float).reshape(-1, 6)
        vis = np.ones(len(b))
        for i in range(len(b)):  # covered by anyone whose bottom edge is lower (closer)
            xi, yi, wi, hi = b[i, 2:6]
            cover = 0.0
            for j in range(len(b)):
                if j == i or b[j, 3] + b[j, 5] <= yi + hi:
                    continue
                ix = max(0.0, min(xi + wi, b[j, 2] + b[j, 4]) - max(xi, b[j, 2]))
                iy = max(0.0, min(yi + hi, b[j, 3] + b[j, 5]) - max(yi, b[j, 3]))
                cover = max(cover, ix * iy / (wi * hi))
            inside = max(0.0, min(xi + wi, W) - max(xi, 0.0)) / wi
            vis[i] = max(0.0, min(1.0 - cover, inside))
        for r, v in zip(b, vis, strict=True):
            rows.append([*r, 1, 1, round(float(v), 3)])
    return np.array(rows, float)


@dataclass
class GtSource:
    name: str
    seq_length: int
    frame_rate: float = FPS
    width: int = W
    height: int = H

    def __iter__(self) -> Iterator[tuple[int, npt.NDArray[np.uint8]]]:
        for f in range(1, self.seq_length + 1):
            img = np.zeros((1, 2, 3), np.uint8)  # stands in for the frame; carries its index
            img[0, 0, :2] = (f // 256, f % 256)
            yield f, img


def _fid(img: npt.NDArray[np.uint8]) -> int:
    return int(img[0, 0, 0]) * 256 + int(img[0, 0, 1])


@dataclass
class GtDetector:
    gt: npt.NDArray[np.float64]
    seed: int
    name: str = "yolox_x"
    last_ids: dict[int, npt.NDArray[np.int64]] = field(default_factory=dict)

    def info(self) -> dict[str, object]:
        return {"name": "yolox_x", "exp": "synthetic", "weights": "synthetic.pth.tar",
                "weights_sha256": "0" * 64, "test_size": [800, 1440], "conf_floor": 0.05,
                "nms_thresh": 0.7, "trained_on": "synthetic gt (tests only)"}  # fmt: skip

    def __call__(
        self, images: list[npt.NDArray[np.uint8]]
    ) -> list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
        out = []
        for img in images:
            f = _fid(img)
            rng = np.random.default_rng(self.seed * 100003 + f)
            g = self.gt[self.gt[:, 0] == f]
            keep = (g[:, 8] > 0.15) & (rng.random(len(g)) > 0.08)
            g = g[keep]
            tlwh = g[:, 2:6] + rng.normal(0, 2.0, (len(g), 4))
            tlwh[:, 2:] = np.maximum(tlwh[:, 2:], 4.0)
            score = np.clip(0.25 + 0.7 * g[:, 8] + rng.normal(0, 0.05, len(g)), 0.06, 0.99)
            ids = g[:, 1].astype(np.int64)
            n_fp = int(rng.integers(0, 3))
            fp = np.column_stack([rng.uniform(0, W - 40, n_fp), rng.uniform(0, H - 100, n_fp),
                                  np.full(n_fp, 40.0), np.full(n_fp, 100.0)])  # fmt: skip
            tlwh = np.vstack([tlwh, fp])
            score = np.concatenate([score, rng.uniform(0.05, 0.4, n_fp)])
            ids = np.concatenate([ids, np.full(n_fp, -1)])
            # clip to the image like the real detector; drop zero-area boxes
            x1 = np.clip(tlwh[:, 0], 0, W)
            y1 = np.clip(tlwh[:, 1], 0, H)
            x2 = np.clip(tlwh[:, 0] + tlwh[:, 2], 0, W)
            y2 = np.clip(tlwh[:, 1] + tlwh[:, 3], 0, H)
            ok = (x2 - x1 > 1) & (y2 - y1 > 1)
            box = np.column_stack([x1, y1, x2 - x1, y2 - y1])[ok].astype(np.float32)
            self.last_ids[f] = ids[ok]
            out.append((box, score[ok].astype(np.float32)))
        return out


@dataclass
class GtEmbedder:
    detector: GtDetector
    seed: int
    name: str = "osnet_ain_x1_0"
    dim: int = 512

    def info(self) -> dict[str, object]:
        return {"name": "osnet_ain_x1_0", "weights": "synthetic.pth", "weights_sha256": "0" * 64,
                "input_hw": [256, 128], "dim": 512}  # fmt: skip

    def __call__(
        self, image: npt.NDArray[np.uint8], tlwh: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float16]:
        f = _fid(image)
        ids = self.detector.last_ids[f]
        rng = np.random.default_rng(self.seed * 7919 + f)
        out = np.empty((len(ids), self.dim), np.float32)
        for k, pid in enumerate(ids):
            base = np.random.default_rng(1000 + int(pid) if pid > 0 else 10**6 + k).normal(
                size=self.dim
            )
            out[k] = base / np.linalg.norm(base) + rng.normal(0, 0.03, self.dim)
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        return out.astype(np.float16)


def write_sequence(seq_dir: Path, gt: npt.NDArray[np.float64], n_frames: int) -> None:
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    (seq_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq_dir.name}\nimDir=img1\nframeRate={int(FPS)}\n"
        f"seqLength={n_frames}\nimWidth={W}\nimHeight={H}\nimExt=.jpg\n"
    )
    lines = [f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},1,1,{r[8]:.3f}"
             for r in gt]  # fmt: skip
    (seq_dir / "gt" / "gt.txt").write_text("\n".join(lines) + "\n")


def build(root: Path, n_frames: int = 80, n_people: int = 10) -> Path:
    """Create the fake ZED_ROOT, run Stage A's writer, finalize and pack. Returns the tar dir."""
    from models import bringback, pack
    from models.stage_a import run_many

    seqs: dict[tuple[str, str], npt.NDArray[np.float64]] = {}
    for k, (ds, _, split, names, _) in enumerate(LAYOUT):
        for i, name in enumerate(names):
            if (ds, name) in seqs:
                continue
            gt = make_gt(seed=17 * k + i, n_frames=n_frames, n_people=n_people)
            seqs[(ds, name)] = gt
            write_sequence(root / "data" / ds / split / name, gt, n_frames)
    for k, (ds, label, split, names, saw) in enumerate(LAYOUT):
        srcs = [GtSource(n, n_frames) for n in names]
        for src in srcs:
            det = GtDetector(seqs[(ds, src.name)], seed=k)
            run_many([src], det, GtEmbedder(det, seed=k), root / "cache" / ds / label / split,
                     dataset=ds, split=split, detector_saw_eval_half=saw,
                     batch_size=8)  # fmt: skip
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "run_mot17_synthetic.log").write_text("synthetic run\n")
    (root / "weights").mkdir(parents=True, exist_ok=True)
    (root / "weights" / "SHA256SUMS").write_text("0" * 64 + "  ./synthetic.pth.tar\n")
    if bringback.main(["--root", str(root)]) != 0:
        raise RuntimeError("bringback reported integrity errors on the synthetic bundle")
    if pack.main(["--root", str(root)]) != 0:
        raise RuntimeError("pack failed")
    return root / "bring_back_tars"


def stage_a_one(root: Path, dataset: str, label: str, saw_eval_half: bool = False,
                n_frames: int = 80, n_people: int = 10) -> Path:  # fmt: skip
    """One dataset of LAYOUT under any detector label, as one ``run.sh <dataset>`` would leave
    it (``data/`` with gt + caches + ``stage_a_log.jsonl``). The stand-in for Stage A in the
    CPU tests of ``run.sh all`` (tests/test_run_all.py)."""
    from models.stage_a import run_many

    k, (ds, _, split, names, _) = next((k, x) for k, x in enumerate(LAYOUT) if x[0] == dataset)
    out = root / "cache" / ds / label / split
    for i, name in enumerate(names):
        gt = make_gt(seed=17 * k + i, n_frames=n_frames, n_people=n_people)
        write_sequence(root / "data" / ds / split / name, gt, n_frames)
        det = GtDetector(gt, seed=k)
        run_many([GtSource(name, n_frames)], det, GtEmbedder(det, seed=k), out, dataset=ds,
                 split=split, detector_saw_eval_half=saw_eval_half, batch_size=8)  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.synth_stagea", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--root", type=Path, required=True, help="fake ZED_ROOT to create")
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--people", type=int, default=10)
    p.add_argument("--only", default=None, help="one dataset (MOT17, MOT20, DanceTrack) only")
    p.add_argument("--label", default=None, help="with --only: the detector label folder")
    p.add_argument("--saw-eval-half", action="store_true")
    a = p.parse_args(argv)
    if a.only:
        out = stage_a_one(a.root, a.only, a.label or "synthetic", a.saw_eval_half, a.frames,
                          a.people)  # fmt: skip
        print(out)
        return 0
    tars = build(a.root, a.frames, a.people)
    print(json.dumps(sorted(p.name for p in tars.glob("*.tar"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

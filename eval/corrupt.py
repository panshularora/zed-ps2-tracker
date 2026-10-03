"""Synthetic detections from ground truth with controlled, visibility-dependent corruption.

Used by the go/no-go check (``eval/go_no_go.py``) to test the tracker logic apart from detector
quality. Every gt pedestrian box (class 1, and conf 1 on MOT17) in every frame goes through:

* **drop** with probability ``p_drop0 + p_drop_occ · (1 − v)``, where v is the gt visibility. A
  hidden person is missed more often, as with a real detector.
* **jitter**: centre ``+ N(0, σ_pos · (w, h))``, size ``× exp(N(0, σ_size))``.
* **score** ``clip(0.5 + 0.5·v + N(0, σ_score), 0.05, 1)``. Occluded boxes score low and so go to
  the low-score (BYTE) stage.
* **false positives**: Poisson(``fp_rate``) per frame. Each has the size of a random gt box of
  that frame, a uniform position in the image and a score drawn from U(0.1, 0.6).

All randomness comes from ``numpy.random.default_rng(seed + sequence index)``.

    python -m eval.corrupt --dataset MOT17 --root data/MOT17 --seed 0 --out /tmp/corrupt/MOT17_s0
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval.splits import find_sequences
from tracker.io import CacheMeta, SequenceCache, read_seqinfo, save_cache


@dataclass(frozen=True)
class Corruption:
    p_drop0: float = 0.05
    p_drop_occ: float = 0.5
    sigma_pos: float = 0.04
    sigma_size: float = 0.04
    sigma_score: float = 0.1
    fp_rate: float = 2.0


def corrupt_sequence(gt: np.ndarray, info: object, dataset: str, rng: np.random.Generator,
                     c: Corruption) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Return ``(frame, tlwh, score)`` arrays for one sequence's gt rows."""
    ped = gt[:, 7] == 1
    if dataset.upper() == "MOT17":
        ped &= gt[:, 6] == 1
    g = gt[ped]
    vis = np.clip(g[:, 8], 0.0, 1.0)
    keep = rng.uniform(size=len(g)) >= c.p_drop0 + c.p_drop_occ * (1.0 - vis)
    g, vis = g[keep], vis[keep]
    w, h = g[:, 4], g[:, 5]
    cx = g[:, 2] + w / 2 + rng.normal(0, c.sigma_pos, len(g)) * w
    cy = g[:, 3] + h / 2 + rng.normal(0, c.sigma_pos, len(g)) * h
    w2 = w * np.exp(rng.normal(0, c.sigma_size, len(g)))
    h2 = h * np.exp(rng.normal(0, c.sigma_size, len(g)))
    tlwh = np.c_[cx - w2 / 2, cy - h2 / 2, w2, h2]
    score = np.clip(0.5 + 0.5 * vis + rng.normal(0, c.sigma_score, len(g)), 0.05, 1.0)
    frames = [g[:, 0]]
    boxes, scores = [tlwh], [score]
    W, H = info.im_width, info.im_height  # type: ignore[attr-defined]
    for f in np.unique(gt[ped, 0]):
        n = rng.poisson(c.fp_rate)
        if n == 0:
            continue
        sizes = gt[ped & (gt[:, 0] == f)][:, 4:6]
        pick = sizes[rng.integers(0, len(sizes), n)]
        xy = rng.uniform(0, 1, (n, 2)) * np.maximum([W, H] - pick, 1.0)
        frames.append(np.full(n, f))
        boxes.append(np.c_[xy, pick])
        scores.append(rng.uniform(0.1, 0.6, n))
    fr = np.concatenate(frames)
    order = np.argsort(fr, kind="stable")
    return (fr[order].astype(np.int32), np.concatenate(boxes)[order].astype(np.float32),
            np.concatenate(scores)[order].astype(np.float32))  # fmt: skip


def write_corrupted(root: Path, dataset: str, seed: int, out: Path,
                    c: Corruption | None = None) -> list[Path]:  # fmt: skip
    c = c or Corruption()
    paths = []
    for i, seq in enumerate(find_sequences(root, dataset)):
        info = read_seqinfo(seq)
        gt = np.loadtxt(seq / "gt" / "gt.txt", delimiter=",", ndmin=2)
        rng = np.random.default_rng(seed + i)
        fr, tlwh, sc = corrupt_sequence(gt, info, dataset, rng, c)
        meta = CacheMeta(
            sequence=seq.name, dataset=dataset, split="train", frame_rate=info.frame_rate,
            seq_length=info.seq_length, im_width=info.im_width, im_height=info.im_height,
            detector={"name": "corrupted_gt", "seed": seed, **asdict(c)},
        )  # fmt: skip
        paths.append(save_cache(out / f"{seq.name}.npz", SequenceCache(meta, fr, tlwh, sc)))
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.corrupt", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--dataset", required=True, choices=["MOT17", "MOT20"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    for path in write_corrupted(a.root, a.dataset, a.seed, a.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

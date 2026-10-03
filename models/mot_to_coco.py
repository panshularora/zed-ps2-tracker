"""MOT gt → COCO json for the half split (ByteTrack convention), boxes clipped to the image.

``python -m models.mot_to_coco --root DATA/MOT20 --split train_half``

Frames: with N frames, ``train_half`` = 1 .. N//2 + 1 and ``val_half`` = N//2 + 2 .. N
(ByteTrack's ``convert_mot17_to_coco.py``). Kept annotations: ``consider == 1`` and class 1
(pedestrian), as ByteTrack; boxes are clipped to the image as ByteTrack's README requires for
MOT20 training. ``file_name`` is ``train/<seq>/img1/%06d.jpg`` relative to ``--root``.
"""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path

import numpy as np

from eval.splits import half_range

__all__ = ["convert", "half_range"]


def convert(root: Path, split: str) -> dict[str, object]:
    images, anns = [], []
    img_id = ann_id = 0
    for seq in sorted((root / "train").iterdir()):
        if not (seq / "seqinfo.ini").exists():
            continue
        cp = configparser.ConfigParser()
        cp.read(seq / "seqinfo.ini")
        s = cp["Sequence"]
        n, w, h = int(s["seqLength"]), int(s["imWidth"]), int(s["imHeight"])
        lo, hi = half_range(n, split)
        gt = np.loadtxt(seq / "gt" / "gt.txt", delimiter=",", ndmin=2)
        first = img_id + 1
        for f in range(lo, hi + 1):
            img_id += 1
            images.append({"file_name": f"train/{seq.name}/img1/{f:06d}.jpg", "id": img_id,
                           "frame_id": f - lo + 1, "height": h, "width": w,
                           "video_id": seq.name})  # fmt: skip
        keep = (gt[:, 0] >= lo) & (gt[:, 0] <= hi) & (gt[:, 6] == 1) & (gt[:, 7] == 1)
        for row in gt[keep]:
            x1, y1 = max(row[2], 0.0), max(row[3], 0.0)
            x2, y2 = min(row[2] + row[4], w), min(row[3] + row[5], h)
            if x2 <= x1 or y2 <= y1:
                continue
            ann_id += 1
            anns.append({"id": ann_id, "category_id": 1, "image_id": first + int(row[0]) - lo,
                         "track_id": int(row[1]), "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": (x2 - x1) * (y2 - y1), "iscrowd": 0, "conf": 1.0})  # fmt: skip
    return {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "pedestrian"}]}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m models.mot_to_coco")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", choices=["train_half", "val_half", "train"], required=True)
    a = p.parse_args(argv)
    out = a.root / "annotations" / f"{a.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    d = convert(a.root, a.split)
    out.write_text(json.dumps(d))
    print(f"{out}: {len(d['images'])} images, {len(d['annotations'])} boxes")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

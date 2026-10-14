"""Normalise third-party dataset layouts into MOT-style sequence folders.

``python -m models.prepare sompt22 --root DATA/SOMPT22 [--fps 30]``

SOMPT22's archive layout could not be inspected from the CPU box (the Drive folder refused
anonymous listing), so this is deliberately defensive: every directory with an ``img1/`` of
``%06d`` images becomes a sequence; sequences with ``gt/gt.txt`` go to ``train``, the rest to
``test``. ``seqinfo.ini`` is kept if present, otherwise written from the images (frame rate
from ``--fps``). Output: ``<root>/_seqs/<split>/<seq>/`` with symlinks, so nothing is copied.
"""

from __future__ import annotations

import argparse
import configparser
import re
from pathlib import Path

_FRAME = re.compile(r"^(\d{6})\.(jpg|jpeg|png)$", re.IGNORECASE)


def _image_info(img_dir: Path) -> tuple[int, str, int, int]:
    frames = sorted(p for p in img_dir.iterdir() if _FRAME.match(p.name))
    if not frames:
        raise ValueError(f"{img_dir}: no %06d.jpg/png frames")
    ids = [int(_FRAME.match(p.name).group(1)) for p in frames]  # type: ignore[union-attr]
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError(f"{img_dir}: frames are not 000001..{len(ids):06d}")
    import cv2

    im = cv2.imread(str(frames[0]))
    if im is None:
        raise ValueError(f"cannot read {frames[0]}")
    return len(ids), frames[0].suffix, int(im.shape[1]), int(im.shape[0])


def _seq_dirs(root: Path) -> list[Path]:
    """Sequence folders that contain ``img1/``. Follows a ``train/`` / ``test/`` symlink."""
    found: set[Path] = set()
    for split in ("train", "test"):
        base = root / split
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and (child / "img1").is_dir() and "_seqs" not in child.parts:
                found.add(child)
    for p in root.rglob("img1"):
        if p.is_dir() and "_seqs" not in p.parts:
            found.add(p.parent)
    return sorted(found, key=lambda p: (p.parent.name, p.name))


def prepare_mot_like(root: Path, fps: float) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"train": [], "test": []}
    seq_dirs = _seq_dirs(root)
    if not seq_dirs:
        raise SystemExit(f"no <seq>/img1 folders under {root}; check the extracted archives")
    for seq in seq_dirs:
        split = "train" if (seq / "gt" / "gt.txt").exists() else "test"
        dst = root / "_seqs" / split / seq.name
        dst.mkdir(parents=True, exist_ok=True)
        for sub in ("img1", "gt", "det"):
            if (seq / sub).exists() and not (dst / sub).exists():
                (dst / sub).symlink_to((seq / sub).resolve())
        ini = dst / "seqinfo.ini"
        if (seq / "seqinfo.ini").exists():
            if not ini.exists():
                ini.symlink_to((seq / "seqinfo.ini").resolve())
        elif not ini.exists():
            n, ext, w, h = _image_info(seq / "img1")
            cp = configparser.ConfigParser()
            cp.optionxform = str  # type: ignore[assignment,method-assign]
            cp["Sequence"] = {"name": seq.name, "imDir": "img1", "frameRate": f"{fps:g}",
                              "seqLength": str(n), "imWidth": str(w), "imHeight": str(h),
                              "imExt": ext}  # fmt: skip
            with ini.open("w") as fh:
                cp.write(fh, space_around_delimiters=False)
            print(f"[warn] {seq.name}: wrote seqinfo.ini with frameRate={fps:g} (not in archive)")
        out[split].append(seq.name)
    for k, v in out.items():
        print(f"{k}: {len(v)} sequences {v}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m models.prepare")
    p.add_argument("dataset", choices=["sompt22"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--fps", type=float, default=30.0)
    a = p.parse_args(argv)
    prepare_mot_like(a.root, a.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

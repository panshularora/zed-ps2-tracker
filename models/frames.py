"""Frame sources: a MOTChallenge / DanceTrack image folder, or a video file (the gate clip)."""

from __future__ import annotations

import configparser
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from models.protocols import BGRImage


class ImageFolderSource:
    """``<seq>/img1/000001.jpg ...`` described by ``<seq>/seqinfo.ini``."""

    def __init__(self, seq_dir: str | Path) -> None:
        self.seq_dir = Path(seq_dir)
        cp = configparser.ConfigParser()
        if not cp.read(self.seq_dir / "seqinfo.ini"):
            raise FileNotFoundError(self.seq_dir / "seqinfo.ini")
        s = cp["Sequence"]
        self.name = s["name"]
        self.frame_rate = float(s["frameRate"])
        self.seq_length = int(s["seqLength"])
        self.width = int(s["imWidth"])
        self.height = int(s["imHeight"])
        self.im_dir = self.seq_dir / s.get("imDir", "img1")
        self.ext = s.get("imExt", ".jpg")

    def path(self, frame_id: int) -> Path:
        return self.im_dir / f"{frame_id:06d}{self.ext}"

    def __iter__(self) -> Iterator[tuple[int, BGRImage]]:
        import cv2

        for f in range(1, self.seq_length + 1):
            img = cv2.imread(str(self.path(f)))
            if img is None:
                raise FileNotFoundError(self.path(f))
            yield f, np.asarray(img, dtype=np.uint8)


class VideoSource:
    """Any file OpenCV/FFmpeg can decode. Frame count is probed by decoding (exact)."""

    def __init__(self, path: str | Path, name: str | None = None) -> None:
        import cv2

        self.path = Path(path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video {self.path}")
        self.frame_rate = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        self.seq_length = n
        self.name = name or self.path.stem

    def __iter__(self) -> Iterator[tuple[int, BGRImage]]:
        import cv2

        cap = cv2.VideoCapture(str(self.path))
        f = 0
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                f += 1
                yield f, np.asarray(img, dtype=np.uint8)
        finally:
            cap.release()
        if f != self.seq_length:
            raise RuntimeError(f"{self.path}: decoded {f} frames, probed {self.seq_length}")

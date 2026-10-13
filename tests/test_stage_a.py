"""Stage A orchestration with fake detector / embedder / frames (no torch, no GPU)."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from models.stage_a import is_done, process_sequence, run_many
from tracker.config import load_config
from tracker.io import load_cache
from tracker.runner import run_sequence


class FakeSource:
    def __init__(self, name: str = "FAKE-01", n: int = 5, broken_at: int | None = None) -> None:
        self.name, self.seq_length, self.frame_rate = name, n, 25.0
        self.width, self.height = 320, 240
        self.broken_at = broken_at

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        for f in range(1, self.seq_length + 1):
            if f == self.broken_at:
                return
            yield f, np.full((self.height, self.width, 3), f, np.uint8)


class FakeDetector:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def info(self) -> dict[str, object]:
        return {"name": "fake", "conf_floor": 0.0}

    def __call__(self, images: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        self.calls += 1
        out = []
        for im in images:
            f = int(im[0, 0, 0])
            n = f % 3  # 1, 2, 0, 1, 2 detections; includes empty frames
            tlwh = np.array([[10 + 4 * f + 50 * i, 20, 30, 80] for i in range(n)], np.float32)
            score = np.linspace(0.95, 0.06, n).astype(np.float32)  # includes sub-0.1 scores
            out.append((tlwh.reshape(-1, 4), score))
        return out


class FakeEmbedder:
    name, dim = "fake", 4

    def info(self) -> dict[str, object]:
        return {"name": "fake", "dim": 4}

    def __call__(self, image: np.ndarray, tlwh: np.ndarray) -> np.ndarray:
        e = np.zeros((len(tlwh), 4), np.float32)
        e[np.arange(len(tlwh)), np.arange(len(tlwh)) % 4] = 1.0
        return e.astype(np.float16)


def test_process_sequence_keeps_everything() -> None:
    c = process_sequence(FakeSource(), FakeDetector(), FakeEmbedder(), dataset="X", split="val",
                         batch_size=2)  # fmt: skip
    assert c.frame.tolist() == [1, 2, 2, 4, 5, 5]
    assert c.score.min() < 0.1  # nothing filtered by score
    assert c.emb is not None
    assert c.emb.shape == (6, 4)
    assert c.meta.seq_length == 5
    assert c.meta.frame_rate == 25.0
    assert c.meta.extra["stage_a_seconds"] >= 0


def test_frame_count_mismatch_raises() -> None:
    with pytest.raises(RuntimeError, match="read 2 frames"):
        process_sequence(FakeSource(broken_at=3), FakeDetector(), None, dataset="X", split="val")


def test_run_many_is_resumable(tmp_path: Path) -> None:
    det = FakeDetector()
    srcs = [FakeSource("A-01"), FakeSource("A-02")]
    paths = run_many(srcs, det, FakeEmbedder(), tmp_path, dataset="X", split="train",
                     detector_saw_eval_half=True, batch_size=4)  # fmt: skip
    assert [p.name for p in paths] == ["A-01.npz", "A-02.npz"]
    assert all(is_done(p) for p in paths)
    assert load_cache(paths[0]).meta.detector_saw_eval_half
    calls = det.calls
    run_many(srcs, det, FakeEmbedder(), tmp_path, dataset="X", split="train",
             detector_saw_eval_half=True, batch_size=4)  # fmt: skip
    assert det.calls == calls  # nothing recomputed
    log = (tmp_path / "stage_a_log.jsonl").read_text().splitlines()
    assert len(log) == 2
    # a corrupt file is detected and recomputed
    paths[1].write_bytes(b"garbage")
    assert not is_done(paths[1])
    run_many(srcs, det, FakeEmbedder(), tmp_path, dataset="X", split="train",
             detector_saw_eval_half=True, batch_size=4)  # fmt: skip
    assert is_done(paths[1])


def test_stage_a_cache_feeds_stage_b(tmp_path: Path) -> None:
    (p,) = run_many([FakeSource()], FakeDetector(), FakeEmbedder(), tmp_path, dataset="X",
                    split="train", detector_saw_eval_half=False, batch_size=3)  # fmt: skip
    cfg = load_config(Path(__file__).parents[1] / "configs/baselines/bytetrack_reid.yaml")
    run = run_sequence(load_cache(p), cfg)
    assert len(run.results) == 5


def _seq(root: Path, name: str, digits: int, n: int = 3) -> Path:
    import cv2

    d = root / name
    (d / "img1").mkdir(parents=True)
    (d / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate=20\nseqLength={n}\n"
        "imWidth=32\nimHeight=24\nimExt=.jpg\n"
    )
    for f in range(1, n + 1):
        cv2.imwrite(str(d / "img1" / f"{f:0{digits}d}.jpg"), np.full((24, 32, 3), 100, np.uint8))
    return d


@pytest.mark.parametrize("digits", [6, 8])
def test_image_folder_6_and_8_digit_names(tmp_path: Path, digits: int) -> None:
    from models.frames import ImageFolderSource

    src = ImageFolderSource(_seq(tmp_path, "s", digits))
    assert src.frame_digits == digits
    assert [f for f, _ in src] == [1, 2, 3]


def test_image_folder_without_first_frame_names_both_candidates(tmp_path: Path) -> None:
    from models.frames import ImageFolderSource

    d = _seq(tmp_path, "s", 6)
    (d / "img1" / "000001.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="00000001"):
        ImageFolderSource(d)


def test_dancetrack_8_digit_sequence_produces_a_cache(tmp_path: Path) -> None:
    """The October 2026 laptop run crashed here (DanceTrack img1/00000001.jpg)."""
    from models.frames import ImageFolderSource

    src = ImageFolderSource(_seq(tmp_path / "val", "dancetrack0004", 8))
    out = tmp_path / "cache" / "DanceTrack" / "dancetrack" / "val"
    (p,) = run_many(
        [src],
        FakeDetector(),
        FakeEmbedder(),
        out,
        dataset="DanceTrack",
        split="val",
        detector_saw_eval_half=False,
        batch_size=2,
    )
    assert p.name == "dancetrack0004.npz"
    assert is_done(p)

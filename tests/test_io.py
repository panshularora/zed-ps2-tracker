import json
from pathlib import Path

import numpy as np
import pytest

from tracker.config import TrackerConfig
from tracker.io import (
    FRAME_JSON_SCHEMA,
    CacheFormatError,
    CacheMeta,
    SequenceCache,
    load_cache,
    load_mot_det_txt,
    save_cache,
    validate,
    write_frames_jsonl,
    write_mot_results,
)
from tracker.runner import run_sequence


def tiny_cache(with_emb: bool = True, seq_length: int = 6) -> SequenceCache:
    """3 frames with detections, frames 4-6 empty; one person walking right, one static."""
    rng = np.random.default_rng(0)
    frames, boxes, scores = [], [], []
    for f in (1, 2, 3):
        frames += [f, f]
        boxes += [[100 + 5 * f, 100, 40, 100], [400, 120, 42, 104]]
        scores += [0.95, 0.35]
    emb = rng.normal(size=(6, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    meta = CacheMeta(
        sequence="TINY-01",
        dataset="synthetic",
        split="train",
        frame_rate=30.0,
        seq_length=seq_length,
        im_width=640,
        im_height=480,
        detector={"name": "fake", "conf_floor": 0.01},
        reid={"name": "fake", "dim": 8} if with_emb else None,
    )
    return SequenceCache(
        meta,
        np.array(frames, np.int32),
        np.array(boxes, np.float32),
        np.array(scores, np.float32),
        emb.astype(np.float16) if with_emb else None,
    )


@pytest.mark.parametrize("with_emb", [True, False])
def test_roundtrip(tmp_path: Path, with_emb: bool) -> None:
    c = tiny_cache(with_emb)
    p = save_cache(tmp_path / "seq.npz", c)
    assert not (tmp_path / "seq.npz.tmp").exists()
    c2 = load_cache(p)
    assert c2.meta == c.meta
    np.testing.assert_array_equal(c2.frame, c.frame)
    np.testing.assert_array_equal(c2.tlwh, c.tlwh)
    if with_emb:
        assert c2.emb is not None
        assert c2.emb.dtype == np.float16
        assert c2.embedding_dim == 8
    else:
        assert c2.emb is None


def test_frames_iterates_every_frame() -> None:
    frames = list(tiny_cache().frames())
    assert [f for f, _ in frames] == [1, 2, 3, 4, 5, 6]
    assert [len(d) for _, d in frames] == [2, 2, 2, 0, 0, 0]
    d = frames[0][1]
    assert d.embeddings is not None
    assert d.embeddings.dtype == np.float32


@pytest.mark.parametrize(
    ("mutate", "msg"),
    [
        (
            lambda c: SequenceCache(c.meta, c.frame.astype(np.int64), c.tlwh, c.score, c.emb),
            "int32",
        ),
        (lambda c: SequenceCache(c.meta, c.frame[::-1].copy(), c.tlwh, c.score, c.emb), "sorted"),
        (lambda c: SequenceCache(c.meta, c.frame + 10, c.tlwh, c.score, c.emb), "outside"),
        (lambda c: SequenceCache(c.meta, c.frame, c.tlwh * 0, c.score, c.emb), "width"),
        (lambda c: SequenceCache(c.meta, c.frame, c.tlwh, c.score, c.emb * 2), "normalised"),
        (
            lambda c: SequenceCache(c.meta, c.frame, c.tlwh, c.score, c.emb.astype(np.float32)),
            "float16",
        ),
    ],
)
def test_validate_rejects(mutate, msg: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(CacheFormatError, match=msg):
        validate(mutate(tiny_cache()))


def test_load_rejects_missing_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.npz"
    np.savez(p, frame=np.zeros(1, np.int32))
    with pytest.raises(CacheFormatError, match="missing"):
        load_cache(p)


def test_mot_det_loader(tmp_path: Path) -> None:
    seq = tmp_path / "MOT17-99-FRCNN"
    (seq / "det").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-99-FRCNN\nimDir=img1\nframeRate=25\nseqLength=4\n"
        "imWidth=1920\nimHeight=1080\nimExt=.jpg\n"
    )
    (seq / "det" / "det.txt").write_text(
        "2,-1,10,20,30,40,0.9\n1,-1,1,2,3,4,0.5\n1,-1,1,2,0,4,0.5\n"
    )
    c = load_mot_det_txt(seq)
    assert c.meta.frame_rate == 25
    assert c.frame.tolist() == [1, 2]  # sorted, zero-width box dropped
    assert c.emb is None
    validate(c)


def test_run_and_write_outputs(tmp_path: Path) -> None:
    run = run_sequence(tiny_cache(), TrackerConfig())
    assert len(run.results) == 6
    assert run.latency_ms.shape == (6,)
    assert np.all(run.latency_ms >= 0)
    mot = write_mot_results(tmp_path / "out.txt", run.results)
    rows = [r.split(",") for r in mot.read_text().splitlines()]
    assert rows
    assert {r[1] for r in rows} == {"1"}  # walking person; the 0.35 box never starts a track
    assert all(len(r) == 10 for r in rows)
    js = write_frames_jsonl(tmp_path / "out.jsonl", run.results, {"sequence": "TINY-01"})
    lines = [json.loads(x) for x in js.read_text().splitlines()]
    assert lines[0] == {"schema": FRAME_JSON_SCHEMA, "sequence": "TINY-01"}
    assert len(lines) == 7
    assert lines[1]["pending"] == []
    assert lines[1]["tracks"][0]["id"] == 1
    assert set(lines[1]["tracks"][0]) == {"id", "tlwh", "score", "state", "visibility"}

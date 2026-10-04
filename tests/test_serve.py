"""Demo service: event contract and FastAPI endpoints (skipped without the ``serve`` extra)."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from eval.gate_events import Gate
from tracker.io import CacheMeta, SequenceCache, save_cache
from tracker.types import FrameResult, TrackOutput

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient

from serve.app import app
from serve.events import EVENT_TYPES, EventParams, gate_events

START = datetime(2026, 10, 5, 9, 0, tzinfo=UTC)


def walkers(n: int = 60, people: tuple[tuple[float, int], ...] = ((300, 0), (700, 8)),
            score: float = 0.9) -> SequenceCache:  # fmt: skip
    """People walking down through y = 648 (60 % of 1080) at 12 px/frame, 25 fps."""
    fr, tl, sc = [], [], []
    for f in range(1, n + 1):
        for x, lag in people:
            fr.append(f)
            tl.append([x, 300 + (f - lag) * 12, 80, 200])
            sc.append(score)
    meta = CacheMeta("SYN-01", "MOT17", "full", 25.0, n, 1920, 1080)
    return SequenceCache(meta, np.array(fr, np.int32), np.array(tl, np.float32),
                         np.array(sc, np.float32))  # fmt: skip


def results_for(track_vis: list[float], score: float = 0.9) -> list[FrameResult]:
    """One track moving down through y = 100, frames 1..len; visibility per frame."""
    return [
        FrameResult(f, (TrackOutput(1, (0.0, 4.0 * f + 20, 10.0, 40.0), score, "confirmed", v),))
        for f, v in enumerate(track_vis, start=1)
    ]


GATE = Gate("g", (0.0, 100.0), (200.0, 100.0))


def contract_ok(e: dict[str, object]) -> None:
    assert e["type"] in EVENT_TYPES
    assert e["direction"] in ("in", "out")
    assert isinstance(e["track_id"], int)
    assert 0.0 <= float(e["confidence"]) <= 1.0  # type: ignore[arg-type]
    assert datetime.fromisoformat(str(e["timestamp"])).tzinfo is not None


def test_clean_crossing() -> None:
    ev = gate_events(results_for([1.0] * 40), 25.0, GATE, START)
    assert [e["type"] for e in ev] == ["crossing"]
    contract_ok(ev[0])
    assert ev[0]["confidence"] == 0.9
    # timestamp = start + (frame - 1) / fps
    t = datetime.fromisoformat(ev[0]["timestamp"])
    assert ev[0]["frame"] == 13
    assert t == START + timedelta(seconds=12 / 25)


def test_occluded_crossing_is_held_not_guessed() -> None:
    vis = [1.0] * 40
    vis[10] = 0.3  # frame 11: occluded just before the crossing at frame 13
    ev = gate_events(results_for(vis), 25.0, GATE, START)
    assert [e["type"] for e in ev] == ["ambiguous_hold"]
    assert ev[0]["reasons"] == ["occlusion"]
    assert ev[0]["confidence"] == 0.3
    contract_ok(ev[0])
    # the same dip after the crossing does not hold it (only the past window counts)
    vis = [1.0] * 40
    vis[20] = 0.3
    assert [e["type"] for e in gate_events(results_for(vis), 25.0, GATE, START)] == ["crossing"]


def test_low_score_crossing_is_held() -> None:
    ev = gate_events(results_for([1.0] * 40, score=0.3), 25.0, GATE, START)
    assert ev[0]["type"] == "ambiguous_hold"
    assert ev[0]["reasons"] == ["low_confidence"]
    off = EventParams(hold_score=0.0)
    assert gate_events(results_for([1.0] * 40, score=0.3), 25.0, GATE, START, off)[0]["type"] == (
        "crossing"
    )


def test_tailgate_and_hold_excluded(tmp_path: Path) -> None:
    from tracker.config import load_config
    from tracker.runner import run_sequence

    run = run_sequence(walkers(), load_config(Path("configs/baselines/bytetrack.yaml")))
    g = Gate("h60", (0.0, 648.0), (1920.0, 648.0))
    ev = gate_events(run.results, 25.0, g, START)
    kinds = [e["type"] for e in ev]
    assert kinds.count("crossing") == 2
    assert kinds.count("tailgate") == 1
    tg = next(e for e in ev if e["type"] == "tailgate")
    contract_ok(tg)
    assert tg["gap_s"] == 0.32  # 8 frames at 25 fps
    assert tg["lead_track_id"] != tg["track_id"]
    # far apart in time: no tailgate
    run = run_sequence(walkers(people=((300, 0), (700, 40)), n=100),
                       load_config(Path("configs/baselines/bytetrack.yaml")))  # fmt: skip
    assert "tailgate" not in [e["type"] for e in gate_events(run.results, 25.0, g, START)]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def npz_bytes(tmp_path: Path, cache: SequenceCache) -> bytes:
    return save_cache(tmp_path / "c.npz", cache).read_bytes()


def test_health_and_configs(client: TestClient) -> None:
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    names = client.get("/api/configs").json()
    assert "baselines/bytetrack" in names
    assert "ours/visibility" in names
    assert h["configs"] == len(names)


def test_track_endpoint(client: TestClient, tmp_path: Path) -> None:
    files = {"cache": ("c.npz", io.BytesIO(npz_bytes(tmp_path, walkers())))}
    r = client.post("/api/track", files=files,
                    data={"config": "ours/visibility",
                          "start_time": "2026-10-05T09:00:00+05:30"})  # fmt: skip
    assert r.status_code == 200, r.text
    body = r.json()
    kinds = [e["type"] for e in body["events"]]
    assert kinds.count("crossing") + kinds.count("ambiguous_hold") == 2
    for e in body["events"]:
        contract_ok(e)
        assert e["timestamp"].startswith("2026-10-05T09:00:")
        assert e["timestamp"].endswith("+05:30")
    clip = body["clip"]
    assert clip["gate"] == {"a": [0.0, 648.0], "b": [1920.0, 648.0]}
    assert clip["frames"] == 60
    assert len(clip["tracks"]) == 2
    assert body["latency"]["frames"] == 60
    assert body["latency"]["p95_ms"] > 0


def test_track_rejects_bad_input(client: TestClient, tmp_path: Path) -> None:
    good = npz_bytes(tmp_path, walkers())
    bad_cfg = client.post("/api/track", files={"cache": ("c.npz", good)},
                          data={"config": "../../etc/passwd"})  # fmt: skip
    assert bad_cfg.status_code == 422
    junk = client.post("/api/track", files={"cache": ("c.npz", b"not a zip")})
    assert junk.status_code == 422
    gate = client.post("/api/track", files={"cache": ("c.npz", good)}, data={"gate": "1,2,3"})
    assert gate.status_code == 422
    naive = client.post("/api/track", files={"cache": ("c.npz", good)},
                        data={"start_time": "2026-10-05T09:00:00"})  # fmt: skip
    assert naive.status_code == 422


def test_upload_limit(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import serve.app as sa

    monkeypatch.setattr(sa, "MAX_UPLOAD_MB", 0.0001)
    r = client.post("/api/track", files={"cache": ("c.npz", npz_bytes(tmp_path, walkers()))})
    assert r.status_code == 413

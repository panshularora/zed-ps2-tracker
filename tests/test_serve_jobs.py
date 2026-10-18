"""Product API of serve/: clips, uploads, jobs, analysis, detector decoding, benchmarks.

Video tests use a small synthetic MP4 (OpenCV writer) and a fake detector that finds the white
rectangles in each frame, so they run in a few seconds without model files.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")
pytest.importorskip("cv2")

import cv2
from fastapi.testclient import TestClient

from eval.gate_events import Gate
from serve import analysis
from serve.api import Service
from serve.app import create_app
from serve.benchmarks import benchmarks
from serve.jobs import JobStore
from serve.pipeline import JobOptions
from serve.settings import Settings
from tracker.config import config_to_dict, load_config

ROOT = Path(__file__).resolve().parent.parent
START = datetime(2026, 10, 5, 9, 0, tzinfo=UTC)
W, H, FPS, N = 320, 240, 10.0, 40


# ---------------------------------------------------------------- synthetic video + detector
def person_boxes(f: int) -> list[tuple[int, int, int, int]]:
    """Two 'people' walking down through y = 0.6 H; the second is hidden on frames 15-19."""
    out = [(60, -40 + 6 * f, 30, 60)]
    if not 15 <= f <= 19:
        out.append((200, -60 + 6 * f, 30, 60))
    return [b for b in out if b[1] + b[3] > 0 and b[1] < H]


def write_video(path: Path, n: int = N) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(1, n + 1):
        img = np.zeros((H, W, 3), np.uint8)
        for x, y, w, h in person_boxes(f):
            cv2.rectangle(img, (x, max(0, y)), (x + w, min(H - 1, y + h)), (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    assert path.stat().st_size > 0
    return path


class FakeDetector:
    name = "fake"

    def detect(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = (bgr[:, :, 0] > 127).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        boxes = [stats[i, :4].astype(np.float32) for i in range(1, n) if stats[i, 4] > 50]
        tl = np.array(boxes, np.float32).reshape(-1, 4)
        return tl, np.full(len(tl), 0.9, np.float32)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    raw, blurred = tmp_path / "raw", tmp_path / "blurred"
    raw.mkdir()
    blurred.mkdir()
    write_video(raw / "walk.mp4")
    write_video(blurred / "walk.mp4")
    (raw / "notblurred.mp4").write_bytes((raw / "walk.mp4").read_bytes())
    manifest = tmp_path / "clips.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "clips": [
                    {
                        "id": "walk",
                        "title": "Walk",
                        "file": "walk.mp4",
                        "page": "https://example.invalid/1",
                        "consent": False,
                        "event_styling": "neutral",
                        "gate": [0.0, 0.6, 1.0, 0.6],
                    },
                    {
                        "id": "own_gate",
                        "title": "Own gate",
                        "file": "own.mp4",
                        "page": "local",
                        "consent": True,
                        "event_styling": "alert",
                    },
                    {"id": "raw_only", "title": "Raw", "file": "notblurred.mp4", "page": "x"},
                ]
            }
        )
    )
    s = Settings(
        data_dir=tmp_path / "serve",
        clip_dir=blurred,
        raw_clip_dir=raw,
        demo_cut_dir=tmp_path / "cuts",
        clip_manifest=manifest,
        model_dir=tmp_path / "models",
        results_dir=ROOT / "docs" / "results",
        config_root=ROOT / "configs",
        web_dist=tmp_path / "web",
        max_video_mb=1.0,
        max_video_seconds=10.0,
    )
    monkeypatch.setattr(Service, "detector", lambda self, name: FakeDetector())

    def fake_blur(src: Path, dst: Path, model_dir: Path, **kw: Any) -> dict[str, float]:
        prog = kw.get("progress")
        if prog:
            prog("blur", 1, 1)
        shutil.copy(src, dst)
        return {"frames": 0}

    monkeypatch.setattr("serve.blur.blur_video", fake_blur)
    app = create_app(s)
    with TestClient(app) as client:
        yield {"client": client, "settings": s, "tmp": tmp_path, "app": app}


def wait(client: TestClient, job_id: str, timeout: float = 60.0) -> dict[str, Any]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec: dict[str, Any] = client.get(f"/api/jobs/{job_id}").json()
        if rec["status"] in ("succeeded", "failed", "cancelled"):
            return rec
        time.sleep(0.05)
    raise AssertionError("job did not finish")


# ---------------------------------------------------------------- clips
def test_clip_listing_serves_only_blurred_copies(env: dict[str, Any]) -> None:
    c = env["client"]
    clips = c.get("/api/clips").json()
    assert [x["id"] for x in clips] == ["walk"]  # own.mp4 absent, raw_only has no blurred copy
    walk = clips[0]
    assert walk["blurred"] is True
    assert walk["consent"] is False
    assert walk["event_styling"] == "neutral"
    assert (walk["width"], walk["height"], walk["frames"]) == (W, H, N)
    r = c.get(walk["video_url"])
    assert r.status_code == 200
    assert r.content == (env["settings"].clip_dir / "walk.mp4").read_bytes()
    part = c.get(walk["video_url"], headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert len(part.content) == 100
    assert c.get("/api/clips/raw_only/video").status_code == 404
    err = c.get("/api/clips/raw_only").json()
    assert err["error"]["code"] == "clip_missing"
    assert c.get("/api/clips/../etc/passwd").status_code == 404
    assert c.get("/api/clips/walk/poster.jpg").headers["content-type"] == "image/jpeg"


# ---------------------------------------------------------------- jobs
def test_demo_cut_fallback_when_full_clip_absent(env: dict[str, Any]) -> None:
    """No full clip, but a committed (already blurred) demo cut: listed, served, trackable."""
    c, s = env["client"], env["settings"]
    (s.clip_dir / "walk.mp4").unlink()
    assert c.get("/api/clips").json() == []
    s.demo_cut_dir.mkdir()
    write_video(s.demo_cut_dir / "walk.mp4")
    clips = c.get("/api/clips").json()
    assert [(x["id"], x["demo_cut"]) for x in clips] == [("walk", True)]
    assert c.get("/api/clips/walk/video").status_code == 200
    rec = c.post("/api/jobs", json={"clip_id": "walk", "detector": "yolox_s"}).json()
    assert wait(c, rec["id"])["status"] == "succeeded"
    # the full clip wins once it exists
    write_video(s.clip_dir / "walk.mp4")
    assert c.get("/api/clips").json()[0]["demo_cut"] is False


def test_demo_cut_is_rejected_when_neither_copy_exists(env: dict[str, Any]) -> None:
    c, s = env["client"], env["settings"]
    (s.clip_dir / "walk.mp4").unlink()
    assert c.get("/api/clips").json() == []
    assert c.get("/api/clips/walk/video").status_code == 404


def test_job_lifecycle_and_outputs(env: dict[str, Any]) -> None:
    c = env["client"]
    r = c.post("/api/jobs", json={"clip_id": "walk", "config": "ours/visibility_botsort"})
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] in ("queued", "running", "succeeded")
    done = wait(c, job["id"])
    assert done["status"] == "succeeded", done
    assert done["progress"] == 1.0
    m = done["metrics"]
    assert m["frames"] == N
    assert m["tracks"] == 2
    assert m["crossings_in"] + m["crossings_out"] == 2
    assert m["id_preserving_recoveries"] >= 1  # the hidden walker keeps its ID
    assert done["timing"]["job_s"] > 0
    assert done["timing"]["detector_p50_ms"] >= 0
    res = c.get(f"/api/jobs/{job['id']}/result").json()
    assert res["schema"] == "zed-ps2-job-result/v1"
    assert [d["frame"] for d in res["sample_detections"]] == [1, 2, 3]
    assert "sample_detections" not in res["timing"]
    assert res["video"]["fps"] == FPS
    assert res["clip"]["event_styling"] == "neutral"
    assert set(res["tracks"])
    assert all(len(r) == 7 for t in res["tracks"].values() for r in t)
    kinds = {e["type"] for e in res["events"]}
    assert "tailgate" not in kinds
    assert "ambiguous_hold" not in kinds  # neutral clip
    rec = [e for e in res["events"] if e["type"] == "recovery"]
    assert rec
    assert rec[0]["reason"] == "lost"
    assert rec[0]["gap_frames"] >= 3
    txt = c.get(f"/api/jobs/{job['id']}/tracks.txt").text.strip().splitlines()
    assert len(txt) == m["boxes"]
    f, tid, *_rest = txt[0].split(",")
    assert int(f) >= 1
    assert int(tid) >= 1
    assert _rest[-3:] == ["-1", "-1", "-1"]
    ev = c.get(f"/api/jobs/{job['id']}/events.json").json()
    assert ev == res["events"]
    # the same request is answered from the finished job
    again = c.post("/api/jobs", json={"clip_id": "walk", "config": "ours/visibility_botsort"})
    assert again.status_code == 200
    assert again.json()["id"] == job["id"]
    fresh = c.post(
        "/api/jobs", json={"clip_id": "walk", "config": "ours/visibility_botsort", "reuse": False}
    )
    assert fresh.status_code == 202
    assert fresh.json()["id"] != job["id"]
    wait(c, fresh.json()["id"])
    assert any(j["id"] == job["id"] for j in c.get("/api/jobs?clip_id=walk").json())
    assert c.delete(f"/api/jobs/{job['id']}").status_code == 204
    assert c.get(f"/api/jobs/{job['id']}").status_code == 404


def test_job_options_validated(env: dict[str, Any]) -> None:
    c = env["client"]
    bad = [
        ({"clip_id": "nope"}, 404, "clip_not_found"),
        ({"clip_id": "walk", "config": "../../etc/passwd"}, 422, "unknown_config"),
        ({"clip_id": "walk", "config": "baselines/botsort_reid"}, 422, "needs_gpu"),
        ({"clip_id": "walk", "stride": 0}, 422, "invalid_input"),
        ({"clip_id": "walk", "gate": [0, 0.5, 2, 0.5]}, 422, "invalid_input"),
        ({"clip_id": "walk", "gate": [0.5, 0.5, 0.5, 0.5]}, 422, "invalid_input"),
        ({"clip_id": "walk", "detector": "yolov99"}, 422, "invalid_input"),
    ]
    for body, status, code in bad:
        r = c.post("/api/jobs", json=body)
        assert r.status_code == status, (body, r.text)
        assert r.json()["error"]["code"] == code
        assert r.json()["detail"]  # FastAPI's key kept for old clients


def test_max_seconds_and_stride(env: dict[str, Any]) -> None:
    c = env["client"]
    r = c.post("/api/jobs", json={"clip_id": "walk", "stride": 2, "max_seconds": 2})
    job = wait(c, r.json()["id"])
    assert job["options"]["max_frames"] == 10  # 2 s × 10 fps / stride 2
    res = c.get(f"/api/jobs/{job['id']}/result").json()
    assert res["frames_tracked"] == 10
    assert res["fps_tracked"] == 5.0


def test_result_before_done_is_409(env: dict[str, Any]) -> None:
    c = env["client"]
    store: JobStore = env["app"].state.service.store
    rec = store.create({"id": "walk"}, JobOptions())
    r = c.get(f"/api/jobs/{rec['id']}/result")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "job_not_done"
    cancelled = c.post(f"/api/jobs/{rec['id']}/cancel").json()
    assert cancelled["status"] == "cancelled"


def test_interrupted_jobs_are_marked_failed(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    rec = store.create({"id": "walk"}, JobOptions())
    store.update(rec["id"], status="running")
    assert JobStore(tmp_path).recover() == 1
    assert store.get(rec["id"])["error"]["code"] == "interrupted"


def test_failed_job_reports_structured_error(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: Service, name: str) -> Any:
        raise RuntimeError("model missing")

    monkeypatch.setattr(Service, "detector", boom)
    c = env["client"]
    job = wait(c, c.post("/api/jobs", json={"clip_id": "walk", "reuse": False}).json()["id"])
    assert job["status"] == "failed"
    assert job["error"]["code"] == "job_failed"
    assert "model missing" in job["error"]["message"]


# ---------------------------------------------------------------- uploads
def test_upload_validation(env: dict[str, Any]) -> None:
    c = env["client"]
    vid = (env["settings"].clip_dir / "walk.mp4").read_bytes()
    r = c.post("/api/uploads", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_type"
    r = c.post("/api/uploads", files={"file": ("a.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "not_a_video"
    r = c.post("/api/uploads", files={"file": ("big.mp4", b"0" * (2 * 2**20), "video/mp4")})
    assert r.status_code == 413
    long = write_video(env["tmp"] / "long.mp4", n=120)  # 12 s > 10 s limit
    r = c.post("/api/uploads", files={"file": ("long.mp4", long.read_bytes(), "video/mp4")})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "too_long"
    assert list(env["settings"].uploads_dir.iterdir()) == []  # rejected uploads are removed
    r = c.post("/api/uploads", files={"file": ("my gate.mp4", vid, "video/mp4")})
    assert r.status_code == 201, r.text
    up = r.json()
    assert up["id"].startswith("upload-")
    assert up["blurred"] is False
    assert up["event_styling"] == "neutral"
    assert up["video_url"] is None
    # not listed and not served until blurred
    assert up["id"] not in [x["id"] for x in c.get("/api/clips").json()]
    assert c.get(f"/api/clips/{up['id']}/video").status_code == 409


def test_upload_job_blurs_first_and_consent_enables_contract(env: dict[str, Any]) -> None:
    c = env["client"]
    vid = (env["settings"].clip_dir / "walk.mp4").read_bytes()
    up = c.post(
        "/api/uploads", files={"file": ("gate.mp4", vid, "video/mp4")}, data={"consent": "true"}
    ).json()
    assert up["consent"] is True
    assert up["event_styling"] == "alert"
    job = wait(c, c.post("/api/jobs", json={"clip_id": up["id"]}).json()["id"])
    assert job["status"] == "succeeded", job
    assert c.get(f"/api/clips/{up['id']}/video").status_code == 200
    assert up["id"] in [x["id"] for x in c.get("/api/clips").json()]
    res = c.get(f"/api/jobs/{job['id']}/result").json()
    gate_types = {e["type"] for e in res["events"] if e["kind"] == "gate"}
    assert gate_types <= {"crossing", "tailgate", "ambiguous_hold"}
    for e in res["events"]:
        if e["kind"] == "gate":
            assert "confidence" in e  # full contract only for consented footage


# ---------------------------------------------------------------- health, benchmarks
def test_health(env: dict[str, Any]) -> None:
    h = env["client"].get("/api/health").json()
    assert h["status"] == "ok"
    assert h["clips"] == 1
    assert set(h["detectors"]) == {"yolox_nano", "yolox_tiny", "yolox_s"}
    assert h["limits"]["max_video_mb"] == 1.0


def test_benchmarks_come_from_repo_files(env: dict[str, Any]) -> None:
    b = env["client"].get("/api/benchmarks").json()
    raw = json.loads((ROOT / "docs/results/ablation/ablation.json").read_text())
    by = {r["name"]: r for r in raw}
    for row in b["ablation"]["rows"]:
        assert row["HOTA"] == round(by[row["name"]]["HOTA"], 1)
        assert row["IDSW"] == int(by[row["name"]]["IDSW"])
    head = b["headline"]
    assert head["name"] == "ours_vis_botsort"
    assert (head["HOTA"], head["IDF1"], head["IDSW"]) == (52.5, 59.8, 180)
    assert b["phase3"]["status"] == "available"  # docs/results/phase3 = laptop GPU run
    assert "docs/results/phase3/go_no_go_v2.md" in b["phase3"]["files"]
    assert b["go_no_go"]["sanity_check"] is True
    assert b["go_no_go"]["verdict"] == "no-go"
    assert any(r["check"] for r in b["gate_events"]["rows"])
    plot = env["client"].get(b["plots"]["stress_visibility"])
    assert plot.status_code == 200
    assert plot.content[:4] == b"\x89PNG"
    assert env["client"].get("/api/benchmarks/plots/nope.png").status_code == 404


def test_benchmarks_gpu_run_comes_from_phase3_files(env: dict[str, Any]) -> None:
    g = env["client"].get("/api/benchmarks").json()["gpu"]
    p3 = ROOT / "docs/results/phase3"
    summary = json.loads((p3 / "summary.json").read_text())
    assert [d["id"] for d in g["datasets"]] == [c["id"] for c in summary["ran"]]
    assert "mot20_val_half__mot20full_SAW_EVAL_HALF" in g["excluded"]
    assert all("SAW_EVAL_HALF" not in d["id"] for d in g["datasets"])
    for d in g["datasets"]:
        raw = {r["name"]: r for r in json.loads((p3 / d["id"] / "ablation.json").read_text())}
        assert [r["name"] for r in d["rows"]] == list(raw)
        for r in d["rows"]:
            for k in ("HOTA", "IDF1", "MOTA", "AssA"):
                assert r[k] == round(raw[r["name"]][k], 1)
            assert r["IDSW"] == int(raw[r["name"]]["IDSW"])
        plot = env["client"].get(d["plots"]["stress_visibility"])
        assert plot.status_code == 200
        assert plot.content[:4] == b"\x89PNG"
    mot17 = {r["name"]: r for r in g["datasets"][0]["rows"]}
    assert (mot17["botsort"]["HOTA"], mot17["botsort_reid_vis_gated"]["HOTA"]) == (64.6, 63.9)
    gng = json.loads((p3 / "go_no_go_v2.json").read_text())
    assert g["go_no_go_v2"]["verdict"] == gng["verdict"] == "NO-GO"
    assert {c["id"]: c["passed"] for c in g["go_no_go_v2"]["criteria"]} == gng["criteria"]
    crit = {c["id"]: c["got"] for c in g["go_no_go_v2"]["criteria"]}
    assert crit["1_idsw"] == "IDSW 183 -> 198"
    assert crit["5_replication"] == "MOT20 IDSW 1462 -> 1575 (fail)"
    rep = {r["metric"]: r for r in g["reproduction"]["rows"]}
    assert g["reproduction"]["verdict"] == "NOT REPRODUCED"
    assert (rep["MOTA"]["paper"], rep["MOTA"]["ours"], rep["MOTA"]["delta"]) == (76.6, 66.4, -10.2)
    assert (rep["IDF1"]["delta"], rep["IDF1"]["within"]) == (-5.5, False)
    for bad in ("gpu-..-stress_density", "gpu-mot17_val_half-ablation", "gpu-a-b-c"):
        assert env["client"].get(f"/api/benchmarks/plots/{bad}.png").status_code == 404


def test_benchmarks_missing_files(tmp_path: Path) -> None:
    b = benchmarks(tmp_path)
    assert b["ablation"] is None
    assert b["headline"] is None
    assert b["plots"] == {}
    assert b["phase3"]["status"] == "pending"
    assert b["gpu"] is None


def test_visibility_botsort_config_is_the_ablation_row() -> None:
    manifest = yaml.safe_load((ROOT / "configs/ablations/manifest.yaml").read_text())
    entry = next(e for e in manifest["entries"] if e["name"] == "ours_vis_botsort")
    a = load_config(ROOT / "configs/ours/visibility_botsort.yaml")
    b = load_config(ROOT / entry["config"], entry["overrides"])
    assert config_to_dict(a) == config_to_dict(b)


# ---------------------------------------------------------------- analysis
def rows_from(tracks: dict[int, list[tuple[int, float, float]]]) -> np.ndarray:
    """``{id: [(frame, y_bottom, visibility)]}`` -> rows frame,id,x,y,w,h,score,vis."""
    out = [
        [f, tid, 100.0, yb - 50.0, 20.0, 50.0, 0.9, v]
        for tid, pts in tracks.items()
        for f, yb, v in pts
    ]
    return np.array(out, float)


def test_recoveries_lost_and_occluded() -> None:
    lost = [(f, 10.0 * f, 1.0) for f in [*range(1, 11), *range(15, 21)]]
    occl = [(f, 300.0, 0.3 if 5 <= f <= 9 else 0.9) for f in range(1, 21)]
    blip = [(f, 500.0, 1.0) for f in [*range(1, 6), *range(7, 12)]]  # 1-frame gap: not counted
    ev = analysis.recoveries(rows_from({1: lost, 2: occl, 3: blip}), 10.0)
    assert [(e["track_id"], e["reason"], e["gap_frames"]) for e in ev] == [
        (2, "occluded", 5),
        (1, "lost", 4),
    ]
    assert ev[1]["frame"] == 15
    assert ev[1]["since"] == 10
    assert ev[1]["gap_s"] == 0.4


def test_suspected_switch_heuristic() -> None:
    a = [(f, 100.0 + 5 * f, 1.0) for f in range(1, 11)]
    b = [(f, 100.0 + 5 * f, 1.0) for f in range(13, 30)]  # born 3 frames later where a ended
    far = [(f, 900.0, 1.0) for f in range(13, 30)]  # born at the same time, far away
    ev = analysis.suspected_switches(rows_from({1: a, 2: b, 3: far}), 10.0, 29)
    assert [(e["from_id"], e["to_id"]) for e in ev] == [(1, 2)]
    assert ev[0]["heuristic"] is True
    # a track alive until the last frame cannot "switch"
    assert analysis.suspected_switches(rows_from({1: a}), 10.0, 10) == []


def test_neutral_crossings_and_metrics() -> None:
    walk = [(f, 50.0 + 10 * f, 1.0) for f in range(1, 21)]  # bottom y 60 -> 250
    back = [(f, 260.0 - 10 * f, 1.0) for f in range(1, 21)]
    rows = rows_from({1: walk, 2: back})
    gate = Gate("g", (0.0, 150.0), (400.0, 150.0))
    ev = analysis.neutral_crossings(rows, 10.0, gate, START)
    assert sorted(e["direction"] for e in ev) == ["in", "out"]
    assert all(
        set(e) == {"type", "track_id", "direction", "timestamp", "frame", "gate"} for e in ev
    )
    m = analysis.metrics(rows, 10.0, 20, ev, [], [])
    assert m["tracks"] == 2
    assert m["avg_track_length_frames"] == 20.0
    assert m["avg_track_length_s"] == 2.0
    assert (m["crossings_in"], m["crossings_out"], m["tailgates"]) == (1, 1, 0)
    empty = analysis.metrics(np.zeros((0, 8)), 10.0, 0, [], [], [])
    assert empty["tracks"] == 0
    assert empty["avg_track_length_s"] is None


# ---------------------------------------------------------------- upload -> Studio ("Your video")
@pytest.fixture
def blurred_frames(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    def fake_blur_frame(img: np.ndarray, model_dir: Path, **kw: Any) -> np.ndarray:
        calls.append(img.shape[0])
        return img

    monkeypatch.setattr("serve.blur.blur_frame", fake_blur_frame)
    return calls


def test_upload_first_frame_is_blurred_before_any_job(
    env: dict[str, Any], blurred_frames: list[int]
) -> None:
    c = env["client"]
    vid = (env["settings"].clip_dir / "walk.mp4").read_bytes()
    up = c.post("/api/uploads", files={"file": ("gate.mp4", vid, "video/mp4")}).json()
    assert c.get(f"/api/clips/{up['id']}/video").status_code == 409  # video still not served
    r = c.get(f"/api/clips/{up['id']}/first-frame.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert blurred_frames == [H]  # went through the face blur
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (H, W)


def test_user_drawn_gate_counts_crossings(env: dict[str, Any]) -> None:
    c = env["client"]
    vid = (env["settings"].clip_dir / "walk.mp4").read_bytes()
    up = c.post("/api/uploads", files={"file": ("gate.mp4", vid, "video/mp4")}).json()
    gate = [0.0, 0.45, 1.0, 0.45]
    job = wait(c, c.post("/api/jobs", json={"clip_id": up["id"], "gate": gate}).json()["id"])
    assert job["status"] == "succeeded", job
    res = c.get(f"/api/jobs/{job['id']}/result").json()
    assert res["gate"]["norm"] == gate
    m = res["metrics"]
    assert m["crossings_in"] + m["crossings_out"] >= 1  # both walkers cross y = 0.45 H
    assert res["config"] == "ours/visibility_botsort"
    assert res["tracker_info"]["reid"] is False
    assert res["tracker_info"]["ground_truth"] is False


def test_neutral_crossings_carry_unsure_flag() -> None:
    # a weak (score 0.3) track crossing: neutral styling keeps it a crossing but flags it unsure
    walk = [(f, 50.0 + 10 * f, 1.0) for f in range(1, 21)]
    rows = rows_from({1: walk})
    rows[:, 6] = 0.3
    from tracker.types import FrameResult, TrackOutput

    results = [
        FrameResult(
            int(r[0]),
            (TrackOutput(int(r[1]), tuple(r[2:6]), float(r[6]), "confirmed", float(r[7])),),
        )
        for r in rows
    ]
    gate = Gate("g", (0.0, 150.0), (400.0, 150.0))
    ev = analysis.gate_output(results, 10.0, gate, START, "neutral")
    assert [e["type"] for e in ev] == ["crossing"]
    assert ev[0]["unsure"] is True
    assert ev[0]["reasons"] == ["low_confidence"]
    m = analysis.metrics(rows, 10.0, 20, ev, [], [])
    assert m["unsure_crossings"] == 1
    assert m["crossings_in"] + m["crossings_out"] == 1


def test_tracks_csv_and_annotated_clip(env: dict[str, Any]) -> None:
    c = env["client"]
    job = wait(c, c.post("/api/jobs", json={"clip_id": "walk"}).json()["id"])
    r = c.get(f"/api/jobs/{job['id']}/tracks.csv")
    assert r.status_code == 200
    lines = r.text.strip().split("\n")
    assert lines[0] == "frame,time_s,track_id,x,y,w,h,score,visibility"
    assert len(lines) > 10
    assert "attachment" in r.headers["content-disposition"]
    r = c.get(f"/api/jobs/{job['id']}/annotated.mp4")
    assert r.status_code == 200, r.text
    out = env["tmp"] / "ann.mp4"
    out.write_bytes(r.content)
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n == N


def test_reappearances_are_events_not_people() -> None:
    # the "2 people, 4 kept their number" bug: one track recovering several times is several
    # recovery events but one ID; the per-ID count can never exceed the number of tracks
    walk = [(f, 50.0 + 10 * f, 1.0) for f in range(1, 21)]
    rows = rows_from({1: walk, 2: walk})
    rec = [{"track_id": 1, "reason": "lost"}] * 3 + [{"track_id": 2, "reason": "occluded"}]
    m = analysis.metrics(rows, 10.0, 20, [], rec, [])
    assert m["tracks"] == 2
    assert m["id_preserving_recoveries"] == 4
    assert m["reappeared_ids"] == 2
    assert m["reappeared_ids"] <= m["tracks"]


# ---------------------------------------------------------------- raw in, blurred out
def _blur_file(mode: str) -> Any:
    """A stand-in for serve.blur.blur_video that really blurs (with the real oval / square
    functions) the top third of every 'person', so blurred frames detect differently."""
    from serve.blur import blur_with

    def run(src: Path, dst: Path, model_dir: Path, **kw: Any) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(src))
        vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        det = FakeDetector()
        while True:
            ok, img = cap.read()
            if not ok:
                break
            tl, _ = det.detect(img)
            heads = np.array(
                [[x - 4, y - 4, x + w + 4, y + h / 3] for x, y, w, h in tl], np.float32
            )
            if len(heads):
                blur_with(mode)(img, heads.reshape(-1, 4))
            vw.write(img)
        cap.release()
        vw.release()
        return {"mode": mode}

    return run


def _upload_and_track(c: TestClient, vid: bytes) -> tuple[str, dict[str, Any]]:
    up = c.post("/api/uploads", files={"file": ("gate.mp4", vid, "video/mp4")}).json()
    job = wait(
        c, c.post("/api/jobs", json={"clip_id": up["id"], "gate": [0, 0.6, 1, 0.6]}).json()["id"]
    )
    assert job["status"] == "succeeded", job
    return up["id"], c.get(f"/api/jobs/{job['id']}/result").json()


def test_tracker_output_is_identical_whatever_the_blur_mode(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    c, s = env["client"], env["settings"]
    vid = (s.clip_dir / "walk.mp4").read_bytes()
    outs: dict[str, Any] = {}
    served: dict[str, bytes] = {}
    for mode in ("face", "head", "none"):
        monkeypatch.setattr("serve.blur.blur_video", _blur_file(mode))
        cid, res = _upload_and_track(c, vid)
        assert res["tracked_on"] == "raw"
        outs[mode] = (res["tracks"], [e for e in res["events"] if e["kind"] == "gate"])
        served[mode] = c.get(f"/api/clips/{cid}/video").content
    assert outs["face"] == outs["head"] == outs["none"]
    assert len(outs["none"][0]) == 2  # both people tracked
    # the blurred copies really differ, and detecting on them would not give the same boxes
    assert served["face"] != served["none"]
    assert served["head"] != served["none"]
    tmp = env["tmp"] / "served_head.mp4"
    tmp.write_bytes(served["head"])
    raw_cap, blur_cap = cv2.VideoCapture(str(s.clip_dir / "walk.mp4")), cv2.VideoCapture(str(tmp))
    differ = 0
    for _ in range(N):
        ok1, a = raw_cap.read()
        ok2, b = blur_cap.read()
        if not (ok1 and ok2):
            break
        da, db = FakeDetector().detect(a)[0], FakeDetector().detect(b)[0]
        differ += int(da.shape != db.shape or not np.allclose(da, db, atol=1))
    assert differ > 0


def test_raw_upload_is_deleted_when_its_first_job_ends(env: dict[str, Any]) -> None:
    c, s = env["client"], env["settings"]
    vid = (s.clip_dir / "walk.mp4").read_bytes()
    up = c.post("/api/uploads", files={"file": ("gate.mp4", vid, "video/mp4")}).json()
    d = s.uploads_dir / up["id"].removeprefix("upload-")
    assert (d / "original.mp4").exists()  # on disk only until the first job ends
    job = wait(c, c.post("/api/jobs", json={"clip_id": up["id"]}).json()["id"])
    assert job["status"] == "succeeded"
    assert not (d / "original.mp4").exists()
    assert (d / "blurred.mp4").exists()
    assert c.get(f"/api/clips/{up['id']}/video").status_code == 200
    assert c.get(f"/api/jobs/{job['id']}/result").status_code == 200
    r = c.post("/api/jobs", json={"clip_id": up["id"], "gate": [0, 0.5, 1, 0.5]})
    assert r.status_code == 410
    assert r.json()["error"]["code"] == "original_gone"


def test_raw_upload_is_deleted_when_its_job_fails(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    c, s = env["client"], env["settings"]

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("tracker exploded")

    monkeypatch.setattr("serve.api.run_video", boom)
    up = c.post(
        "/api/uploads",
        files={"file": ("gate.mp4", (s.clip_dir / "walk.mp4").read_bytes(), "video/mp4")},
    ).json()
    job = wait(c, c.post("/api/jobs", json={"clip_id": up["id"]}).json()["id"])
    assert job["status"] == "failed"
    assert not (s.uploads_dir / up["id"].removeprefix("upload-") / "original.mp4").exists()


def test_stale_raw_uploads_without_a_job_are_swept(env: dict[str, Any]) -> None:
    import os

    c, s = env["client"], env["settings"]
    vid = (s.clip_dir / "walk.mp4").read_bytes()
    up = c.post("/api/uploads", files={"file": ("gate.mp4", vid, "video/mp4")}).json()
    raw = s.uploads_dir / up["id"].removeprefix("upload-") / "original.mp4"
    old = time.time() - s.upload_raw_ttl_s - 5
    os.utime(raw, (old, old))
    fresh = c.post("/api/uploads", files={"file": ("b.mp4", vid, "video/mp4")}).json()  # sweeps
    assert not raw.exists()
    assert (s.uploads_dir / fresh["id"].removeprefix("upload-") / "original.mp4").exists()


def test_library_clips_are_tracked_on_the_raw_copy(env: dict[str, Any]) -> None:
    c, s = env["client"], env["settings"]
    # the served (blurred) copy shows nobody; the raw copy has the two walkers
    vw = cv2.VideoWriter(str(s.clip_dir / "walk.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(N):
        vw.write(np.zeros((H, W, 3), np.uint8))
    vw.release()
    job = wait(c, c.post("/api/jobs", json={"clip_id": "walk"}).json()["id"])
    res = c.get(f"/api/jobs/{job['id']}/result").json()
    assert res["tracked_on"] == "raw"
    assert len(res["tracks"]) == 2
    assert (s.raw_clip_dir / "walk.mp4").exists()  # library raw clips are not deleted

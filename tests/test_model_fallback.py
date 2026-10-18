"""Missing models on a fresh clone (e.g. Windows, no data/models): download on first use with a
sha256 check, and a head-region blur fallback when the YuNet face model cannot be loaded, so the
tracker still runs. The network is always mocked here."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from serve import blur, detector
from serve.api import Service
from serve.app import create_app
from serve.detector import ModelUnavailable, ensure_file
from serve.settings import Settings
from tests.test_serve_jobs import ROOT, FakeDetector, write_video

PAYLOAD = b"fake onnx bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    detector._verified.clear()
    monkeypatch.delenv("ZED_AUTO_DOWNLOAD", raising=False)


def test_missing_model_is_downloaded_into_a_new_dir(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, float]] = []

    def fake_download(url: str, tmp: Path, timeout: float) -> None:
        calls.append((url, timeout))
        tmp.write_bytes(PAYLOAD)

    monkeypatch.setattr(detector, "_download", fake_download)
    dst = tmp_path / "data" / "models" / "m.onnx"  # neither data/ nor models/ exists yet
    assert ensure_file(dst, "https://example.invalid/m.onnx", DIGEST) == dst
    assert dst.read_bytes() == PAYLOAD
    assert calls == [("https://example.invalid/m.onnx", detector.DOWNLOAD_TIMEOUT_S)]
    ensure_file(dst, "https://example.invalid/m.onnx", DIGEST)
    assert len(calls) == 1  # present and verified: no second download
    assert not list(dst.parent.glob("*.part"))


def test_wrong_sha256_is_rejected_and_nothing_is_left(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(detector, "_download", lambda u, t, s: t.write_bytes(b"evil"))
    with pytest.raises(ModelUnavailable, match="wrong sha256") as e:
        ensure_file(tmp_path / "m.onnx", "https://example.invalid/m.onnx", DIGEST)
    assert e.value.fix == detector.FETCH_FIX
    assert list(tmp_path.iterdir()) == []


def test_offline_and_disabled_download_give_a_clear_cause(tmp_path: Path, monkeypatch) -> None:
    def offline(url: str, tmp: Path, timeout: float) -> None:
        raise TimeoutError("timed out")

    monkeypatch.setattr(detector, "_download", offline)
    with pytest.raises(ModelUnavailable, match=r"could not download .*timed out"):
        ensure_file(tmp_path / "m.onnx", "https://example.invalid/m.onnx", DIGEST)
    monkeypatch.setenv("ZED_AUTO_DOWNLOAD", "0")
    with pytest.raises(ModelUnavailable, match="auto-download is off"):
        ensure_file(tmp_path / "m.onnx", "https://example.invalid/m.onnx", DIGEST)


def test_ensure_yunet_uses_the_official_url_and_hash(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []

    def dl(url: str, tmp: Path, timeout: float) -> None:
        seen.append(url)
        tmp.write_bytes(b"x")

    monkeypatch.setattr(detector, "_download", dl)
    with pytest.raises(ModelUnavailable):
        detector.ensure_yunet(tmp_path)
    assert seen == [detector.YUNET[1]]
    assert seen[0].startswith("https://github.com/opencv/opencv_zoo/")


def test_face_detector_falls_back_without_a_readable_yunet(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ZED_AUTO_DOWNLOAD", "0")
    fd, why = blur.face_detector(tmp_path, (320, 240))
    assert fd is None
    assert "face_detection_yunet_2023mar.onnx" in str(why)
    junk = tmp_path / "junk.onnx"
    junk.write_bytes(b"not an onnx model")
    monkeypatch.setattr(blur, "ensure_yunet", lambda d: junk)
    fd, why = blur.face_detector(tmp_path, (320, 240))  # the OpenCV 5 'Can't read ONNX' path
    assert fd is None
    assert f"OpenCV {cv2.__version__} could not load" in str(why)


def test_face_detector_accepts_the_old_factory_name(tmp_path, monkeypatch) -> None:
    made: list[tuple[Any, ...]] = []
    monkeypatch.setattr(blur, "ensure_yunet", lambda d: tmp_path / "y.onnx")
    monkeypatch.delattr(cv2, "FaceDetectorYN", raising=False)

    def create(*a: Any) -> str:
        made.append(a)
        return "fd"

    monkeypatch.setattr(cv2, "FaceDetectorYN_create", create, raising=False)
    fd, why = blur.face_detector(tmp_path, (320, 240))
    assert (fd, why) == ("fd", None)
    assert made[0][2] == (320, 240)


def test_blur_frame_fallback_blurs_heads_and_reports_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ZED_AUTO_DOWNLOAD", "0")
    monkeypatch.setattr(blur, "person_detector", lambda *a: FakeDetector())
    img = np.zeros((240, 320, 3), np.uint8)
    img[40:200, 100:160] = 255
    img[40:60, 100:160] = np.arange(60, dtype=np.uint8)[None, :, None] * 4  # detail in the head
    st: dict[str, Any] = {}
    out = blur.blur_frame(img, tmp_path, status=st)
    assert st["fallback_blur"] is True
    assert "yunet" in st["fallback_reason"]
    assert not np.array_equal(out[40:60, 100:160], img[40:60, 100:160])  # head region changed
    assert np.array_equal(out[150:200], img[150:200])  # legs untouched


def test_upload_job_runs_with_fallback_blur_and_counts(tmp_path, monkeypatch) -> None:
    """Panshul's Windows case: no data/models at all, YuNet unavailable -> still counts."""
    monkeypatch.setenv("ZED_AUTO_DOWNLOAD", "0")
    monkeypatch.setattr(blur, "person_detector", lambda *a: FakeDetector())
    monkeypatch.setattr(Service, "detector", lambda self, name: FakeDetector())
    s = Settings(
        data_dir=tmp_path / "serve",
        clip_dir=tmp_path / "none",
        raw_clip_dir=tmp_path / "none",
        demo_cut_dir=tmp_path / "none",
        clip_manifest=ROOT / "configs" / "demo" / "clips.yaml",
        model_dir=tmp_path / "data" / "models",  # does not exist
        results_dir=ROOT / "docs" / "results",
        config_root=ROOT / "configs",
        web_dist=tmp_path / "web",
    )
    vid = write_video(tmp_path / "gate.mp4").read_bytes()
    with TestClient(create_app(s)) as c:
        up = c.post("/api/uploads", files={"file": ("gate.mp4", vid, "video/mp4")}).json()
        r = c.get(f"/api/clips/{up['id']}/first-frame.jpg")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        job = c.post("/api/jobs", json={"clip_id": up["id"], "gate": [0, 0.6, 1, 0.6]}).json()
        t0 = time.time()
        while (j := c.get(f"/api/jobs/{job['id']}").json())["status"] not in (
            "succeeded",
            "failed",
        ) and time.time() - t0 < 60:
            time.sleep(0.1)
        assert j["status"] == "succeeded", j.get("error")
        res = c.get(f"/api/jobs/{job['id']}/result").json()
        assert res["blur"]["fallback_blur"] is True
        assert "yunet" in res["blur"]["fallback_reason"]
        crossings = [e for e in res["events"] if e.get("type") == "crossing"]
        assert len(crossings) >= 2  # both synthetic people cross the line
        assert c.get(f"/api/clips/{up['id']}/video").status_code == 200


def test_model_missing_job_error_carries_cause_and_fix(tmp_path, monkeypatch) -> None:
    def no_model(self: Service, name: str) -> Any:
        raise ModelUnavailable("yolox_s.onnx", "could not download x (offline)", "fix-cmd")

    monkeypatch.setattr(Service, "detector", no_model)
    from tests.test_serve_jobs import write_video as wv

    s = Settings(
        data_dir=tmp_path / "serve",
        clip_dir=tmp_path / "blurred",
        raw_clip_dir=tmp_path / "none",
        demo_cut_dir=tmp_path / "none",
        clip_manifest=ROOT / "configs" / "demo" / "clips.yaml",
        model_dir=tmp_path / "models",
        results_dir=ROOT / "docs" / "results",
        config_root=ROOT / "configs",
        web_dist=tmp_path / "web",
    )
    (tmp_path / "blurred").mkdir()
    import yaml

    first = yaml.safe_load(s.clip_manifest.read_text())["clips"][0]
    wv(tmp_path / "blurred" / first["file"])
    with TestClient(create_app(s)) as c:
        job = c.post("/api/jobs", json={"clip_id": first["id"]}).json()
        t0 = time.time()
        while (j := c.get(f"/api/jobs/{job['id']}").json())["status"] != "failed":
            assert time.time() - t0 < 30
            time.sleep(0.1)
        assert j["error"]["code"] == "model_missing"
        assert j["error"]["cause"] == "could not download x (offline)"
        assert j["error"]["fix"] == "fix-cmd"

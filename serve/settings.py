"""Service configuration, all from environment variables (defaults suit a local checkout).

| Variable | Default | Meaning |
|---|---|---|
| ``ZED_DATA_DIR`` | ``data/serve`` | job store and uploads (filesystem) |
| ``ZED_CLIP_DIR`` | ``data/demo/blurred`` | face-blurred demo clips (the only clips served) |
| ``ZED_RAW_CLIP_DIR`` | ``data/demo/raw`` | unblurred demo clips (tracker input, not served) |
| ``ZED_UPLOAD_BLUR`` | ``face`` | blur of an upload's served copy (``face``/``head``/``none``) |
| ``ZED_UPLOAD_RAW_TTL_S`` | ``3600`` | raw upload deleted after its first job, or after this |
| ``ZED_CLIP_MANIFEST`` | ``configs/demo/clips.yaml`` | clip metadata (consent, styling, gate) |
| ``ZED_MODEL_DIR`` | ``data/models`` | ONNX models (``python -m serve.detector fetch``) |
| ``ZED_RESULTS_DIR`` | ``docs/results`` | benchmark results served by ``/api/benchmarks`` |
| ``ZED_CONFIG_ROOT`` | ``configs`` | tracker configs |
| ``ZED_DETECTOR`` | ``yolox_s`` | default detector (``yolox_nano``, ``yolox_tiny``, ``yolox_s``) |
| ``ZED_DETECTOR_THREADS`` | ``4`` | onnxruntime intra-op threads |
| ``ZED_JOB_WORKERS`` | ``1`` | concurrent tracking jobs |
| ``ZED_MAX_VIDEO_MB`` | ``200`` | upload size limit for videos |
| ``ZED_MAX_VIDEO_SECONDS`` | ``120`` | upload duration limit |
| ``ZED_MAX_JOB_FRAMES`` | ``1800`` | hard cap on tracked frames per job |
| ``ZED_CORS_ORIGINS`` | (empty) | comma-separated origins for a separately hosted frontend |
| ``ZED_WEB_DIST`` | ``web/dist`` | built web app, served at ``/`` when present |
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


def _float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data" / "serve"
    clip_dir: Path = ROOT / "data" / "demo" / "blurred"
    # committed 720p face-blurred cuts (make_demo_pack.sh) for clips missing from clip_dir
    demo_cut_dir: Path = ROOT / "web" / "public" / "demo"
    # unblurred demo clips (gitignored): the tracker reads these; nothing from here is served
    raw_clip_dir: Path = ROOT / "data" / "demo" / "raw"
    clip_manifest: Path = ROOT / "configs" / "demo" / "clips.yaml"
    model_dir: Path = ROOT / "data" / "models"
    results_dir: Path = ROOT / "docs" / "results"
    config_root: Path = ROOT / "configs"
    web_dist: Path = ROOT / "web" / "dist"
    detector: str = "yolox_s"
    detector_threads: int = 4
    job_workers: int = 1
    max_video_mb: float = 200.0
    max_video_seconds: float = 120.0
    max_job_frames: int = 1800
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    upload_blur: str = "face"
    upload_raw_ttl_s: float = 3600.0

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            data_dir=_path("ZED_DATA_DIR", d.data_dir),
            clip_dir=_path("ZED_CLIP_DIR", d.clip_dir),
            demo_cut_dir=_path("ZED_DEMO_CUT_DIR", d.demo_cut_dir),
            raw_clip_dir=_path("ZED_RAW_CLIP_DIR", d.raw_clip_dir),
            clip_manifest=_path("ZED_CLIP_MANIFEST", d.clip_manifest),
            model_dir=_path("ZED_MODEL_DIR", d.model_dir),
            results_dir=_path("ZED_RESULTS_DIR", d.results_dir),
            config_root=_path("ZED_CONFIG_ROOT", d.config_root),
            web_dist=_path("ZED_WEB_DIST", d.web_dist),
            detector=os.environ.get("ZED_DETECTOR", d.detector),
            detector_threads=_int("ZED_DETECTOR_THREADS", d.detector_threads),
            job_workers=_int("ZED_JOB_WORKERS", d.job_workers),
            max_video_mb=_float("ZED_MAX_VIDEO_MB", d.max_video_mb),
            max_video_seconds=_float("ZED_MAX_VIDEO_SECONDS", d.max_video_seconds),
            max_job_frames=_int("ZED_MAX_JOB_FRAMES", d.max_job_frames),
            upload_blur=os.environ.get("ZED_UPLOAD_BLUR", d.upload_blur),
            upload_raw_ttl_s=_float("ZED_UPLOAD_RAW_TTL_S", d.upload_raw_ttl_s),
            cors_origins=tuple(
                o.strip() for o in os.environ.get("ZED_CORS_ORIGINS", "").split(",") if o.strip()
            ),
        )

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

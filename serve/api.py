"""Product API of the demo service: clips, uploads, tracking jobs, results, benchmarks.

* ``GET /api/clips``: demo clips that have a face-blurred copy (+ uploads once blurred).
* ``GET /api/clips/{id}``, ``/video`` (blurred file, HTTP range), ``/poster.jpg`` (a still).
* ``POST /api/uploads``: multipart ``file`` (mp4/mov/m4v/webm/mkv/avi) and ``consent`` (bool).
* ``POST /api/jobs``: JSON ``{clip_id, config?, detector?, stride?, max_seconds?, gate?,
  reuse?}`` → 202 (200 with the existing job when an identical one exists and ``reuse``).
* ``GET /api/jobs``, ``/api/jobs/{id}``: status, stage, progress, metrics, timing.
* ``GET /api/jobs/{id}/result``, ``/tracks.txt``, ``/events.json``: outputs (409 until done).
* ``POST /api/jobs/{id}/cancel``, ``DELETE /api/jobs/{id}``.
* ``GET /api/benchmarks``, ``/api/benchmarks/plots/{name}.png``: the MOT17 results in
  ``docs/results``.

Uploaded videos are face-blurred as the first stage of their first job, and only the blurred copy
is ever served back. Uploads default to neutral event styling; ``consent=true`` (the uploader
states that the people shown agreed) switches on the full alert contract.
"""

from __future__ import annotations

import json
import math
import re
import threading
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from serve.benchmarks import PLOTS, benchmarks
from serve.clips import DemoClip, load_manifest
from serve.detector import MODELS, YoloxDetector, model_path
from serve.errors import ApiError
from serve.jobs import TERMINAL, JobManager, JobStore, now
from serve.pipeline import JobOptions, VideoInfo, probe, run_video
from serve.settings import Settings
from tracker.config import load_config

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
VIDEO_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "video/x-matroska",
    "video/x-msvideo",
    "video/x-m4v",
    "application/octet-stream",
    "",
}
DEFAULT_CONFIG = "ours/visibility_botsort"
CONFIG_DIRS = ("baselines", "ours")


def available_configs(root: Path) -> dict[str, Path]:
    out = {}
    for d in CONFIG_DIRS:
        for p in sorted((root / d).glob("*.yaml")):
            out[f"{d}/{p.stem}"] = p
    return out


class JobRequest(BaseModel):
    clip_id: str = Field(min_length=1, max_length=80)
    config: str = DEFAULT_CONFIG
    detector: Literal["yolox_nano", "yolox_tiny", "yolox_s"] | None = None
    stride: int = Field(1, ge=1, le=6)
    max_seconds: float | None = Field(None, gt=0, le=600)
    gate: list[float] | None = None
    reuse: bool = True

    @field_validator("gate")
    @classmethod
    def _gate(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        if len(v) != 4 or not all(0.0 <= x <= 1.0 for x in v):
            raise ValueError("gate must be [x1, y1, x2, y2] in 0..1 (normalised)")
        if (v[0], v[1]) == (v[2], v[3]):
            raise ValueError("gate has zero length")
        return v


class Service:
    """Everything the routes need; one per app (tests build their own with tmp dirs)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = JobStore(settings.jobs_dir)
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        self.recovered = self.store.recover()
        self.manager = JobManager(self.store, self.run_job, settings.job_workers)
        self._detectors: dict[str, YoloxDetector] = {}
        self._det_lock = threading.Lock()
        self._info: dict[str, VideoInfo] = {}

    # ------------------------------------------------------------------ clips
    def manifest(self) -> list[DemoClip]:
        return load_manifest(self.settings.clip_manifest)

    def info(self, path: Path) -> VideoInfo:
        key = f"{path}:{path.stat().st_mtime_ns}"
        if key not in self._info:
            self._info[key] = probe(path)
        return self._info[key]

    def clip_record(self, c: DemoClip) -> dict[str, Any] | None:
        path = self.settings.clip_dir / c.file
        if not path.exists():
            return None
        i = self.info(path)
        return {
            "id": c.id,
            "title": c.title,
            "source": "library",
            "note": c.note,
            "page": c.page,
            "licence": c.licence,
            "consent": c.consent,
            "event_styling": c.event_styling,
            "blurred": True,
            "gate": list(c.gate) if c.gate else None,
            "hysteresis_px": c.hysteresis_px,
            "width": i.width,
            "height": i.height,
            "fps": round(i.fps, 3),
            "frames": i.frames,
            "duration_s": round(i.duration_s, 2),
            "video_url": f"/api/clips/{c.id}/video",
            "poster_url": f"/api/clips/{c.id}/poster.jpg",
        }

    def upload_dir(self, clip_id: str) -> Path:
        m = re.fullmatch(r"upload-([0-9a-f]{12})", clip_id)
        if not m:
            raise ApiError(404, "clip_not_found", f"no clip {clip_id!r}")
        return self.settings.uploads_dir / m.group(1)

    def upload_record(self, clip_id: str) -> dict[str, Any]:
        d = self.upload_dir(clip_id)
        if not (d / "meta.json").exists():
            raise ApiError(404, "clip_not_found", f"no clip {clip_id!r}")
        meta: dict[str, Any] = json.loads((d / "meta.json").read_text())
        blurred = (d / "blurred.mp4").exists()
        meta.update(
            blurred=blurred,
            video_url=f"/api/clips/{clip_id}/video" if blurred else None,
            poster_url=f"/api/clips/{clip_id}/poster.jpg" if blurred else None,
        )
        return meta

    def clips(self) -> list[dict[str, Any]]:
        out = [r for c in self.manifest() if (r := self.clip_record(c)) is not None]
        for d in sorted(self.settings.uploads_dir.glob("*/meta.json")):
            rec = self.upload_record("upload-" + str(d.parent.name))
            if rec["blurred"]:
                out.append(rec)
        return out

    def clip(self, clip_id: str) -> dict[str, Any]:
        if clip_id.startswith("upload-"):
            return self.upload_record(clip_id)
        for c in self.manifest():
            if c.id == clip_id:
                rec = self.clip_record(c)
                if rec is None:
                    raise ApiError(
                        404,
                        "clip_missing",
                        f"{clip_id}: no face-blurred copy on this server (run `make clips blur`)",
                    )
                return rec
        raise ApiError(404, "clip_not_found", f"no clip {clip_id!r}")

    def video_path(self, clip_id: str) -> Path:
        """The servable (blurred) file; never the original."""
        if clip_id.startswith("upload-"):
            p = self.upload_dir(clip_id) / "blurred.mp4"
            if not p.exists():
                raise ApiError(
                    409, "not_blurred", "the upload is served after its first job has blurred it"
                )
            return p
        rec = self.clip(clip_id)
        c = next(c for c in self.manifest() if c.id == rec["id"])
        return self.settings.clip_dir / c.file

    def poster(self, clip_id: str) -> Path:
        import cv2

        src = self.video_path(clip_id)
        out = self.settings.cache_dir / f"poster-{clip_id}-{src.stat().st_mtime_ns}.jpg"
        if not out.exists():
            cap = cv2.VideoCapture(str(src))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps))  # 1 s in
            ok, img = cap.read()
            cap.release()
            if not ok:
                raise ApiError(500, "poster_failed", "could not decode a frame")
            h, w = img.shape[:2]
            s = 1280 / max(w, h)
            if s < 1:
                img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return out

    # ------------------------------------------------------------------ uploads
    async def save_upload(self, upload: UploadFile, consent: bool) -> dict[str, Any]:
        s = self.settings
        name = Path(upload.filename or "").name
        ext = Path(name).suffix.lower()
        if ext not in VIDEO_EXTS:
            raise ApiError(
                415,
                "unsupported_type",
                f"file type {ext or '(none)'!r} not accepted; use one of {sorted(VIDEO_EXTS)}",
            )
        if (upload.content_type or "") not in VIDEO_TYPES and not (
            upload.content_type or ""
        ).startswith("video/"):
            raise ApiError(415, "unsupported_type", f"content type {upload.content_type!r}")
        uid = uuid.uuid4().hex[:12]
        d = s.uploads_dir / uid
        d.mkdir(parents=True)
        dst = d / f"original{ext}"
        limit, n = int(s.max_video_mb * 2**20), 0
        try:
            with dst.open("wb") as fh:
                while chunk := await upload.read(2**20):
                    n += len(chunk)
                    if n > limit:
                        raise ApiError(413, "too_large", f"video larger than {s.max_video_mb:g} MB")
                    fh.write(chunk)
            try:
                i = probe(dst)
            except ValueError as err:
                raise ApiError(422, "not_a_video", f"could not decode the video: {err}") from err
            if i.duration_s > s.max_video_seconds:
                raise ApiError(
                    422,
                    "too_long",
                    f"video is {i.duration_s:.1f} s; the limit is {s.max_video_seconds:g} s",
                )
        except BaseException:
            for p in d.iterdir():
                p.unlink()
            d.rmdir()
            raise
        title = re.sub(r"[^\w .-]", "", Path(name).stem)[:60] or "upload"
        meta = {
            "id": f"upload-{uid}",
            "title": title,
            "source": "upload",
            "note": "",
            "page": None,
            "licence": "uploaded by the user",
            "consent": consent,
            "event_styling": "alert" if consent else "neutral",
            "gate": None,
            "hysteresis_px": 10.0,
            "original": dst.name,
            "bytes": n,
            "width": i.width,
            "height": i.height,
            "fps": round(i.fps, 3),
            "frames": i.frames,
            "duration_s": round(i.duration_s, 2),
            "created_at": now(),
        }
        (d / "meta.json").write_text(json.dumps(meta))
        return self.upload_record(f"upload-{uid}")

    # ------------------------------------------------------------------ jobs
    def detector(self, name: str) -> YoloxDetector:
        with self._det_lock:
            if name not in self._detectors:
                p = model_path(name, self.settings.model_dir)
                if not p.exists():
                    raise RuntimeError(f"model {p} missing (run `python -m serve.detector fetch`)")
                det = YoloxDetector(p, MODELS[name][0], 0.1, threads=self.settings.detector_threads)
                det.name = name  # type: ignore[attr-defined]
                self._detectors[name] = det
            return self._detectors[name]

    def create_job(self, req: JobRequest) -> tuple[dict[str, Any], bool]:
        clip = self.clip(req.clip_id)
        cfgs = available_configs(self.settings.config_root)
        if req.config not in cfgs:
            raise ApiError(
                422, "unknown_config", f"unknown config {req.config!r}", known=list(cfgs)
            )
        if load_config(cfgs[req.config]).appearance.enabled:
            raise ApiError(422, "needs_gpu", f"{req.config} needs re-ID embeddings (GPU)")
        det = req.detector or self.settings.detector
        if det not in MODELS:
            raise ApiError(422, "unknown_detector", f"unknown detector {det!r}")
        stride = req.stride
        tracked = math.ceil(clip["frames"] / stride)
        if req.max_seconds:
            tracked = min(tracked, math.ceil(req.max_seconds * clip["fps"] / stride))
        tracked = min(tracked, self.settings.max_job_frames)
        gate = req.gate or clip.get("gate")
        opts = JobOptions(
            config=req.config,
            detector=det,
            stride=stride,
            max_frames=tracked,
            gate=tuple(gate) if gate else None,
            event_styling=clip["event_styling"],
            hysteresis_px=float(clip.get("hysteresis_px", 10.0)),
        )
        if req.reuse:
            for j in self.store.list():
                if (
                    j["status"] in ("queued", "running", "succeeded")
                    and j["clip"]["id"] == clip["id"]
                    and j["options"] == _opts_dict(opts)
                ):
                    return j, False
        summary = {
            k: clip[k]
            for k in (
                "id",
                "title",
                "source",
                "consent",
                "event_styling",
                "width",
                "height",
                "fps",
                "frames",
            )
        }
        summary["needs_blur"] = clip["source"] == "upload" and not clip["blurred"]
        rec = self.store.create(summary, opts)
        self.manager.submit(rec)
        return rec, True

    def run_job(self, rec: dict[str, Any], progress: Any) -> dict[str, Any]:
        o = rec["options"]
        opts = JobOptions(**{**o, "gate": tuple(o["gate"]) if o["gate"] else None})
        clip_id = rec["clip"]["id"]
        if clip_id.startswith("upload-"):
            d = self.upload_dir(clip_id)
            blurred = d / "blurred.mp4"
            if not blurred.exists():
                from serve.blur import blur_video

                meta = json.loads((d / "meta.json").read_text())
                blur_video(
                    d / meta["original"],
                    blurred,
                    self.settings.model_dir,
                    threads=self.settings.detector_threads,
                    progress=progress,
                )
            video = blurred
        else:
            video = self.video_path(clip_id)
        cfg = load_config(available_configs(self.settings.config_root)[opts.config])
        out = run_video(video, opts, cfg, self.detector(opts.detector), progress)
        out["clip"] = {**rec["clip"], "video_url": f"/api/clips/{clip_id}/video"}
        out["config"] = opts.config
        out["detector"] = {
            "name": opts.detector,
            "input": MODELS[opts.detector][0],
            "weights": "YOLOX COCO (Megvii release 0.1.1rc0)",
            "score_floor": 0.1,
        }
        return out

    def job(self, job_id: str) -> dict[str, Any]:
        try:
            return self.store.get(job_id)
        except KeyError:
            raise ApiError(404, "job_not_found", f"no job {job_id!r}") from None

    def job_file(self, job_id: str, name: str) -> Path:
        rec = self.job(job_id)
        if rec["status"] != "succeeded":
            raise ApiError(409, "job_not_done", f"job is {rec['status']}", job_status=rec["status"])
        return self.store.dir(job_id) / name


def _opts_dict(o: JobOptions) -> dict[str, Any]:
    from dataclasses import asdict

    d = asdict(o)
    d["gate"] = list(d["gate"]) if d["gate"] else None
    return d


router = APIRouter(prefix="/api")


def svc(request: Request) -> Service:
    s: Service = request.app.state.service
    return s


@router.get("/clips")
def list_clips(request: Request) -> list[dict[str, Any]]:
    return svc(request).clips()


@router.get("/clips/{clip_id}")
def get_clip(clip_id: str, request: Request) -> dict[str, Any]:
    return svc(request).clip(clip_id)


@router.get("/clips/{clip_id}/video")
def clip_video(clip_id: str, request: Request) -> FileResponse:
    return FileResponse(
        svc(request).video_path(clip_id),
        media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/clips/{clip_id}/poster.jpg")
def clip_poster(clip_id: str, request: Request) -> FileResponse:
    return FileResponse(svc(request).poster(clip_id), media_type="image/jpeg")


@router.post("/uploads", status_code=201)
async def upload(
    request: Request,
    file: Annotated[UploadFile, File(description="video file")],
    consent: Annotated[bool, Form()] = False,
) -> dict[str, Any]:
    return await svc(request).save_upload(file, consent)


@router.post("/jobs", status_code=202)
def create_job(req: JobRequest, request: Request) -> JSONResponse:
    rec, created = svc(request).create_job(req)
    return JSONResponse(rec, status_code=202 if created else 200)


@router.get("/jobs")
def list_jobs(
    request: Request, clip_id: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    jobs = svc(request).store.list()
    if clip_id:
        jobs = [j for j in jobs if j["clip"]["id"] == clip_id]
    return jobs[: max(1, min(limit, 500))]


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    return svc(request).job(job_id)


@router.get("/jobs/{job_id}/result")
def job_result(job_id: str, request: Request) -> FileResponse:
    return FileResponse(svc(request).job_file(job_id, "result.json"), media_type="application/json")


@router.get("/jobs/{job_id}/tracks.txt")
def job_tracks(job_id: str, request: Request) -> PlainTextResponse:
    p = svc(request).job_file(job_id, "tracks.txt")
    return PlainTextResponse(
        p.read_text(),
        headers={"Content-Disposition": f'attachment; filename="{job_id}.tracks.txt"'},
    )


@router.get("/jobs/{job_id}/events.json")
def job_events(job_id: str, request: Request) -> FileResponse:
    return FileResponse(svc(request).job_file(job_id, "events.json"), media_type="application/json")


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    s = svc(request)
    s.job(job_id)
    return s.manager.cancel(job_id)


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str, request: Request) -> None:
    s = svc(request)
    rec = s.job(job_id)
    if rec["status"] not in TERMINAL:
        raise ApiError(409, "job_active", "cancel the job before deleting it")
    s.store.delete(job_id)


@router.get("/benchmarks")
def get_benchmarks(request: Request) -> dict[str, Any]:
    return benchmarks(svc(request).settings.results_dir)


@router.get("/benchmarks/plots/{name}.png")
def benchmark_plot(name: str, request: Request) -> FileResponse:
    rel = PLOTS.get(name)
    p = svc(request).settings.results_dir / rel if rel else None
    if p is None or not p.exists():
        raise ApiError(404, "plot_not_found", f"no plot {name!r}")
    return FileResponse(p, media_type="image/png")

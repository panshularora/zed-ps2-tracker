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

Detection and tracking always read the unblurred frames (in memory): the raw upload, or the raw
demo clip in ``raw_clip_dir``. Blurring applies only to what is shown or saved: the served video,
the first-frame still, the poster and the annotated render are made from the face-blurred copy,
so the blur mode can never change a count. Uploaded videos are face-blurred as the first stage
of their first job, and only the blurred copy is ever served back. The raw upload is deleted when
that first job ends (succeeded, failed or cancelled), or after ``upload_raw_ttl_s`` if no job ever
runs; later jobs on the same upload get 410 ``original_gone`` (upload it again).
Uploads default to neutral event styling; ``consent=true`` (the uploader
states that the people shown agreed) switches on the full alert contract.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from serve.benchmarks import PLOTS, benchmarks, gpu_plot_path
from serve.clips import DemoClip, load_manifest
from serve.detector import MODELS, ModelUnavailable, YoloxDetector, ensure_yolox
from serve.errors import ApiError
from serve.jobs import TERMINAL, JobManager, JobStore, now
from serve.pipeline import (
    JobOptions,
    VideoInfo,
    close_capture,
    open_capture,
    probe,
    run_video,
    unlink_locked,
)
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

    def clip_file(self, c: DemoClip) -> tuple[Path, bool] | None:
        """The blurred full clip, else the committed blurred demo cut; (path, is_demo_cut)."""
        full = self.settings.clip_dir / c.file
        if full.exists():
            return full, False
        cut = self.settings.demo_cut_dir / f"{c.id}.mp4"
        if cut.exists():
            return cut, True
        return None

    def clip_record(self, c: DemoClip) -> dict[str, Any] | None:
        found = self.clip_file(c)
        if found is None:
            return None
        path, demo_cut = found
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
            "demo_cut": demo_cut,
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
        found = self.clip_file(c)
        if found is None:
            raise ApiError(404, "clip_missing", f"clip {clip_id} has no video on this server")
        return found[0]

    def raw_path(self, clip_id: str) -> Path | None:
        """The unblurred source the tracker reads (never served): the raw upload, or the raw demo
        clip. None for a library clip whose raw file is not on this server."""
        if clip_id.startswith("upload-"):
            d = self.upload_dir(clip_id)
            meta = json.loads((d / "meta.json").read_text())
            p = d / str(meta["original"])
            if not p.exists():
                raise ApiError(
                    410, "original_gone",
                    "the original of this upload was deleted after its first job; upload it again",
                )  # fmt: skip
            return p
        rec = self.clip(clip_id)
        c = next(c for c in self.manifest() if c.id == rec["id"])
        p = self.settings.raw_clip_dir / c.file
        return p if p.exists() and not rec.get("demo_cut") else None

    def drop_original(self, clip_id: str) -> None:
        """Delete an upload's raw file (its blurred copy, stills and job outputs stay)."""
        d = self.upload_dir(clip_id)
        meta = json.loads((d / "meta.json").read_text())
        unlink_locked(d / meta["original"], missing_ok=True)

    def sweep_originals(self) -> int:
        """Delete raw uploads older than ``upload_raw_ttl_s`` (no job ran on them). Returns n."""
        n, now_s = 0, time.time()
        for meta in self.settings.uploads_dir.glob("*/meta.json"):
            m = json.loads(meta.read_text())
            p = meta.parent / m["original"]
            if p.exists() and now_s - p.stat().st_mtime > self.settings.upload_raw_ttl_s:
                unlink_locked(p)
                n += 1
        return n

    def blur_status(self, clip_id: str) -> dict[str, Any]:
        """How the shown copy was blurred: ``fallback_blur`` = person-box heads, no face model."""
        if not clip_id.startswith("upload-"):
            return {"mode": "prepared demo copy", "fallback_blur": False, "fallback_reason": None}
        f = self.upload_dir(clip_id) / "blur.json"
        if f.is_file():
            return {"mode": self.settings.upload_blur, **json.loads(f.read_text())}
        return {"mode": self.settings.upload_blur, "fallback_blur": False, "fallback_reason": None}

    def first_frame(self, clip_id: str) -> Path:
        """A face-blurred still of the first frame, for drawing the gate before the first job.

        Uploads: the original's frame 0 goes through the face blur before it is written; the
        original itself is still never served. Library clips: frame 0 of the blurred copy.
        """
        import cv2

        from serve import blur

        if clip_id.startswith("upload-"):
            d = self.upload_dir(clip_id)
            rec = self.upload_record(clip_id)
            out = d / "first_frame.jpg"
            if out.exists():
                return out
            cap = open_capture(self.raw_path(clip_id))
            try:
                ok, img = cap.read()
            finally:
                close_capture(cap)
                cap = None
            if not ok:
                raise ApiError(500, "frame_failed", f"could not decode a frame of {rec['id']}")
            # blurred in memory before anything is written
            status: dict[str, Any] = {}
            try:
                img = blur.blur_frame(
                    img,
                    self.settings.model_dir,
                    mode=self.settings.upload_blur,
                    threads=self.settings.detector_threads,
                    status=status,
                )
            except ModelUnavailable as err:
                fix, name = err.fix, err.name
                raise ApiError(503, "model_missing", err.cause, fix=fix, model=name) from err
            (d / "blur.json").write_text(json.dumps(status))
        else:
            src = self.video_path(clip_id)
            out = self.settings.cache_dir / f"first-{clip_id}-{src.stat().st_mtime_ns}.jpg"
            if out.exists():
                return out
            cap = open_capture(src)
            try:
                ok, img = cap.read()
            finally:
                close_capture(cap)
                cap = None
            if not ok:
                raise ApiError(500, "frame_failed", "could not decode a frame")
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return out

    def poster(self, clip_id: str) -> Path:
        import cv2

        src = self.video_path(clip_id)
        out = self.settings.cache_dir / f"poster-{clip_id}-{src.stat().st_mtime_ns}.jpg"
        if not out.exists():
            cap = open_capture(src)
            try:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps))  # 1 s in
                ok, img = cap.read()
            finally:
                close_capture(cap)
                cap = None
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
        self.sweep_originals()
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
            # OpenCV / Defender can still hold original.mp4; never replace the real API error
            # (not_a_video, too_long, …) with WinError 32 from this cleanup.
            for p in list(d.iterdir()):
                with contextlib.suppress(OSError):
                    unlink_locked(p)
            with contextlib.suppress(OSError):
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
                p = ensure_yolox(name, self.settings.model_dir)  # fetched on first use
                det = YoloxDetector(p, MODELS[name][0], 0.1, threads=self.settings.detector_threads)
                det.name = name  # type: ignore[attr-defined]
                self._detectors[name] = det
            return self._detectors[name]

    def create_job(self, req: JobRequest) -> tuple[dict[str, Any], bool]:
        clip = self.clip(req.clip_id)
        if clip["source"] == "upload":
            self.raw_path(clip["id"])  # 410 once the original is gone
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
        upload = clip_id.startswith("upload-")
        try:
            raw = self.raw_path(clip_id)
            if upload:
                assert raw is not None
                blurred = self.upload_dir(clip_id) / "blurred.mp4"
                if not blurred.exists():  # the copy that is shown; the tracker never reads it
                    from serve.blur import blur_video

                    info = blur_video(
                        raw,
                        blurred,
                        self.settings.model_dir,
                        threads=self.settings.detector_threads,
                        progress=progress,
                        mode=self.settings.upload_blur,
                    )
                    (self.upload_dir(clip_id) / "blur.json").write_text(json.dumps(
                        {k: info.get(k) for k in ("mode", "fallback_blur", "fallback_reason")}
                    ))  # fmt: skip
            cfg = load_config(available_configs(self.settings.config_root)[opts.config])
            # detection + tracking on the unblurred frames (decoded in memory, never written)
            video = raw if raw is not None else self.video_path(clip_id)
            out = run_video(video, opts, cfg, self.detector(opts.detector), progress)
        finally:
            if upload:
                self.drop_original(clip_id)
        out["tracked_on"] = "raw" if raw is not None else "blurred (no raw copy on this server)"
        out["blur"] = self.blur_status(clip_id)
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


@router.get("/clips/{clip_id}/first-frame.jpg")
def clip_first_frame(clip_id: str, request: Request) -> FileResponse:
    return FileResponse(svc(request).first_frame(clip_id), media_type="image/jpeg")


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


@router.get("/jobs/{job_id}/tracks.csv")
def job_tracks_csv(job_id: str, request: Request) -> PlainTextResponse:
    """Per-box CSV for spreadsheets: tracked frame, video time, track id, box, score, visibility."""
    s = svc(request)
    res = json.loads(s.job_file(job_id, "result.json").read_text())
    stride, fps = int(res["video"]["stride"]), float(res["video"]["fps"])
    lines = ["frame,time_s,track_id,x,y,w,h,score,visibility"]
    rows = sorted((int(r[0]), int(tid), r) for tid, rs in res["tracks"].items() for r in rs)
    for f, tid, r in rows:
        t = (f - 1) * stride / fps
        lines.append(f"{f},{t:.3f},{tid},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}")
    return PlainTextResponse(
        "\n".join(lines) + "\n",
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.tracks.csv"'},
    )


@router.get("/jobs/{job_id}/annotated.mp4")
def job_annotated(job_id: str, request: Request) -> FileResponse:
    """The blurred clip with this job's boxes, numbers and gate line drawn in (rendered once)."""
    from serve.render import render_annotated

    s = svc(request)
    res_path = s.job_file(job_id, "result.json")
    out = res_path.parent / "annotated.mp4"
    if not out.exists():
        res = json.loads(res_path.read_text())
        render_annotated(s.video_path(res["clip"]["id"]), res, out)
    return FileResponse(
        out,
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.annotated.mp4"'},
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
    root = svc(request).settings.results_dir
    rel = PLOTS.get(name)
    p = root / rel if rel else gpu_plot_path(root, name)
    if p is None or not p.exists():
        raise ApiError(404, "plot_not_found", f"no plot {name!r}")
    return FileResponse(p, media_type="image/png")

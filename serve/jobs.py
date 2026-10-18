"""Filesystem job store and a background worker pool for tracking jobs.

Layout under ``<ZED_DATA_DIR>/jobs/<job id>/``:

* ``job.json``     status record (written atomically on every change)
* ``result.json``  the full result (tracks, events, metrics, timing): ``serve.pipeline``
* ``tracks.txt``   MOTChallenge format ``frame,id,x,y,w,h,score,-1,-1,-1`` (tracked frames)
* ``events.json``  gate events (contract for consented clips, crossings only otherwise),
  recoveries and heuristic ID-switch flags

Statuses: ``queued → running → succeeded | failed | cancelled``. A job that was running when the
process stopped is marked ``failed`` (``interrupted``) on the next start. Jobs run in a thread
pool (``ZED_JOB_WORKERS``, default 1: the detector already uses several cores).
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from serve.detector import ModelUnavailable
from serve.pipeline import JobOptions

STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
TERMINAL = ("succeeded", "failed", "cancelled")


class Cancelled(Exception):
    pass


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.replace(path)


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def dir(self, job_id: str) -> Path:
        if not job_id.isalnum():
            raise KeyError(job_id)
        return self.root / job_id

    def create(self, clip: dict[str, Any], opts: JobOptions) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        rec = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "clip": clip,
            "options": asdict(opts),
            "created_at": now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "metrics": None,
            "timing": None,
        }
        self.dir(job_id).mkdir(parents=True)
        self.save(rec)
        return rec

    def save(self, rec: dict[str, Any]) -> None:
        with self._lock:
            _write_json(self.dir(rec["id"]) / "job.json", rec)

    def get(self, job_id: str) -> dict[str, Any]:
        try:
            path = self.dir(job_id) / "job.json"
        except KeyError:
            raise KeyError(job_id) from None
        if not path.exists():
            raise KeyError(job_id)
        rec: dict[str, Any] = json.loads(path.read_text())
        return rec

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            path = self.dir(job_id) / "job.json"
            rec: dict[str, Any] = json.loads(path.read_text())
            rec.update(fields)
            _write_json(path, rec)
        return rec

    def list(self) -> list[dict[str, Any]]:
        out = []
        for p in self.root.glob("*/job.json"):
            try:
                out.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(out, key=lambda r: r["created_at"], reverse=True)

    def delete(self, job_id: str) -> None:
        d = self.dir(job_id)
        for p in sorted(d.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        d.rmdir()

    def recover(self) -> int:
        """Mark jobs left ``queued``/``running`` by a previous process as failed."""
        n = 0
        for rec in self.list():
            if rec["status"] not in TERMINAL:
                self.update(
                    rec["id"],
                    status="failed",
                    stage="interrupted",
                    finished_at=now(),
                    error={
                        "code": "interrupted",
                        "message": "the service restarted while this job ran",
                    },
                )
                n += 1
        return n

    def write_outputs(self, job_id: str, result: dict[str, Any]) -> None:
        d = self.dir(job_id)
        rows = result.pop("mot_rows")
        lines = [
            f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},"
            f"{r[6]:.4f},-1,-1,-1"
            for r in np.asarray(rows)
        ]
        (d / "tracks.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        _write_json(d / "events.json", result["events"])
        _write_json(d / "result.json", result)


Runner = Callable[[dict[str, Any], Callable[[str, int, int], None]], dict[str, Any]]
"""``runner(job record, progress) -> result``: the job body (injected; tests use a fake)."""

# progress weight of each stage in the overall 0..1 bar
STAGE_SPAN = {"blur": (0.0, 0.35), "detect": (0.0, 0.95), "track": (0.95, 0.99)}


class JobManager:
    def __init__(self, store: JobStore, runner: Runner, workers: int = 1) -> None:
        self.store = store
        self.runner = runner
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="zed-job")
        self._cancel: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, rec: dict[str, Any]) -> None:
        self.pool.submit(self._run, rec["id"])

    def cancel(self, job_id: str) -> dict[str, Any]:
        rec = self.store.get(job_id)
        if rec["status"] in TERMINAL:
            return rec
        with self._lock:
            self._cancel.add(job_id)
        if rec["status"] == "queued":
            return self.store.update(
                job_id, status="cancelled", stage="cancelled", finished_at=now()
            )
        return rec

    def _run(self, job_id: str) -> None:
        rec = self.store.get(job_id)
        if rec["status"] != "queued":
            return
        t0 = time.perf_counter()
        has_blur = bool(rec["clip"].get("needs_blur"))
        last = [0.0]

        def progress(stage: str, done: int, total: int) -> None:
            with self._lock:
                if job_id in self._cancel:
                    raise Cancelled()
            lo, hi = STAGE_SPAN.get(stage, (0.0, 1.0))
            if stage == "detect" and has_blur:
                lo, hi = 0.35, 0.95
            p = lo + (hi - lo) * (done / total if total else 0.0)
            now_t = time.perf_counter()
            if now_t - last[0] > 0.25 or done == total:  # throttle disk writes
                last[0] = now_t
                self.store.update(
                    job_id,
                    stage=stage,
                    progress=round(p, 4),
                    frames_done=done if stage == "detect" else None,
                    frames_total=total if stage == "detect" else None,
                )

        self.store.update(job_id, status="running", stage="starting", started_at=now())
        try:
            result = self.runner(rec, progress)
            self.store.write_outputs(job_id, result)
            self.store.update(
                job_id,
                status="succeeded",
                stage="done",
                progress=1.0,
                finished_at=now(),
                metrics=result.get("metrics"),
                timing={
                    **(result.get("timing") or {}),
                    "job_s": round(time.perf_counter() - t0, 2),
                },
            )
        except Cancelled:
            self.store.update(job_id, status="cancelled", stage="cancelled", finished_at=now())
        except Exception as err:  # the job fails, the service keeps running
            code = "invalid_input" if isinstance(err, ValueError) else "job_failed"
            extra: dict[str, Any] = {}
            if isinstance(err, ModelUnavailable):  # missing model / ffmpeg: say what to run
                code, extra = "model_missing", {"cause": err.cause, "fix": err.fix}
            self.store.update(
                job_id,
                status="failed",
                stage="failed",
                finished_at=now(),
                error={
                    "code": code,
                    "message": str(err) or type(err).__name__,
                    **extra,
                    "trace": traceback.format_exc(limit=3),
                },
            )
        finally:
            with self._lock:
                self._cancel.discard(job_id)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

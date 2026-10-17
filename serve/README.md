# serve/: CPU demo service

FastAPI app with two parts:

1. **Video jobs** (`serve/api.py`): pick a face-blurred demo clip or upload a video, run a CPU
   person detector (YOLOX ONNX) and the repo's tracker on it as a background job, then fetch MOT
   tracks, gate events, occlusion recoveries and per-job metrics. This is what the web app
   (`web/`) uses.
2. **Detection-cache replay** (`serve/app.py`, `POST /api/track`): replays a Stage A `.npz`
   cache through any tracker config and returns the **gate-event contract** for ZIG
   (`serve/events.py`, `docs/integration.md` §2). Unchanged from PR #19.

It also serves the **benchmark results** from `docs/results` (`GET /api/benchmarks`), so every
number in the UI comes from a repo file or a real job run.

```bash
uv pip install -e ".[serve]"       # or: pip install -r serve/requirements.txt
make models clips blur             # ONNX models, Pexels clips, face-blurred copies (~7 min CPU)
make serve                         # http://127.0.0.1:8000  (API docs at /docs)

curl -s localhost:8000/api/clips
curl -s -XPOST localhost:8000/api/jobs -H 'content-type: application/json' \
     -d '{"clip_id": "subway_turnstile"}'                       # -> {"id": "...", "status": "queued"}
curl -s localhost:8000/api/jobs/<id>                             # status, stage, progress, metrics
curl -s localhost:8000/api/jobs/<id>/tracks.txt                  # MOTChallenge rows
```

## Video jobs

`POST /api/jobs` (JSON):

| field | default | |
|---|---|---|
| `clip_id` | (required) | a library clip id or an `upload-…` id |
| `config` | `ours/visibility_botsort` | any name from `/api/configs` that needs no re-ID (re-ID needs the GPU) |
| `detector` | `ZED_DETECTOR` (`yolox_s`) | `yolox_nano` (416), `yolox_tiny` (416), `yolox_s` (640) |
| `stride` | 1 | run on every n-th frame (1–6); the tracker then runs at fps / n |
| `max_seconds` | whole clip | limit the clip length; hard cap `ZED_MAX_JOB_FRAMES` tracked frames |
| `gate` | the clip's line, else 60 % height | `[x1, y1, x2, y2]` normalised 0..1 |
| `reuse` | true | return the existing queued/running/finished job with identical options (200) |

Job body (`serve/pipeline.py`): OpenCV decode → YOLOX (score ≥ 0.1, persons) → a Stage A-style
cache (no embeddings) → `tracker.runner.run_sequence` → events and metrics (`serve/analysis.py`).
Jobs run in a thread pool (`ZED_JOB_WORKERS`, default 1) and are stored on disk
(`ZED_DATA_DIR/jobs/<id>/`: `job.json`, `result.json`, `tracks.txt`, `events.json`). A job that
was running when the process stopped is marked `failed` / `interrupted` on restart.
`POST /api/jobs/{id}/cancel` stops it at the next frame.

Events in `result.json` (`kind` = `gate` | `recovery` | `identity`):

| type | meaning |
|---|---|
| `crossing` | the box's bottom-centre changed side of the gate line (hysteresis per clip) |
| `tailgate`, `ambiguous_hold` | **only for consented footage** (`event_styling: alert`): the full ZIG contract of `serve/events.py` |
| `recovery` | a track missing ≥ 3 frames (`lost`) or with estimated visibility < 0.5 (`occluded`) came back **with the same ID**: an ID-preserving recovery |
| `suspected_id_switch` | heuristic, no ground truth: a track ended and a new ID was born within 1 s, close to where it was last seen |

Metrics: tracks (≥ 3 frames), average / median track length, ID-preserving recoveries (lost /
occluded), suspected ID switches, crossings in / out, detector and tracker p50 / p95 ms, wall time.
**These describe the tracker's behaviour on a clip without ground truth; they are not accuracy.**
Accuracy is the benchmark (`/api/benchmarks`).

## Clips, uploads and privacy

* `GET /api/clips` lists only clips that have a **face-blurred** copy in `ZED_CLIP_DIR`
  (`make blur`); `/api/clips/{id}/video` serves that copy (HTTP range). The originals are never
  served.
* Pexels clips are `consent: false`, `event_styling: neutral`: crossings and counts only, no
  tailgate / hold alerts (`docs/licenses.md`, "Demo clips").
* `POST /api/uploads` (multipart `file`, `consent`): mp4 / mov / m4v / webm / mkv / avi, at most
  `ZED_MAX_VIDEO_MB` (200) and `ZED_MAX_VIDEO_SECONDS` (120), must decode. The first job on an
  upload writes its face-blurred copy (`serve/blur.py`, `ZED_UPLOAD_BLUR`, default `face`: soft
  ovals over faces, bodies sharp); only the blurred copy is served.
  `consent=true` (the uploader states that the people shown agreed) enables the alert contract.

### Raw in, blurred out

* **Detection and tracking read the unblurred frames**, decoded in memory: the raw upload, or the
  raw demo clip in `ZED_RAW_CLIP_DIR` (`data/demo/raw`, gitignored, from `make clips`). The blur
  mode therefore never changes a box, an ID or a count (`tests/test_serve_jobs.py::
  test_tracker_output_is_identical_whatever_the_blur_mode`, face / head / none). A result says
  `"tracked_on": "raw"`; if a library clip has no raw copy on the server it is tracked on the
  blurred copy and the result says so.
* **Everything shown or saved is blurred first**: the served video, the first-frame still for
  drawing the gate (blurred in memory before it is written), the poster, the annotated render and
  the browser-side row stills (grabbed from the served blurred video). No raw pixels are written
  by the service apart from the uploaded file itself.
* **Lifetime of a raw upload**: written to `ZED_DATA_DIR/uploads/<id>/original.<ext>` on upload;
  read for the first-frame still and by the first job; **deleted when that first job ends**
  (succeeded, failed or cancelled while running). An upload that never gets a job is deleted
  after `ZED_UPLOAD_RAW_TTL_S` (3600 s), checked on every new upload. Later jobs on the same
  upload get 410 `original_gone`: upload the clip again. The blurred copy, the stills and the job
  outputs (boxes, events) stay until the upload / job is deleted.

## Errors

Every error is `{"error": {"code", "message", ...}, "detail": message}` with the HTTP status:
404 `clip_not_found` / `job_not_found`, 409 `job_not_done` / `not_blurred`, 410 `original_gone`,
413 `too_large`,
415 `unsupported_type`, 422 `invalid_input` / `unknown_config` / `needs_gpu` / `not_a_video` /
`too_long`. A failed job keeps `error: {code, message}` in its record.

## Configuration (environment)

See `serve/settings.py`: `ZED_DATA_DIR`, `ZED_CLIP_DIR`, `ZED_CLIP_MANIFEST`, `ZED_MODEL_DIR`,
`ZED_RESULTS_DIR`, `ZED_CONFIG_ROOT`, `ZED_WEB_DIST`, `ZED_DETECTOR`, `ZED_DETECTOR_THREADS`,
`ZED_JOB_WORKERS`, `ZED_MAX_VIDEO_MB`, `ZED_MAX_VIDEO_SECONDS`, `ZED_MAX_JOB_FRAMES`,
`ZED_CORS_ORIGINS`; plus `ZED_MAX_UPLOAD_MB` (cache uploads of `/api/track`) and
`ZED_VIEWER_DIST`.

## Detection-cache replay (`/api/track`)

| Endpoint | |
|---|---|
| `GET /api/health` | status, version, configs, detector models present, clips, job counts, limits |
| `GET /api/configs` | names accepted by `config` (`baselines/*`, `ours/*`); paths are never accepted |
| `GET /api/workstation` | viewer-clip count and which published result files are present |
| `GET /api/config/{name}` | parsed YAML for a name from `/api/configs` |
| `GET /api/viewer-clips` | MOT17 viewer clips in `viewer/data`; `GET /api/viewer-clips/{file}` the gzip |
| `GET /api/results/ablation` | `docs/results/ablation/ablation.json` |
| `GET /api/results/gate-events` | `docs/results/gate_events/gate_events.json` |
| `GET /api/results/go-no-go` | `docs/results/go_no_go/rows_v2.json` |
| `GET /api/outputs` | COMBINED metrics from local `outputs/*/metrics/*/metrics.json` |
| `POST /api/track` | multipart `cache` (.npz, ≤ `ZED_MAX_UPLOAD_MB`, default 200), `config`, optional `gate` `x1,y1,x2,y2`, optional `start_time` (ISO 8601 with offset) |

```bash
curl -s -F cache=@data/stagea/<...>/MOT17-04-FRCNN.npz -F config=ours/visibility \
     -F gate=0,648,1920,648 -F start_time=2026-10-05T09:00:00+05:30 localhost:8000/api/track
```

Response: `{"sequence", "config", "events": [...], "latency": {"frames", "p50_ms", "p95_ms",
"machine"}, "clip": {viewer clip}}`. In the clip, crossings have `correct: null` because there is
no ground truth for an uploaded cache.

## Tests

`tests/test_serve.py` (event contract, `/api/track`), `tests/test_serve_jobs.py` (clips, uploads,
job lifecycle, analysis, benchmarks, errors; synthetic MP4 + fake detector) and
`tests/test_demo_clips.py` (manifest rules, YOLOX decoding, face blurring, keyframe interval).

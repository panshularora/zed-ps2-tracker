# Status: what is done, what waits for the GPU, what was cut

Audit of every open item in `docs/` (roadmap, placeholders, ⟨GPU⟩ markers, "not built yet"
notes) as of 9 Oct 2026. **GPU-only** items need the DGX session (YOLOX-X / re-ID caches) and are
deliberately not attempted on CPU; the web app labels them "pending DGX run".

## Done in this round (CPU)

| Item | Where it was open | Now |
|---|---|---|
| Product frontend (was `dashboard/`, "React + Vite live overlay, later phase") | `dashboard/README.md`, roadmap phase 3 | `web/`: Overview, Demo, Results, Architecture; vitest, eslint, tsc |
| Demo / deploy placeholder | `deploy/README.md` | `docker-compose.yml` (backend + nginx frontend), `Dockerfile` builds the web app too |
| Video jobs on CPU (detector + tracker on a real clip) | roadmap phase 3 "live pipeline prototype" | `serve/`: uploads, background jobs with progress and cancel, MOT tracks, gate events, recoveries, per-job metrics, benchmarks API; YOLOX ONNX on onnxruntime |
| Demo footage | `docs/research/video_and_data_sources.md` (Pexels links) | `make models clips blur`: sha256-checked models, 5 Pexels clips, face-blurred copies (YuNet + head regions); only blurred copies are served; credits in `NOTICE` and `docs/licenses.md` |
| Presentation-mode overlays in `docs/demo_video_script.md` ("not built yet") | gate-line pulse, counters, event cards | partly: the web Demo page has the gate pulse, live in/out counters, the recovery ring and an event log. Still missing: swap flash, per-tracker ID-switch counters, sliding JSON cards (composite in editing, or add later) |
| CI for the new code | `ci/github-actions.yml` | moved to `docs/ci/github-actions.yml` with `web` and `docker` jobs; still needs adding as `.github/workflows/ci.yml` by hand (token lacks `workflow` scope) |
| `make demo`, `make web`, `make test`, README quickstart with screenshots | task brief | done; screenshots in `docs/media/web_*.jpg` from a real turnstile run |

## GPU-only (skipped: needs the DGX run)

| Item | Source |
|---|---|
| YOLOX-X and re-ID rows for MOT17 / MOT20 / DanceTrack (ablation, stress, gate events) | `docs/next_gpu_session.md`, `configs/phase3/plan.yaml` |
| ByteTrack reproduction vs the paper (MOTA 76.6 / IDF1 79.3) | roadmap phase 2a step 1, deck A2 |
| Go/no-go v2 (delayed look-ahead with re-ID), visibility-gated appearance | `docs/preregistration/`, roadmap 2a.3 |
| Detector latency on the edge box / GPU, ⟨GPU⟩ placeholders in `docs/integration.md`, `docs/deck_outline.md`, `docs/demo_video_script.md` | those files |
| Real-frame re-shoot of the MOT17 viewer screenshots | `docs/next_gpu_session.md` step 7 |

## Cut or deferred (with reasons)

| Item | Why |
|---|---|
| Live RTSP / webcam stream prototype | The CPU box runs YOLOX-S at ~16 fps on 1080p, so "live" would mean dropped frames or a toy model; the job service processes clips at the same per-frame cost and is honest about it. A stream adapter is a small wrapper over `serve/pipeline.py` once a GPU or edge box is available |
| Tailgate / fare-evasion / "swap" alerts on the demo clips | The Pexels clips have no consent from the people in them; they are shown with neutral styling only (IDs, crossings, counts, recoveries). Alert styling exists only for uploads whose uploader ticks "people in my video consented" |
| Benchmark numbers from the demo clips | No ground truth; the Demo page reports behaviour (tracks, recoveries, crossings, CPU time), never accuracy |
| Explainer / code-walkthrough videos, PPT | Need the GPU numbers and a human narrator; scripts and outline exist (`docs/demo_video_script.md`, `docs/deck_outline.md`) |
| Recording the consented gate clip | Needs a camera, a gate and consent (Panshul) |

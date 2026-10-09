# Modules: file by file

The layout is described in [architecture.md](architecture.md). Every Python module starts with a
docstring that is the authoritative description; this page is the map.

## `tracker/`: the online tracker (NumPy + SciPy only, `mypy --strict`)

| File | Responsibility | Key names |
|---|---|---|
| `types.py` | value types and box conversions shared by all components | `Detections`, `TrackOutput`, `PendingDecision`, `FrameResult`, `tlwh_to_xyxy` … |
| `kalman.py` | constant-velocity Kalman filter in `xywh` (BoT-SORT noise model), vectorised predict, Mahalanobis gating | `KalmanFilterXYWH`, `KalmanConfig`, `CHI2INV95` |
| `cost.py` | cost matrices (rows = tracks, cols = detections, lower = better) | `iou_matrix`, `iou_cost`, `fuse_score`, `cosine_cost`, `fuse_iou_appearance`, `gate_cost` |
| `assignment.py` | thresholded rectangular assignment (scipy or `lap`, same results); the `Assigner` protocol that the delayed assignment will implement | `linear_assignment`, `HungarianAssigner`, `Matching`, `Assigner` |
| `lifecycle.py` | explicit track FSM with validated transitions, plus the per-track record | `TrackState`, `ALLOWED_TRANSITIONS`, `Track`, `LifecycleConfig` |
| `gallery.py` | appearance memory per track; EMA with a visibility gate | `AppearanceMemory`, `EmaAppearance` |
| `visibility.py` | per-detection visibility estimators: constant, and `depth_order` (bottom-edge depth order + border crop, fused with the detector score) | `VisibilityEstimator`, `ConstantVisibility`, `DepthOrderVisibility`, `build_visibility` |
| `delayed.py` | near-online K-frame look-ahead over ambiguity clusters, wrapping a cloned `ByteTrackLike`; exports `PendingDecision`s (**no-go**, see results) | `DelayedTracker` |
| `cmc.py` | camera-motion compensation from pre-computed affine files (BoT-SORT / Deep OC-SORT format) | `FileCMC`, `SparseOptFlowCMC`, `build_cmc` |
| `baselines/` | ports: `sort_kf.py` (SORT KF), `ocsort.py` (OC-SORT, Deep OC-SORT), `hybrid_sort.py`, `common.py`. BoT-SORT is `ByteTrackLike` plus the parity switches (`configs/baselines/botsort.yaml`) | `OCSortTracker`, `HybridSortTracker`, `OruKalmanFilter` |
| `bytetrack.py` | the baseline tracker: ByteTrack's two-stage association on the components above, with optional BoT-SORT-style re-ID fusion | `ByteTrackLike.update(dets) → FrameResult` |
| `config.py` | typed YAML config: strict keys and types, `inherit:`, dotted `-o key=value` overrides | `TrackerConfig`, `load_config`, `config_from_dict`, `config_to_dict` |
| `io.py` | Stage A cache schema v1 (save/load/validate), MOTChallenge `det.txt` reader, MOT results writer, per-frame JSONL export (`zed-ps2-frames/v1`: tracks + `pending`) | `SequenceCache`, `CacheMeta`, `save_cache`, `load_cache`, `load_mot_det_txt`, `write_mot_results`, `write_frames_jsonl` |
| `runner.py` | runs a tracker over one sequence and times every `update()`; single tracker construction point | `run_sequence`, `build_tracker`, `SequenceRun` |
| `cli.py` | `python -m tracker.cli`: run a config over caches or public dets, write MOT txt, JSONL, latency, an optional submission zip | `main` |

## `eval/`: evaluation (CPU)

| File | Responsibility |
|---|---|
| `splits.py` | the ByteTrack half split; crops and renumbers gt and caches; writes TrackEval gt folders and seqmaps |
| `trackeval_runner.py` | imports the vendored TrackEval with the NumPy alias shim; runs HOTA/CLEAR/Identity; `BENCHMARK` per dataset; markdown tables |
| `stress.py` | visibility and people-per-frame bins from per-gt-box CLEAR/Identity outcomes; totals asserted equal to TrackEval |
| `latency.py` | single-threaded latency protocol (p50/p95/p99/max, CPU model, by detections per frame) |
| `oracle.py` | gt-as-tracker ceiling with visibility > τ (the visibility-0 cap) |
| `run.py` | orchestrates split → track → TrackEval → stress and writes `run.json` with provenance and the leakage flag |
| `gate_events.py` | virtual entry lines (frozen YAML), crossings with hysteresis, count / event P/R / attribution error, tailgating P/R (`make gates`) |
| `visibility_eval.py` | scores visibility estimates against gt column 9 (MAE, Spearman, AUROC/AP) |
| `corrupt.py` | synthetic detection corruption (drops, jitter, false positives) from gt, for the go/no-go |
| `go_no_go.py` | pre-registered go/no-go protocol for the delayed assignment |
| `ablate.py` | `make ablate`: runs `configs/ablations/manifest.yaml`, writes md/csv/json + stress plots |
| `export_viewer.py` | exports tracker outputs, gt, pending decisions and gate events as viewer clips (`.json.gz`) |

## `models/`: Stage A (GPU; torch only here)

| File | Responsibility |
|---|---|
| `protocols.py` | `Detector`, `Embedder` and `FrameSource` protocols, so the orchestration is tested with fakes |
| `frames.py` | `ImageFolderSource` (MOT/DanceTrack `img1/`) and `VideoSource` (MP4 gate clip), with a prefetch thread |
| `yolox_detector.py` | YOLOX-X from ByteTrack's package and checkpoints. Reproduces ByteTrack's preprocessing, uses obj×cls score, clips to the image, keeps every box ≥ the floor |
| `osnet_embedder.py` | torchreid `osnet_ain_x1_0` (MSMT17), 256×128 crops, fp16, L2-normalised 512-d |
| `stage_a.py` | `python -m models.stage_a dataset|video`: resumable per-sequence cache writing, `stage_a_log.jsonl`, provenance |
| `prepare.py` | normalises third-party layouts (SOMPT22) into `<seq>/img1` + `seqinfo.ini` |
| `bringback.py` | integrity check of every cache, then a hard-linked bring-back bundle with `MANIFEST.json` and `SHA256SUMS` |
| `mot_to_coco.py` | half-split COCO annotations for the optional MOT20 first-half fine-tune |
| `exps/yolox_x_mot20_half.py` | YOLOX experiment file for that fine-tune (ByteTrack conventions; excluded from lint) |

## `scripts/`

| Path | Responsibility |
|---|---|
| `dgx/setup.sh`, `run.sh`, `finalize.sh`, `common.sh`, `config.env`, `selftest.py`, `finetune_mot20_half.sh`, `requirements-stage-a.txt` | the GPU session; see [dgx_guide.md](dgx_guide.md) |
| `research/vis_oracle_check.py` | Research Lab's gt-box visibility reproduction (basis of the phase-2 estimator) |

## `configs/`

| Path | Contents |
|---|---|
| `baselines/bytetrack.yaml` | ByteTrack code defaults (MOT17) |
| `baselines/bytetrack_mot20.yaml` | `--mot20`: no score fusion, match 0.7 |
| `baselines/bytetrack_reid.yaml` | + BoT-SORT-style IoU/re-ID fusion (needs embeddings; no camera-motion compensation) |
| `baselines/{ocsort,deep_ocsort,hybrid_sort,botsort}*.yaml` | the ported baselines (MOT17 and `_mot20`), upstream values cited inline |
| `ours/visibility.yaml`, `ours/delayed.yaml` | phase 2 components (delayed v2 frozen for the go/no-go) |
| `ablations/` | `manifest.yaml` (the `make ablate` rows) and the buffer-90 control |
| `gates/mot17_val_half.yaml` | frozen gate lines for MOT17-02/04/09 |

## `tests/`

| File | Covers |
|---|---|
| `test_kalman.py`, `test_cost.py`, `test_assignment.py`, `test_lifecycle.py`, `test_gallery_visibility.py` | component maths and contracts |
| `test_bytetrack.py` | end-to-end tracker behaviour on synthetic scenes (crossing, occlusion, low-score rescue, thresholds, output filters) |
| `test_config.py`, `test_io.py` | strict config loading; cache schema, MOT and JSONL round trips |
| `test_stage_a.py`, `test_dgx_helpers.py` | Stage A orchestration with fake models; split and SOMPT22 helpers |
| `test_eval.py` | splits, TrackEval wrapper, stress totals against TrackEval, oracle, and the MOT17-09 regression (marker `data`) |
| `test_package.py` | imports and version |

## `viewer/`

A static Vite + React + TypeScript viewer, built into one `dist/index.html` (`make viewer`), that
opens over `file://`. It shows two trackers side by side with a timeline of ID-switch and crossing
marks, dashed held decisions and gate counters. See [../viewer/README.md](../viewer/README.md).

## `serve/`

CPU job service (FastAPI). See [../serve/README.md](../serve/README.md).

| File | Purpose |
|---|---|
| `app.py` | `create_app(settings)`: routes, error handlers, CORS, web app at `/` (viewer at `/viewer`), the original `/api/track` replay contract |
| `api.py` | `Service`: clips (blurred copies only), uploads with type/size/duration checks, jobs, results, benchmarks routes |
| `jobs.py` | filesystem job store (atomic JSON) and worker pool: progress, cancel, recovery of interrupted jobs |
| `pipeline.py` | decode → YOLOX → tracker → analysis for one video; timing (detector / tracker p50 / p95, wall) |
| `analysis.py` | without ground truth: ID-preserving recoveries, heuristic suspected switches, neutral crossings, per-job metrics |
| `detector.py` | YOLOX ONNX person detector on onnxruntime CPU; sha256-checked model fetch |
| `clips.py`, `blur.py` | Pexels clip manifest + fetch; face blurring (YuNet + head regions) re-encoded with a keyframe every 10 frames |
| `benchmarks.py` | `/api/benchmarks` from `docs/results/` (headline, ablation, gate events, go/no-go sanity check, pending-GPU list) |
| `settings.py`, `errors.py` | `ZED_*` environment settings; structured `{"error": {code, message}}` responses |
| `events.py` | gate-event contract (crossing / tailgate / ambiguous_hold) |

## `web/`

React 18 + Vite + TypeScript product frontend: Overview, Demo, Results, Architecture. `src/lib/playback.ts`
holds the pure frame/time maths (shared `viewer/src/media.ts`), `src/lib/draw.ts` the canvas
overlay. Vitest in `web/tests/`. See [../web/README.md](../web/README.md).

## Other

| Path | Purpose |
|---|---|
| `third_party/trackeval/` | TrackEval `12c8791`, unmodified, MIT |
| `docs/ci/github-actions.yml` | CI (`check`, `web`, `docker`, `regression`); must be added as `.github/workflows/ci.yml` through the GitHub UI |
| `docker-compose.yml`, `Dockerfile`, `web/Dockerfile`, `web/nginx.conf.template` | backend image (API + built web app + viewer, ffmpeg) and nginx frontend proxying `/api`; models and blurred clips are mounted, never baked in |
| `scripts/demo/fetch_clips.sh` | models + Pexels clips + face blurring in one go (`make models clips blur`) |
| `configs/demo/clips.yaml` | demo clip manifest: Pexels id, consent (false), event styling (neutral), gate line |
| `docs/` | this documentation; `docs/research/` holds the Research Lab briefs |
| `demo/`, `dashboard/`, `deploy/` | superseded placeholders; their READMEs point to `viewer/`, `web/` and `docker-compose.yml` |
| `docs/integration/` | index of the remaining presentation deliverables |

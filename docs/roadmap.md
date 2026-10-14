# Roadmap

| Phase | Scope | Status |
|---|---|---|
| **1** | scaffold, tracker core, eval stack (TrackEval, stress, latency, oracle), Stage A + DGX runbook, docs | **done** (PRs #1–#5) |
| **DGX** | Panshul runs `scripts/dgx/` (MOT17, MOT20 × 3 detectors, SOMPT22, DanceTrack, gate clip, Pexels) and brings back caches | waiting on the GPU session |
| **2a** | ours: depth-order visibility (+ scoring vs gt), delayed assignment + pre-registered go/no-go, ported baselines with upstream parity, `make ablate` | **done on CPU / public dets** (PRs #6–#10). Visibility = longer buffer; delayed = **no-go**. Real-detection + re-ID re-runs and visibility-gated appearance wait for the GPU session ([next_gpu_session.md](next_gpu_session.md)) |
| **2b** | `eval/gate_events.py` (below), `make gates` | **done** (PR #11); MOT17 has too few crossings to rank trackers, needs the gate clip |
| **3** | static viewer (`viewer/`, `make viewer`) | viewer **done** (PR #12; real frames + 150 ms fade PR #18); CPU replay service `serve/` + Dockerfile (PR #19); integration proposal, video script, deck outline (PR #20); **product web app `web/` + CPU video-job service + face-blurred Pexels demo clips + docker-compose done**; live stream prototype **cut** for now ([status.md](status.md)) |
| **4** | presentation deliverables (below) | scripts and outline exist; recording waits for the GPU numbers |

## Phase 2a: the method

1. Once the caches land, re-run the ByteTrack-like baseline on YOLOX `bytetrack_ablation`
   detections for MOT17 val-half and compare with ByteTrack's README (MOTA 76.6 / IDF1 79.3)
   to validate the reproduction (ADR-003/004).
2. `tracker/visibility.py::DepthOrderVisibility`: bottom-edge depth order + border crop, from
   detections only. Score it against gt column 9 (MAE, Spearman, AUROC at 0.5/0.25, coverage per
   bin) next to the gt-box oracles.
3. Visibility-gated appearance memory and cost weighting; longer buffers for occluded losses.
4. `delayed` assignment: an ambiguity trigger (top1 − top2 margin, low visibility), bounded
   clusters and hypotheses, commit at t − K, `PendingDecision` export. K = 0 must reproduce the
   baseline bit for bit.
5. Tune on train halves; report on val halves (MOT17, MOT20 with each detector variant
   labelled), plus DanceTrack val; stress bins and latency for every row; synthetic occlusion
   injection.
6. Ablation table: each component switched off on its own (`configs/ablations/`).

## Phase 2b: `eval/gate_events.py`

Gate metrics that work directly on **saved MOT txt outputs**, so they apply to every tracker,
including the baselines, without re-running anything. Definitions follow
`docs/research/implementation_research.md` §4.2.

* **Virtual entry line:** a directed segment per sequence (or for the custom gate clip), frozen
  in a YAML file *before* any tracker is run. The trajectory point is the box's bottom-centre.
* **Crossings from gt and from tracker outputs:** a crossing `(id, t, direction)` fires when the
  point changes side of the line within its extent and stays there for ≥ h frames (hysteresis).
  Gt crossings use class-1/flag-1 gt tracks, including visibility-0 boxes.
* **Crossing-count error:** |N̂ − N| per direction, the signed bias, and MAE over windows (e.g.
  10 s) and over the whole sequence.
* **Event P/R/F1:** Hungarian matching of predicted to gt crossings in the same direction within
  a tolerance δ, plus the recall–distance curve over δ.
* **Attribution error:** the share of matched crossings whose track id does not map to the
  crossing gt id under the global IDF1 bijection (and under a local majority-overlap variant).
* **Tailgating precision / recall within a gap:** a tailgate is two distinct ids crossing in the
  same direction within Δt (e.g. ≤ 1.5 s). Pairs are matched on both member events. Failure modes
  are logged separately: an ID merge (missed tailgate), an ID split or switch at the gate (phantom
  tailgate), and a missed detection.
* Everything is binned by visibility at crossing time, so the effect of the delayed assignment
  on phantom tailgates is visible.

## Phase 3: show it

* **`viewer/`** (built), the static offline viewer: Vite/React, opens from `index.html` with no
  server. Ours and ByteTrack side by side on a timeline with ID-switch and crossing markers,
  dashed pending decisions and gate counters. See [../viewer/README.md](../viewer/README.md).
* **`web/` + `serve/`** (built), the product demo: pick a face-blurred clip or upload one, run
  detector + tracker on CPU as a background job, play back with IDs, trails, gate counters and
  recoveries; Results and Architecture pages read `docs/results/`. `make demo` or
  `docker compose up`. See [../web/README.md](../web/README.md).
* **`docs/integration/`**: how the tracker would sit under ZIG (inputs, outputs, latency budget,
  privacy: short-lived in-memory embeddings, no face recognition, no stored identities).
* `serve/`: the CPU job service exists (clips, uploads, jobs, events, metrics, benchmarks). A live
  stream prototype is cut until a GPU or edge box is available ([status.md](status.md)).

## Phase 4: deliverables

These slot into `docs/integration/` (the AI demo is the live run under Scenarios in `web/`):

* explainer video
* AI demo
* code-walkthrough video
* PPT
* detailed docs
* Zed integration write-up

## Open questions / unverified (tracked)

* GPU paths of Stage A (YOLOX-X fp16, OSNet on GPU) have not run yet.
* The SparseTrack MOT20 ablation weights: training data per README only.
* The SOMPT22 Drive layout, fps and gt columns.
* Whether the DanceTrack checkpoint is distinct from `bytetrack_x_mot17` (same byte size; setup
  warns if the hashes match).
* The CrowdHuman licence.
* Whether BoT-SORT's SBS re-ID training includes the val halves (we avoid it either way).

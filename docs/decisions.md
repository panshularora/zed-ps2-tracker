# Decisions (ADR log)

Short architecture decision records. Each one gives the status, the context, the decision and
its consequences. New decisions are appended; superseded ones stay, marked as superseded.

## ADR-001: Two stages, with the GPU work done once

* **Status:** accepted (phase 1)
* **Context:** the only GPU is a DGX session with A100s that Panshul runs; development happens on
  CPU. The research question is about association, not detection.
* **Decision:** Stage A runs the detector and re-ID once per sequence into a versioned cache
  (`tracker/io.py`, schema v1). Stage B (tracking, evaluation, ablations) runs on CPU from the
  cache.
* **Consequences:**
  * every tracker and ablation sees identical detections, so comparisons isolate tracking;
  * latency is reported for the tracker alone, plus detector fps from the Stage A log;
  * an end-to-end live demo needs the models running again (phase 3: `serve/`).

## ADR-002: Keep every detection down to 0.05, with embeddings, and filter nothing on the GPU

* **Status:** accepted
* **Decision:** Stage A keeps all boxes with score ≥ 0.05 (only zero-area boxes are dropped) and
  embeds each of them.
* **Why:**
  * ByteTrack's low-score association needs boxes down to 0.1;
  * threshold sweeps and the visibility estimator need the low-score boxes too;
  * re-running the GPU is the expensive mistake.
* **Cost:** about 770 B per detection (measured), which comes to ≈ 5–6.5 GB for everything
  (estimate).

## ADR-003: Reimplement the trackers instead of depending on boxmot/Ultralytics

* **Status:** accepted
* **Context:**
  * boxmot and Ultralytics are AGPL-3.0, which is incompatible with a closed Zed product;
  * BoT-SORT bundles a GPL-3.0 `yolov7/` folder;
  * the reference trackers are MIT, but they tangle detection, evaluation and tracking code
    together.
* **Decision:** a framework-free `tracker/` package (NumPy/SciPy), checked with `mypy --strict`.
  ByteTrack's algorithm is reimplemented on our own components. Code ideas from BoT-SORT come
  only from its tracker files.
* **Consequences:**
  * ByteTrack must be validated as a reproduction. A small deviation is accepted and documented
    (ADR-004). Once YOLOX caches arrive, numbers are checked against ByteTrack's README result
    for `bytetrack_ablation` on MOT17 val-half: MOTA 76.6 / IDF1 79.3 / IDs 159. That result
    comes from ByteTrack's own evaluator, with its per-sequence overrides, so a small gap is
    expected.

## ADR-004: xywh Kalman filter (BoT-SORT) for every tracker, including the ByteTrack baseline

* **Status:** accepted, with a known deviation
* **Decision:** one motion model, `KalmanFilterXYWH`, for every tracker. ByteTrack proper uses
  DeepSORT's `xyah` state.
* **Why:** a shared motion model isolates the effect of association changes, and BoT-SORT showed
  that xywh fits box widths better.
* **Consequence:** our "ByteTrack" baseline is ByteTrack-*like*, and is labelled that way
  everywhere (`name: bytetrack_like`). An `xyah` option can be added if a reviewer asks for an
  exact reproduction.

## ADR-005: Baseline parameters are ByteTrack's code defaults, not the paper's

* **Status:** accepted
* **Decision:**
  * `match_thresh` 0.9 (the paper says 0.8), `min_box_area` 100, w/h ≤ 1.6;
  * unconfirmed-track stage 0.7, duplicate removal at 0.15;
  * a score exactly equal to `high_thresh` falls in neither set;
  * MOT20: `fuse_score` off and match 0.7.
* **Also:** ByteTrack's per-sequence overrides (buffer and threshold hacks for MOT17-05/06/13/14
  and others) are **not** reproduced, since they break the online/no-tuning rule.
* **Frame-rate buffer:** ByteTrack's evaluator never passes the frame rate, so its buffer is 30
  frames everywhere. The baseline does the same (`scale_buffer_by_fps: false`). "Ours" may use
  time-based buffers, and will be ablated against this.

## ADR-006: Evaluation on held-out halves, with TrackEval vendored unmodified

* **Status:** accepted
* **Context:** the MOTChallenge server is offline (archive notice of 16 Apr 2026).
* **Decision:**
  * MOT17/MOT20 train half split (ByteTrack convention, identical to the Research Lab reference),
    plus DanceTrack val;
  * TrackEval `12c8791` is vendored under `third_party/` with its MIT licence; a NumPy alias shim
    replaces any patching;
  * `BENCHMARK=MOT20` for MOT20;
  * test-leaderboard numbers only in a separate table.
* **Consequences:** results are reproducible offline; CI pins a regression number on MOT17-09.

## ADR-007: Leakage-safe detectors per dataset

* **Status:** accepted; the SparseTrack weights are unverified
* **Decision:**
  * MOT17 → `bytetrack_ablation`;
  * MOT20 → SparseTrack `yolox_x_mot20_ablation` (MOT20 first half + CrowdHuman) and
    `bytetrack_x_mot17` (no MOT20 data) as clean options;
  * `bytetrack_x_mot20` kept only as a labelled `SAW_EVAL_HALF` variant;
  * an optional own first-half fine-tune.
* **Re-ID:** OSNet-AIN trained on MSMT17, not BoT-SORT's MOT-trained SBS.
* **Consequence:** the `detector_saw_eval_half` flag travels from the cache meta to `run.json`.

## ADR-008: Stress bins computed by replicating TrackEval's matching, with checked totals

* **Status:** accepted
* **Decision:** rather than approximating with our own matcher, `eval/stress.py` runs TrackEval's
  preprocessing and repeats CLEAR and Identity matching per gt box. Bin totals must equal
  TrackEval's on every run, or the run fails.
* **Why:** a stress table that doesn't add up to the headline table would not be trusted.

## ADR-009: Per-frame JSON export with a `pending` field from day one

* **Status:** accepted
* **Decision:** `write_frames_jsonl` emits `{frame, tracks[], pending[]}`. `pending` is empty
  until the delayed assignment exists.
* **Why:** the demo viewer, gate events and integration can be built against a stable schema
  (`zed-ps2-frames/v1`) before the novel component lands.

## ADR-010: YOLOX is the default detector

* **Status:** accepted
* **Decision:** YOLOX (Apache-2.0), with the public ByteTrack/SparseTrack checkpoints for
  benchmarking.
* **Why:** it is the detector every compared tracker uses (like-for-like), and it avoids AGPL.
* **Caveat:** the checkpoints are trained on non-commercial data. They are for benchmarks only;
  see [licenses.md](licenses.md).

## ADR-011: No project licence chosen yet

* **Status:** open, and Panshul's decision
* **Context:** the repo is private. The code has no copyleft dependencies (TrackEval is MIT and
  vendored with its licence).

# Evaluation protocol

Short version: every number in this repo comes from **TrackEval** (vendored, unmodified) on
**held-out halves of MOT17/MOT20 train** and **DanceTrack val**. Trackers are strictly online
with no per-sequence tuning, and a stress breakdown plus a latency measurement goes with each
result.

## Metrics

| Metric | What it rewards | Source |
|---|---|---|
| **HOTA** (+ DetA, AssA) | balanced detection × association, averaged over IoU 0.05–0.95 | TrackEval `HOTA` |
| **IDF1** | keeping the *same* ID on a person for as long as possible (global ID bijection) | TrackEval `Identity` |
| **MOTA** | detection errors dominate: 1 − (FN + FP + IDSW) / GT | TrackEval `CLEAR` |
| **IDSW** | ID switches (CLEAR definition, IoU ≥ 0.5) | TrackEval `CLEAR` |
| IDSW / 1k GT | switch rate, comparable across subsets | `eval/stress.py` |
| Gate events (phase 2b) | crossing-count error, attribution error, tailgating precision/recall | `eval/gate_events.py` (planned) |
| Latency p50 / p95 / p99 / max | per-frame tracker time on CPU | `eval/latency.py` |

For an identity-preservation problem the headline metrics are **IDF1 and HOTA**. MOTA is
reported because it is expected, but it mostly measures the detector.

## Splits

| Dataset | Tune on | Report on | Notes |
|---|---|---|---|
| MOT17 | train half (frames 1..N//2+1) | **val half** (frames N//2+2..N, renumbered from 1) | 7 `*-FRCNN` train sequences (the three detector copies share images and gt) |
| MOT20 | train half | **val half** | 4 train sequences; TrackEval `BENCHMARK=MOT20` |
| DanceTrack | train | **val** (25 sequences) | 20 fps, IDs from 0, visibility always 1 (no visibility bins) |
| SOMPT22 | train | test-format outputs only | no submission (one per week via GitHub issue; Panshul's call) |

The split is the one every compared repo uses (ByteTrack, OC-SORT, BoT-SORT, SparseTrack and
Hybrid-SORT share `tools/convert_mot17_to_coco.py`). `eval/splits.py` implements it.
A row-by-row comparison confirmed that its output
matches the Research Lab's reference split exactly: same rows, same `seqLength` (MOT17 val
lengths 299/524/418/262/326/449/374; MOT20 214/1390/1202/1657) and same seqmap. Stage A caches
cover full sequences; the split is applied on CPU.

### Why there are no MOTChallenge test-set numbers

The MOTChallenge evaluation server is **offline**: the site carries an archive notice dated
**16 April 2026**, and test submissions are no longer accepted. Test-set gt was never public. So:

* all our numbers come from the MOT17/MOT20 **train half-splits** and **DanceTrack val**;
* published MOT17/MOT20 **test** leaderboard numbers (e.g. ByteTrack 80.3 MOTA / 77.3 IDF1 on
  MOT17 test) may be cited for context, **in a separate table, never in the same table as ours**.
  They come from a different split, different detectors and per-sequence tuning;
* for like-for-like baselines we re-run ByteTrack-style and BoT-SORT-style trackers in **our**
  pipeline on the **same Stage A detections**.

## How a run works

```bash
python -m eval.run --dataset MOT17 --root data/MOT17 --split val_half \
    --dets public|<Stage A cache dir> --config configs/baselines/bytetrack.yaml \
    --name bytetrack_public --out outputs/MOT17_val_half
```

Each run:

1. writes the split gt;
2. runs `tracker.update()` frame by frame;
3. writes MOT txt and `run.json` (config, inputs, `detector_saw_eval_half`, git commit, latency);
4. calls TrackEval (HOTA, CLEAR, Identity) with `BENCHMARK=MOT17`, or `MOT20` for MOT20;
5. computes the stress bins and checks their totals against TrackEval's.

`make eval`, `make stress`, `make latency`, `make oracle` and `make smoke` wrap these.

### TrackEval settings

* Vendored commit `12c8791` (MIT) under `third_party/trackeval`, unmodified. A three-line shim
  restores `np.float/np.int/np.bool` (removed in NumPy 1.24) as aliases of the builtins.
* `MotChallenge2DBox`, `SKIP_SPLIT_FOL=True`, our own `SEQMAP_FILE`, preprocessing on (default).
  Only class 1 with `consider = 1` is scored. Tracker boxes matched (IoU ≥ 0.5) to a distractor
  are removed before scoring. The distractor classes are person-on-vehicle (2), static person (7),
  distractor (8) and reflection (12).
* **`BENCHMARK=MOT20` for MOT20.** It adds class 6 (`non_mot_vehicle`) to the distractors.
  Scoring MOT20 with `BENCHMARK=MOT17` would count boxes on those as false positives.
* **TrackEval does not filter by visibility.** Fully occluded people (visibility 0) are gt
  boxes a tracker is expected to output. See the cap below.

### The visibility-0 cap

About 11% of evaluated MOT17 val-half gt boxes (5,891 of 53,890) have visibility exactly 0. A
tracker that never outputs a box for an invisible person cannot score them. `eval/oracle.py`
measures the ceiling by submitting the **gt itself** as a tracker, keeping only boxes with
visibility > τ. That is perfect detection and perfect identity, minus the hidden boxes:

| Oracle (gt boxes with vis > τ) | HOTA | IDF1 | MOTA | FN |
|---|---|---|---|---|
| MOT17 val-half, τ = 0 | **91.1** | **94.2** | **89.1** | 5,891 |
| MOT17 val-half, τ = 0.25 | 79.9 | 85.3 | 74.3 | 13,841 |
| MOT20 val-half, τ = 0 | 97.3 | 98.4 | 96.9 | 19,345 |

These reproduce the Research Lab's brief (`docs/research/implementation_research.md` §2.1). The
practical consequences:

* a detector-driven tracker that does not **coast** through occlusion is capped at about
  IDF1 94 / HOTA 91 on MOT17 val-half;
* coasting (keeping a predicted box alive while the person is hidden) is directly rewarded by
  MOTA and IDF1. A tracker can therefore gain points by guessing; the stress bins expose this,
  because the gain shows up in the 0–0.25 bin.

## Stress method

`eval/stress.py` answers "where does it fail?". It reuses TrackEval's own loading and
preprocessing, adding the gt visibility column through a small subclass. It then repeats
TrackEval's two matchings per gt box:

* **CLEAR**: per-frame Hungarian on IoU ≥ 0.5 with the previous-frame continuation bonus. Each gt
  box is TP or FN, and a matched box may be an IDSW.
* **Identity**: the global gt-id ↔ tracker-id assignment. A gt box is an IDTP when its assigned
  tracker id covers it with IoU ≥ 0.5.

On every run the bin totals are **asserted equal** to TrackEval's TP, FN, FP, IDSW, IDTP, IDFP
and IDFN. If they ever diverge, the run fails.

| Breakdown | Bins | Reported per bin | Why not MOTA/IDF1 |
|---|---|---|---|
| gt visibility (col 9) | 0–.25, .25–.5, .5–.75, .75–1, plus "= 0" as a subset | GT count and share, recall, ID recall, IDSW, IDSW per 1k GT | an FP has no gt box, so it has no visibility |
| people per frame | quartiles of the eval set's per-frame gt counts (edges stored in the output) | frames, GT, **MOTA, IDF1** (exact: whole frames are binned), recall, FP, IDSW, IDSW per 1k GT | – |

Reading the visibility bins: recall in the low bins is tiny even when a tracker box is in the
right place. When two people overlap, one tracker box can match only one of them (one-to-one
matching), and it goes to the more visible one. Compare trackers within a bin, not across bins.

Planned additions (phase 2): synthetic occlusion injection (drop a track's detections for
k = 5/15/30 frames and measure re-association), and per-window regressions against density and
mean visibility.

## Latency method

`eval/latency.py`:

* `OMP_NUM_THREADS=1`, and the MKL/OpenBLAS/NumExpr equivalents, are set **before** NumPy loads;
* one untimed warm-up pass, then `--repeats` (default 3) timed passes over every sequence;
* the timer brackets `tracker.update(dets)` only: detector, re-ID and file I/O are excluded (they
  run in Stage A, and the detector's latency is reported separately from the Stage A log);
* we report p50 / p95 / p99 / max per sequence and combined, a breakdown by detections per frame,
  the count of frames over 10 ms, the **CPU model** and the Python/NumPy versions.

The box is a shared VM, so isolated maxima (a few frames out of ~16k) are scheduler or GC spikes.
p95/p99 are the stable figures, and the max is reported as measured.

## Online-only rule

* The output for frame t depends only on frames 1..t. The planned delayed variant uses 1..t+K,
  is labelled "K-frame-delayed online", and is always shown next to K = 0.
* No interpolation, no offline smoothing, no tracklet merging after the fact, no use of future
  frames for initialisation.
* No per-sequence parameters (ByteTrack's evaluator overrides are not reproduced). Parameters
  are tuned on train halves only and frozen before scoring the val halves.
* No test-set peeking: there is no test server, and we do not use any gt of the eval split
  except to score.

## Leakage policy

| Risk | Rule |
|---|---|
| Detector trained on the eval frames | MOT17: `bytetrack_ablation` (CrowdHuman + MOT17 train **first half**). MOT20: SparseTrack's `yolox_x_mot20_ablation` (CrowdHuman + MOT20 train first half, unverified) or `bytetrack_x_mot17` (no MOT20 data at all). `bytetrack_x_mot20` saw the eval half: its cache sits in a folder named `SAW_EVAL_HALF`, `meta.detector_saw_eval_half = true`, and `run.json` propagates the flag. **Never in a headline table.** |
| Re-ID trained on the eval frames | OSNet-AIN trained on MSMT17 only. BoT-SORT's MOT-trained SBS weights may include the val halves (unverified), so they are not used. |
| Tuning on the eval split | thresholds are chosen on train halves (Stage A caches cover full sequences, so the same cache serves both halves) |
| Visibility estimate fed gt | the estimator sees detections only. Feeding gt boxes reproduces the gt column by construction (circular). |
| SOMPT22 | only test-format outputs are produced; nothing is submitted |
| Public detections | public FRCNN dets give a pipeline check only; they are labelled as such and never compared with YOLOX-based results in one table |

## Current results (phase 1, pipeline check)

See the README. Phase 1 numbers use **public FRCNN detections with an IoU-only ByteTrack-style
tracker**. They validate the pipeline (splits, TrackEval, stress, latency) and are **not**
comparable to the published ByteTrack numbers, which use YOLOX detections.

MOT20's public detections come with confidence values that are almost all 0 or 1 (median 0 on
MOT20-03/05), so a score-thresholded tracker discards most of them. MOT20 results therefore wait
for the Stage A caches.

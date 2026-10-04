# Pre-registration: go/no-go v2 for the delayed decision (with re-ID, tighter trigger)

**Frozen on 4 Oct 2026, before any YOLOX detection or re-ID embedding existed on the CPU box.**
The commit that adds this file is the timestamp. The config, the trigger, the thresholds and the
decision rule below must not change after the caches arrive; any later change is a new,
separately dated pre-registration, and this one is still reported. This is a **sanity check, not
a results table**: whatever the verdict, its rows stay out of the headline tables (as in v1,
[`docs/results/go_no_go.md`](../results/go_no_go.md)).

## Why a v2

v1/v2-IoU was a clear no-go: ID switches rose on all five MOT17 inputs, and the trigger fired on
24–69 % of frames. Our unverified hypothesis was that IoU-only look-ahead costs cannot tell the
hypotheses apart once a Kalman track has adapted. Re-ID gives the look-ahead a signal that does
not adapt that way. So v2 changes exactly two things: **appearance in the cost** and a **much
tighter trigger**.

## What is run (all fixed)

* **Tracker:** [`configs/ours/delayed_v2_reid.yaml`](../../configs/ours/delayed_v2_reid.yaml):
  ByteTrackLike + OSNet-AIN re-ID (BoT-SORT-style fused cost, `bytetrack_reid.yaml` values),
  depth-order visibility estimate used **only** for the trigger.
* **Reference (K = 0):** the same file with `delayed.enabled=false`. `tests/test_delayed.py`
  checks that K = 0 is bit-identical to it.
* **Arms:** K = 5 (primary, 167–357 ms delay at 14–30 fps) and K = 3 (secondary, reported only).
* **Trigger** (a look-ahead happens only for a cluster that meets all of these):
  1. near-tie on the **fused** IoU/appearance first-association cost: margin **0.03**
     (v1: 0.05), so appearance must also fail to separate the candidates;
  2. only **confirmed** tracks, clusters of at most **3** tracks and 3 detections;
  3. at least one cluster detection has estimated visibility **v̂ < 0.5**
     (`trigger_max_visibility`);
  4. an alternative hypothesis replaces the immediate choice only if its look-ahead score is
     better by **> 0.05** (`min_gain`).
* **Look-ahead score:** unchanged from v2-IoU: sum of fused match costs over all
  first-association tracks, miss penalty 1.0 per unmatched track-frame, birth penalty 1.0 per
  track born during the look-ahead, swaps only (`same_tracks`).
* **Data:** MOT17 val-half with the leak-free YOLOX-X `bytetrack_ablation` cache (primary); MOT20
  val-half with the first clean variant ingested, `sparsetrack_ablation` else
  `mot17full_xdomain` (replication), with the same config file unchanged (MOT17 thresholds) for
  both arms. Never a `SAW_EVAL_HALF` cache.
* **Runner:** `make phase3` (`eval/phase3.py`, block `go_no_go_v2` of
  `configs/phase3/plan.yaml`); it computes the verdict mechanically from this rule.

## Decision rule

**GO** only if **all five** hold for the **K = 5** arm; otherwise **NO-GO**.

| # | Criterion | Threshold |
|---|---|---|
| 1 | ID switches, MOT17 val-half (TrackEval) | IDSW(K=5) ≤ **0.90 ×** IDSW(K=0) |
| 2 | Identity quality, MOT17 val-half | IDF1(K=5) ≥ IDF1(K=0) **+ 0.5** and HOTA(K=5) ≥ HOTA(K=0) **− 0.2** |
| 3 | Selectivity, MOT17 val-half | the trigger fires on **≤ 10 %** of frames |
| 4 | Compute, MOT17 val-half | p95 `update()` ≤ **20 ms** single-threaded (`eval.latency`); the K/fps delay comes on top and is reported |
| 5 | Replication, MOT20 val-half | IDSW(K=5) ≤ IDSW(K=0), and the K=5 run finishes within **60 min** (a timeout counts as a fail) |

Notes fixed in advance:

* No threshold above is tuned on val. Nothing in the delayed block is tuned at all in v2.
* If no clean MOT20 cache is available, criterion 5 is "not evaluated" and the verdict can be at
  most **GO (MOT17 only)**, stated that way.
* Gate events are not a criterion: MOT17 has only 25 crossings. They are reported if a gate clip
  exists.
* If the verdict is NO-GO, the demo's "dashed box while holding" shot is dropped and the pitch
  uses the method that does help on GPU data (see `docs/demo_video_script.md`).

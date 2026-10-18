# Laptop GPU run (16–18 Oct 2026): what the numbers say

Plain-English read of the three result bundles from `bash run.sh all --profile laptop`
(RTX 3050 6 GB Laptop GPU, WSL Ubuntu 24.04, YOLOX-X fp16, OSNet-AIN re-ID, TrackEval).
Every number below is copied from a file in this repo; the file is named under each table.
No GPU experiment was re-run for this write-up.

## 1. Where the files are

| What | Path |
|---|---|
| Phase 3 report (ablations, reproduction, go/no-go, gate events, PNGs), installed by `make ingest BRING_BACK=zed_results_20261018_053613_laptop_dancetrack.tar.gz` | `docs/results/phase3/` |
| Manifests + SHA256SUMS of all three bundles, sha256 of the tarballs | `docs/results/gpu/bundles/` |
| Run status, env, GPU, verdict | `docs/results/gpu/run/` |
| All run logs | `docs/results/gpu/logs/` |
| Stage A (detector + re-ID) per-sequence logs | `docs/results/gpu/stagea_logs/` |

**Integrity.** `sha256sum -c SHA256SUMS` passes for every file of all three bundles; each
extracted folder is byte-identical to its tarball; `eval.ingest` re-verified the 121 files of
the DanceTrack bundle before installing `docs/results/phase3/`.

**Which bundle.** The DanceTrack bundle (`zed_results_20261018_053613_laptop_dancetrack`) is the
latest and a superset: it has every file of the MOT17 and MOT20 bundles plus DanceTrack, SOMPT22
and later logs. The MOT17 and MOT20 ablation tables inside it are the same numbers as in the
earlier bundles (only the run-info header line differs). The MOT17 bundle's go/no-go had only
4 criteria evaluated (MOT20 "not evaluated"); the later bundles evaluate all 5.
Large raw tracker outputs (`results/SOMPT22/*.txt`, the 3.9 MB SOMPT22 submission zip) were
not copied.

## 2. Results per dataset

HOTA, IDF1, MOTA, AssA in %, IDSW is a count. **Bold** = best in that column for that dataset.
"Ours" rows are the visibility rules (`configs/ours/`); "ours (BoT-SORT)" is
`BoT-SORT + re-ID + visibility-gated appearance` (`configs/ours/visibility_botsort.yaml`),
"ours (ByteTrack)" is `+ re-ID + visibility-gated appearance (full)`.

### MOT17 val-half (detector `bytetrack_ablation`, leak-free) — `docs/results/phase3/mot17_val_half/ablation.md`

| Group | Tracker | HOTA | IDF1 | MOTA | IDSW | AssA |
|---|---|---|---|---|---|---|
| baseline | ByteTrackLike (IoU) | 63.4 | 73.8 | 66.4 | 165 | 66.5 |
| baseline | OC-SORT | 61.8 | 72.3 | 64.0 | 204 | 65.7 |
| baseline | Deep OC-SORT (emb + CMC) | 63.3 | 74.6 | 64.6 | **125** | 68.2 |
| baseline | Hybrid-SORT | 62.4 | 72.2 | 64.2 | 240 | 65.4 |
| baseline | BoT-SORT (CMC, IoU) | **64.6** | **76.3** | **67.1** | 135 | **68.6** |
| baseline | ByteTrackLike + re-ID | 62.5 | 73.0 | 66.5 | 183 | 64.9 |
| baseline | BoT-SORT (CMC) + re-ID | 63.4 | 74.3 | 66.9 | 193 | 66.3 |
| control | ByteTrackLike + re-ID, buffer 90 | 62.3 | 73.1 | 66.6 | 172 | 64.8 |
| ours | + re-ID + visibility (buffer) | 62.3 | 73.1 | 66.6 | 172 | 64.8 |
| ours | + re-ID + appearance update gate | 62.0 | 72.6 | 66.2 | 201 | 64.0 |
| ours | + re-ID + appearance cost gate | 63.2 | 73.3 | 66.3 | 174 | 66.2 |
| ours | ours (ByteTrack): visibility-gated appearance (full) | 63.2 | 73.9 | 66.2 | 167 | 66.2 |
| control | BoT-SORT + re-ID, buffer 90 | 63.3 | 74.4 | 66.9 | 183 | 66.1 |
| ours | ours (BoT-SORT): visibility-gated appearance | 63.9 | 75.1 | 66.8 | 156 | 67.1 |

**Best: BoT-SORT (CMC, IoU only)**, HOTA 64.6 / IDF1 76.3 / IDSW 135. Ours (BoT-SORT) is second
on HOTA and IDF1 (63.9 / 75.1). Why: on MOT17 adding OSNet re-ID *hurts* BoT-SORT
(64.6 → 63.4 HOTA, IDSW 135 → 193). Our visibility gate wins back most of that loss
(63.4 → 63.9 HOTA, 193 → 156 IDSW against BoT-SORT + re-ID, and 183 → 156 against its own
buffer-90 control), but not all of it.

### MOT20 val-half, primary clean cache (detector `sparsetrack_ablation`) — `docs/results/phase3/mot20_val_half__sparsetrack_ablation/ablation.md`

| Group | Tracker | HOTA | IDF1 | MOTA | IDSW | AssA |
|---|---|---|---|---|---|---|
| baseline | ByteTrackLike (IoU) | 69.4 | 84.1 | 87.9 | **1011** | 66.8 |
| baseline | OC-SORT | 68.0 | 82.7 | 86.7 | 1293 | 64.9 |
| baseline | Deep OC-SORT (emb + CMC) | 68.7 | 83.8 | 86.8 | 1171 | 66.1 |
| baseline | Hybrid-SORT | 68.8 | 83.4 | 87.2 | 1365 | 66.0 |
| baseline | BoT-SORT (CMC, IoU) | 69.3 | 83.9 | 87.9 | 1162 | 66.5 |
| baseline | ByteTrackLike + re-ID | 67.5 | 81.1 | 87.7 | 1278 | 63.5 |
| baseline | BoT-SORT (CMC) + re-ID | 67.6 | 81.2 | 87.8 | 1429 | 63.6 |
| control | ByteTrackLike + re-ID, buffer 90 | 67.4 | 81.2 | 87.7 | 1272 | 63.2 |
| ours | ours (ByteTrack): visibility-gated appearance (full) | **69.5** | **84.5** | 87.9 | 1019 | **66.9** |
| ours | ours (BoT-SORT): visibility-gated appearance | 69.2 | 84.0 | 87.9 | 1162 | 66.3 |

(MOTA: BoT-SORT IoU is highest at 87.94 in `ablation.csv`; all rows round to 86.7–87.9.)

**Best: ours (ByteTrack base)**, HOTA 69.5 / IDF1 84.5 / AssA 66.9, but only +0.1 HOTA and
+0.4 IDF1 over plain ByteTrackLike, with 8 more ID switches (1019 vs 1011). Treat as a tie.
The clear effect is against re-ID without the gate: ByteTrackLike + re-ID drops to 67.5 HOTA /
1278 IDSW; our gate brings it back to 69.5 / 1019. In dense crowds raw re-ID hurts, and the
visibility gate is what stops it hurting.

### MOT20 val-half, cross-domain cache (detector `bytetrack_x_mot17`, never saw MOT20) — `docs/results/phase3/mot20_val_half__mot17full_xdomain/ablation.md`

| Tracker | HOTA | IDF1 | MOTA | IDSW | AssA |
|---|---|---|---|---|---|
| ByteTrackLike (IoU) | 60.8 | 76.8 | 79.6 | 1070 | 58.6 |
| Hybrid-SORT | **62.9** | **79.9** | **82.6** | 1471 | **60.6** |
| BoT-SORT (CMC, IoU) | 60.9 | 77.1 | 79.8 | 1248 | 58.8 |
| ours (ByteTrack) | 60.9 | 77.5 | 79.8 | **1039** | 58.8 |
| ours (BoT-SORT) | 61.1 | 77.7 | 80.0 | 1193 | 58.9 |

(Other rows in the file.) **Best: Hybrid-SORT** on HOTA/IDF1/MOTA; **ours (ByteTrack)** has the
fewest ID switches (1039).

### DanceTrack val (detector `dancetrack_bytetrack_x`, MOT17 tracker settings) — `docs/results/phase3/dancetrack_val/ablation.md`

| Group | Tracker | HOTA | IDF1 | MOTA | IDSW | AssA |
|---|---|---|---|---|---|---|
| baseline | ByteTrackLike (IoU) | 52.5 | 54.9 | 88.1 | 1775 | 36.3 |
| baseline | OC-SORT | 51.2 | 51.3 | 85.1 | 1871 | 34.8 |
| baseline | Deep OC-SORT (emb) | 55.0 | 55.6 | 85.2 | 1738 | 40.2 |
| baseline | Hybrid-SORT | **57.4** | **58.5** | 87.3 | **1667** | **42.9** |
| baseline | BoT-SORT (IoU) | 52.3 | 53.2 | 88.1 | 1934 | 35.9 |
| baseline | ByteTrackLike + re-ID | 53.3 | 54.0 | 87.8 | 1883 | 37.6 |
| baseline | BoT-SORT + re-ID | 53.4 | 53.8 | 87.7 | 2127 | 37.5 |
| control | ByteTrackLike + re-ID, buffer 90 | 54.3 | 55.0 | 87.8 | 1887 | 38.9 |
| ours | ours (ByteTrack): visibility-gated appearance (full) | 53.4 | 55.3 | **88.2** | 1681 | 37.7 |
| ours | ours (BoT-SORT): visibility-gated appearance | 54.3 | 55.8 | 88.1 | 1897 | 38.8 |

**Best: Hybrid-SORT**, HOTA 57.4 / IDF1 58.5 / AssA 42.9 / IDSW 1667, a clear 3.1 HOTA ahead
of ours (BoT-SORT, 54.3). DanceTrack has look-alike dancers and fast, non-linear motion; Hybrid-SORT's
extra motion/height cues help there and OSNet person re-ID helps little. Ours does beat
plain ByteTrack and plain BoT-SORT (+1.8 / +2.0 HOTA), but ties its buffer-90 control on HOTA (54.3 vs 54.3).

### Excluded: `mot20_val_half__mot20full_SAW_EVAL_HALF`

Its detector was trained on the evaluation half (the folder name and the table title say so). It
is a leftover from an earlier run, is not in `summary.json` "ran", and is in no table here.

### Overall

| Tracker | mean HOTA (MOT17, MOT20 sparsetrack, DanceTrack) | mean IDF1 |
|---|---|---|
| Hybrid-SORT | **62.88** | 71.38 |
| ours (BoT-SORT) | 62.50 | **71.62** |
| Deep OC-SORT | 62.34 | 71.35 |
| BoT-SORT (IoU) | 62.09 | 71.14 |
| ours (ByteTrack) | 62.03 | 71.25 |
| ByteTrackLike (IoU) | 61.75 | 70.93 |

(Means computed for this write-up from the three `ablation.csv` files; not in the bundle.)

There is **no single best tracker**: BoT-SORT (IoU) wins MOT17, ours (ByteTrack base) narrowly
wins clean MOT20, Hybrid-SORT wins DanceTrack and cross-domain MOT20. Averaged over the three
clean primary sets Hybrid-SORT has the best HOTA and ours (BoT-SORT) the best IDF1, both by
less than 0.4 points. The honest one-line claim: *our visibility gate reliably removes the damage
that re-ID does in crowds (every dataset), but it does not clearly beat the best IoU/motion
tracker.*

## 3. Why the reports say NOT REPRODUCED and NO-GO

These are **two different checks**. Neither is "ours lost to ByteTrack".

### 3a. ByteTrack reproduction: NOT REPRODUCED (`docs/results/phase3/reproduction.md`)

Our ByteTrackLike on MOT17 val-half with the ByteTrack ablation detector, against the ByteTrack
paper (arXiv 2110.06864v3, Table 1/11). Tolerance fixed in `configs/phase3/plan.yaml`.

| Metric | Paper | Our run | Δ | Tolerance | Pass |
|---|---|---|---|---|---|
| MOTA | 76.6 | 66.4 | −10.2 | ±1.0 | no |
| IDF1 | 79.3 | 73.8 | −5.5 | ±1.5 | no |
| IDSW | 159 | 165 | +6 | reported only | – |
| FP | 3358 | 5379 | +2021 | reported only | – |
| FN | 9081 | 12552 | +3471 | reported only | – |

**What this means:** this is a **reproduction gap on the detection side**, not the tracker losing.
ID switches are almost the paper's (165 vs 159), so association behaves like ByteTrack's. The
gap is +2021 false positives and +3471 missed boxes, i.e. the boxes going into the tracker are
not the boxes the paper's tracker saw. `reproduction.md` lists the known differences (xywh
Kalman state instead of xyah, TrackEval CLEAR instead of py-motmetrics, the output filter, no
per-sequence thresholds); on top of that this run is fp16 on a 6 GB laptop GPU, batch 1. Which of
these causes the 10-point MOTA gap is **not verified** by any file here. Consequence: every GPU
score in this repo is **provisional**; relative comparisons between trackers on the same cache
stay fair (they all use the same detections).

### 3b. Go/no-go v2: NO-GO (`docs/results/phase3/go_no_go_v2.json`, `.md`; rule in `docs/preregistration/go_no_go_v2.md`)

This pre-registered check is about the **delayed decision** (K-frame look-ahead with re-ID,
`configs/ours/delayed_v2_reid.yaml`), *not* about the visibility rules and *not* about ByteTrack.
GO needed all five criteria for the K = 5 arm. All five failed:

| # | Criterion (frozen 4 Oct 2026) | Needed | Got (K=0 → K=5) | Pass |
|---|---|---|---|---|
| 1 | ID switches, MOT17 | IDSW(K5) ≤ 0.90 × IDSW(K0) = 164.7 | 183 → 198 (+15, +8 %) | no |
| 2 | Identity, MOT17 | IDF1 ≥ K0 + 0.5 = 73.47 and HOTA ≥ K0 − 0.2 = 62.26 | IDF1 72.97 → 71.93 (−1.03); HOTA 62.46 → 61.95 (−0.52) | no |
| 3 | Selectivity, MOT17 | trigger on ≤ 10 % of frames | 50.7 % | no |
| 4 | Compute, MOT17 | p95 `update()` ≤ 20 ms | 170.09 ms (K0: 5.36 ms) | no |
| 5 | Replication, MOT20 (`sparsetrack_ablation`) | IDSW(K5) ≤ IDSW(K0), finish ≤ 60 min | 1462 → 1575 | no |

(Thresholds in column 3 are the prereg rule applied to the K0 row of `go_no_go_v2.json`.)
K = 3 was no better (IDSW 196, IDF1 72.2). So the look-ahead adds ID switches, lowers IDF1,
fires on half the frames instead of rarely, and is 30× slower at p95. This is the method
genuinely not working, not a reproduction artifact: both arms use the same detections, the same
re-ID and the same code (K = 0 is bit-identical to the reference per `tests/test_delayed.py`).
Per the prereg, the delayed decision stays out of every headline table and the demo's
"dashed box while holding" shot is dropped.

## 4. Failures in the logs (`docs/results/gpu/logs/`)

| Log | What happened | Affects final numbers? |
|---|---|---|
| `run_all/run_all_20261009_161357.log` | preflight failed: MOT17.zip not present and motchallenge.net unreachable | No (fixed by download; later runs) |
| `run_dancetrack_20261010_213547.log`, `..._213555.log`, `dancetrack.log` | DanceTrack val zip download size mismatch (got 4.78 GB / 4.66 GB, expected 4.21 GB), "Re-run to resume" | No: `run_dancetrack_20261018_025341.log` shows the zip at the expected 4209785614 bytes |
| `run_dancetrack_20261010_213730.log` | `FileNotFoundError` for `dancetrack0004/img1/000001.jpg` (bad extraction from the broken zip) | No: the 18 Oct run processed all 25 val sequences |
| `run_mot20_20261016_204923.log`, `run_all/run_all_20261016_203407.log` | Stage A MOT20 crashed on MOT20-03: `CUDA error: out of memory` inside `osnet_embedder.py` (re-ID batch 256 on a 6 GB GPU) | No: MOT20-01/02 were kept, MOT20-03/05 were redone in the 17 Oct run |
| `run_mot20_20261017_113307.log`, `run_all/run_all_20261017_113303.log` | Same OOM again on MOT20-03 | No: fixed by the local batch-size change, see §5 |
| `run_sompt22_20261018_040257.log` | Completed (9 train + 5 test sequences, submission zip written). SOMPT22 test has no public ground truth (`ingest`: "gt_root MISSING"), and SOMPT22 is not in `configs/phase3/plan.yaml` | No SOMPT22 metric exists anywhere; nothing to report as a score |
| `run/STATUS` | says `finalize_dancetrack [running]` | Snapshot taken *during* finalize, which then wrote the bundle; the bundle itself verifies |

Also seen: earlier MOT17 Stage A (10 Oct, batch 8) and the 16 Oct rerun (batch 1) give a few
detections different per sequence (e.g. MOT17-02: 30739 vs 30742). Only the 16 Oct cache is used.

## 5. The local OSNet change (`docs/results/gpu/osnet_local.diff.txt`, NOT committed)

Panshul's laptop clone (old main `7559df8`) had `models/osnet_embedder.py` `batch_size 256 → 16`
to stop the OOM. `run/env.txt` shows ` M models/osnet_embedder.py` from the 17 Oct 21:13 UTC run
on; the 16 Oct MOT17 run shows a clean tree.

* Batch size only changes how many person crops go through OSNet at once. Each crop's embedding
  is computed independently, so results should be the same; it changes memory and speed.
* It is **not bit-verified**: fp16 cuDNN can pick different kernels for different batch shapes,
  which can move embeddings in the last bits. The MOT20 `sparsetrack_ablation` cache mixes both
  (MOT20-01/02 at batch 256 on 16 Oct, MOT20-03/05 at 16 on 17 Oct). We expect no visible effect
  on the tables; we have not measured it.
* MOT17 (16 Oct) was computed with the committed value 256; MOT20, DanceTrack and SOMPT22 with 16.

## 6. What to say in the pitch

Never "beats ByteTrack" or "best tracker" overall: BoT-SORT without re-ID beats ours on MOT17
(HOTA 64.6 vs 63.9) and Hybrid-SORT wins DanceTrack val (57.4) and MOT20 with the MOT17-trained
detector (62.9).

1. **Lead finding: the visibility rule stops re-ID from hurting.** MOT20 val-half, SparseTrack
   detector: ByteTrack 69.4 HOTA → ByteTrack + re-ID 67.5 → ours (re-ID + visibility-gated
   appearance) 69.5, the best HOTA in that setting (IDSW 1011 → 1278 → 1019). MOT17 val-half:
   ByteTrack + re-ID 62.5 → ours 63.2.
2. **Second beat: the Stress Map** (synthetic gate): ours keeps the ID for hides of about 1–2.5 s
   at every crowd size; past 3 s its memory runs out.
3. **Where we lose.** DanceTrack: Hybrid-SORT 57.4 vs ours 54.3 HOTA. MOT20 frame latency: ours
   p95 87–95 ms (the two ours rows) vs 13.7 ms for ByteTrack (IoU) on the SparseTrack cache. The ByteTrack baseline is
   NOT REPRODUCED (MOTA 66.4 vs 76.6, IDF1 73.8 vs 79.3), so every GPU score is provisional.
4. **NO-GO as discipline.** The delay-K idea was pre-registered and failed all five checks: IDSW
   183 → 198 instead of −10 %, IDF1 73.0 → 71.9, fired on 51 % of frames vs a 10 % limit, p95
   170 ms vs 20 ms, MOT20 IDSW 1462 → 1575. So it was dropped.
5. The tracker runs end to end on a laptop GPU (RTX 3050 6 GB) on MOT17, MOT20 and DanceTrack.

# Implementation notes (phase 2, CPU-only, public detections)

What was built, where it differs from upstream, what went wrong along the way, and which ideas
did not work. Every number here comes from MOT17 val-half with **public FRCNN detections** and
no re-ID, unless stated otherwise. These results are **not** comparable to published YOLOX-based
numbers.

## Deviations from upstream (deliberate)

| Deviation | Where | Why |
|---|---|---|
| **Parity switches** on `ByteTrackLike`: `association.iou_plus_one`, `lifecycle.predict_tentative`, `output.report_tentative`, `lifecycle.removal_lag` | `tracker/bytetrack.py`, `configs/baselines/botsort.yaml` | Needed for BoT-SORT to match upstream on 0 / 2652 frames. All are off by default. See `docs/results/parity/`. |
| **xywh Kalman** (BoT-SORT noise model) for the whole ByteTrack family, instead of ByteTrack's xyah | `tracker/kalman.py` | One KF for every tracker. ByteTrackLike is therefore "ByteTrack-like", not ByteTrack. |
| **No `--post`** (offline linear interpolation) for any tracker | all configs | The product is strictly online. Upstream README numbers (e.g. Deep OC-SORT's) include it. |
| **Deep OC-SORT with embeddings off** (`--emb_off --aw_off`) | `configs/baselines/deep_ocsort.yaml` | Public detections have no embeddings. What remains is OC-SORT minus BYTE plus CMC. |
| **CMC from pre-computed files** (BoT-SORT / Deep OC-SORT affines) | `tracker/cmc.py`, `third_party/cmc_files/` | No images on the CPU box. For a new video there is no CMC file: `SparseOptFlowCMC` (image-based port) exists but is **untested and not exercised** anywhere (no OpenCV or frames here), so verify it against BoT-SORT's own output before relying on it. |
| **Common output filter**: area > 100 px², w/h ≤ 1.6, for every method | `output:` in each config | Upstream values differ (BoT-SORT/Deep OC-SORT use area 10). One filter keeps the comparison like-for-like. |
| **DPM scores logistic-mapped** (`--score-map logistic`) | `eval/run.py` | DPM's raw scores are unbounded SVM margins, so the 0.6/0.1 thresholds would be meaningless on them. Used only in the go/no-go robustness check. |
| **Tie-breaking**: scipy `linear_sum_assignment` instead of `lapjv` | `tracker/assignment.py` | Same optimum, but exact ties can resolve differently. Upstream was patched the same way for the parity check. |

## Pitfalls hit (and fixed)

* **Occluded buffer ignored on the `removal_lag` path.** With `removal_lag` on (BoT-SORT parity),
  the per-track occluded buffer was not honoured, so `BoT-SORT + visibility` matched plain
  BoT-SORT. Fixed in #10, with a regression test.
* **Visibility-0 boxes are scored.** 11 % of MOT17 val-half gt is fully hidden. Even a perfect
  visible-only tracker is capped at HOTA 91.1, so coasting through occlusion is worth points.
* **MOT20 public detections** have confidences that are almost all 0 or 1. Score-thresholded
  trackers keep almost nothing, so MOT20 numbers wait for YOLOX caches.
* **Delayed decision, v1 scoring** (score only the ambiguous cluster) was badly broken: at K=3 on
  corrupted gt, IDSW went 273 → 801. A hypothesis that drops a track looks cheap locally. v2
  scores every track and adds a birth penalty, which is less bad but still a no-go.
* **Delayed decision cost on MOT20.** K > 0 had not finished after 40 min. The ambiguity windows
  are not selective (24–69 % of frames flagged on MOT17), so the hypothesis count explodes in dense
  scenes.
* **Gate events need volume.** MOT17-02/04/09 yield only 25 gt crossings and 9 tailgates, too
  few to rank trackers. The custom gate clip from the GPU session is needed.
* **Latency numbers are single-threaded** (`eval.latency`, one process). The parallel `make
  ablate` run is not used for timing.
* **Repo hygiene.** The `.gitignore` rule `data` also ignores `viewer/data/`, so the viewer demo
  clips are added with `git add -f`. The generated `viewer/src/generated/clips.ts` is not
  committed; `npm run build` regenerates it.

## Negative results (kept, labelled, not hidden)

1. **Visibility = a longer buffer.** `DepthOrderVisibility` is a decent estimate (MAE 0.074,
   Spearman 0.613, AUROC(v < 0.5) 0.938; the detector score alone gives 0.107 / 0.429 / 0.947).
   As a tracking signal on IoU-only association, though, it does exactly what a 90-frame buffer
   does: ByteTrackLike 50.1 → 51.0 HOTA, and the buffer90 control also gives 51.0 (BoT-SORT
   52.5 vs 52.4). Its intended use, gating *appearance* updates and costs, is untested because
   there are no embeddings yet.
2. **Delayed assignment is NO-GO** (`docs/results/go_no_go.md`). With K ∈ {3, 5, 8}, IDSW goes
   *up* on every condition (e.g. FRCNN 191 → 212/215/217; corrupted gt seed 0 273 → 375/395/389),
   HOTA is lower or within ±0.2, and p95 compute goes from 2–4 ms to 13–157 ms, on top of the
   K/fps output delay. It is kept out of the headline tables. It may still be worth retrying with
   re-ID, which would let hypotheses be told apart (see `next_gpu_session.md`).

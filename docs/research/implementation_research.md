# 05 — Implementation Research Brief: identity-preserving online MOT under dense occlusion (Zed Digital PS-2)

**Audience:** the engineering agent who will implement and benchmark a CPU multi-object tracker from cached detections and embeddings.
**Our method (context):** an online, causal tracker built on a BYTE/OC-SORT-style backbone. It uses a **bounded-latency, MHT-style delayed association (fixed lag K)** that is triggered only for ambiguous look-alike pairs. It also computes a **box-only visibility estimate** (overlap + depth order from the box bottom edge), which is scored against MOT `gt.txt` visibility.
**Compiled:** 2026-10-04 (IST). The MOTChallenge evaluation server is offline. The homepage says so, and `motchallenge.net/data/MOT17/` and `/MOT20/` are static pages, with the archived leaderboard frozen on 16 Apr 2026 [VERIFIED, fetched 2026-10-04]. All numbers must therefore come from MOT17/MOT20 train half-splits and DanceTrack val.

**Evidence labels.** [VERIFIED] = I saw it in the source file or paper text (path, function, or section given). [SUPPORTED] = it follows directly from verified material, or comes from a reliable secondary source. [INFERENCE] = my reasoning, not stated anywhere. [UNCERTAIN] = conflicting or partial evidence. [UNVERIFIED] = I could not check it; the note says what I tried.
**Where the material came from:** every repo was shallow-cloned into `/workspace/zed-ps2/src/` (commit SHAs in §0.3). Papers were downloaded from arXiv or CVF into `/workspace/zed-ps2/papers/` and converted with `pdftotext`. MOT17/MOT20 label zips were downloaded into `/workspace/zed-ps2/data/`. The helper scripts I wrote and ran are in `/workspace/zed-ps2/scripts/`.

---

## 0. Summary tables

### 0.1 Hyperparameters (as used for MOT17 half-val unless noted)

Cost conventions differ between repos. In ByteTrack, BoT-SORT and SparseTrack, `match_thresh` is a **cost** cut-off on `1 − IoU` (optionally score-fused), so `match_thresh = 0.9` means IoU > 0.1 is required (via `lap.lapjv(..., cost_limit=thresh)` in `matching.linear_assignment`). In the OC-SORT family, `iou_threshold` is a **similarity** cut-off (IoU must be ≥ the threshold).

| Tracker (repo @ file) | Det. split | 1st-stage cost | 1st-stage gate | 2nd (low-score) stage | Track init | Lost buffer | MOT20 difference | Other key params |
|---|---|---|---|---|---|---|---|---|
| **ByteTrack** `yolox/tracker/byte_tracker.py::BYTETracker.update` | high: `score > track_thresh` (0.6); low: `0.1 < score < track_thresh` [VERIFIED] | `1 − IoU`, then `fuse_score` (IoU × det score) **unless `--mot20`** [VERIFIED] | `match_thresh = 0.9` default in `tools/track.py` (IoU > 0.1) [VERIFIED]. The paper §4.1 says the match is rejected if IoU < 0.2 (i.e. 0.8) [VERIFIED, code/paper mismatch] | `1 − IoU`, thresh 0.5 (IoU > 0.5), only tracks still in `Tracked` state [VERIFIED] | `score ≥ det_thresh = track_thresh + 0.1` (0.7) [VERIFIED] | `track_buffer = 30`, scaled by `frame_rate/30` [VERIFIED] | README MOT20 command: `--match_thresh 0.7 --mot20` (disables `fuse_score`) [VERIFIED] | Unconfirmed-track stage thresh 0.7. Duplicate removal at `iou_distance < 0.15`. Output filter: area > `min_box_area` (100) and `w/h ≤ 1.6`. **Per-sequence overrides** in `yolox/evaluators/mot_evaluator.py` (see §5.1) [VERIFIED] |
| **OC-SORT** `trackers/ocsort_tracker/ocsort.py::OCSort.update`, `association.py::associate` | high `> det_thresh` (0.6); low 0.1–det_thresh, only if `use_byte` (default **False**) [VERIFIED] | `−(IoU + λ·s_det·VDC)` where VDC = `(π/2 − |Δθ|)/π` [VERIFIED] | `iou_thresh = 0.3` [VERIFIED code, paper §5.1 "Implementations"] | OCR: last **observation** box vs unmatched dets, IoU ≥ 0.3 [VERIFIED] | new KalmanBoxTracker for every unmatched high det. Output once `hit_streak ≥ min_hits (3)` or `frame_count ≤ 3` [VERIFIED] | `max_age = 30` (constructor default; `track_buffer` arg not passed) [VERIFIED] | docs: `--track_thresh 0.4 --mot20` [VERIFIED `docs/GET_STARTED.md`; paper §5.1] | `inertia λ = 0.2`, `deltat Δt = 3`. ORU in `kalmanfilter.py::KalmanFilterNew.freeze/unfreeze`. No adaptive per-sequence threshold by default (comment in `mot_evaluator.py`) [VERIFIED] |
| **BoT-SORT** `tracker/bot_sort.py::BoTSORT.update` | high `track_high_thresh = 0.6`, low `track_low_thresh = 0.1` [VERIFIED] | IoU dist (+`fuse_score` unless mot20). With ReID: `min(iou_d, emb_d')` where `emb_d' = cos_d/2`, set to 1 if `> appearance_thresh (0.25)` or `iou_d > proximity_thresh (0.5)` [VERIFIED; paper §3.3 Eq. 12–13] | `match_thresh = 0.8` [VERIFIED] | IoU, thresh 0.5 [VERIFIED] | `new_track_thresh = 0.7` (= high + 0.1 under `--default-parameters`) [VERIFIED] | 30 [VERIFIED] | `--default-parameters`: MOT20 `match_thresh = 0.7`, `args.mot20 = MOT == 20` [VERIFIED `tools/track.py`] | KF state `(x, y, w, h, vx, vy, vw, vh)` (paper §3.1). Feature EMA `alpha = 0.9` (`STrack.update_features`). CMC default `--cmc-method file` (pre-computed VidStab affine) [VERIFIED] |
| **Deep OC-SORT** `trackers/integrated_ocsort_embedding/ocsort.py::OCSort.update`, `association.py::associate` | **high only:** `scores > det_thresh` (0.6). There is **no BYTE stage** in this tracker [VERIFIED: no `dets_second`/`use_byte` in file] | `−(IoU + VDC + w·cos_sim)`. Weight `w` comes from `compute_aw_new_metric` (base `w_assoc_emb = 0.75` + bonus) [VERIFIED] | `iou_thresh = 0.3` [VERIFIED] | OCR (IoU only) [VERIFIED] | as OC-SORT, `min_hits = 3` | `max_age = 30` [VERIFIED] | README: `--track_thresh 0.4` for mot20 [VERIFIED] | Dynamic EMA `α_t = α_f + (1−α_f)(1 − (s−σ)/(1−σ))`, `α_f = 0.95` [VERIFIED code lines 417–420, paper Eq. 3]. `aw_param ε = 0.5` (MOT), `w = 1.25`, `ε = 1` (DanceTrack) [VERIFIED README + paper §4]. Best runs use `--grid_off --new_kf_off` [VERIFIED README] |
| **Hybrid-SORT** `trackers/hybrid_sort_tracker/hybrid_sort.py::Hybrid_Sort.update`, `association.py` | high > 0.6, low 0.1–0.6, `use_byte = True` [VERIFIED exp file] | `−(HMIoU + Σ_4 corners ROCM − w_TCM·|s_det − s_trk_KF|)` [VERIFIED `associate_4_points_with_score`] | `iou_thresh = 0.25` (MOT17/MOT20), 0.15 (DanceTrack) [VERIFIED exp files; paper "Implementation Details"] | BYTE stage subtracts `|s_det − s_trk_linear|·w` [VERIFIED `cal_score_dif_batch_two_score`] | as OC-SORT | 30 (OC-SORT default) | `track_thresh = 0.4` for MOT20 [VERIFIED `yolox_x_mix_mot20_ch_hybrid_sort.py`] | `inertia = 0.05` **per corner** (4 corners summed → 0.2 total, matching the paper's "ROCM weight 0.2") [VERIFIED code; equivalence = INFERENCE]. TCM weights 1.0/1.0 (MOT), 1.5/1.0 (Dance, paper). ROCM sums intervals 1..3 (`KalmanBoxTracker.update`) [VERIFIED] |
| **SparseTrack** `tracker/sparse_tracker.py::SparseTracker` | high > `track_thresh` (0.6), low > 0.1 [VERIFIED] | IoU (+`fuse_score` unless mot20) inside `DCM` per depth level [VERIFIED] | `match_thresh = 0.85` (MOT17 ab.), 0.7 (MOT20 ab.), 0.6 (MOT20 test) [VERIFIED cfgs] | DCM with `depth_levels_low = 8`, cost thresh 0.3 [VERIFIED] | `confirm_thresh = 0.7` | `track_buffer` 30 (MOT17), 60 (MOT20) [VERIFIED] | `mot20 = True` disables `fuse_score` | GMC through C++ `pbcvt.GMC` (OpenCV videostab, `down_scale = 4`) [VERIFIED `python_module.cpp`] |

### 0.2 Licences (code, weights, data)

| Asset | Licence | Status / flag |
|---|---|---|
| ByteTrack code (`ifzhang/ByteTrack`) | MIT | [VERIFIED `LICENSE`] |
| OC-SORT code (`noahcao/OC_SORT`) | MIT (header "Copyright (c) 2021 Yifu Zhang") | [VERIFIED] |
| BoT-SORT code (`NirAharon/BoT-SORT`) | MIT, **but it vendors `yolov7/` under GPL-3.0** (`yolov7/LICENSE.md`) and `fast_reid/` under Apache-2.0 | [VERIFIED] ⚠ Do not ship the `yolov7/` folder |
| Deep OC-SORT code | MIT | [VERIFIED `LICENSE.md`] |
| Hybrid-SORT code | MIT (`fast_reid/` Apache-2.0) | [VERIFIED] |
| SparseTrack code | MIT | [VERIFIED] |
| TrackEval | MIT | [VERIFIED] |
| YOLOX (`Megvii-BaseDetection/YOLOX`) | Apache-2.0 | [VERIFIED `LICENSE`] |
| **boxmot** (`mikel-brostrom/boxmot`) | **AGPL-3.0** (`LICENSE`, `pyproject.toml`; README offers "commercial support") | [VERIFIED] ⛔ Do not import or copy |
| **Ultralytics** (`ultralytics/ultralytics`) | **AGPL-3.0** (GitHub licence API) | [VERIFIED] ⛔ This includes YOLOv8/11 code and the Ultralytics-distributed weights [SUPPORTED] |
| MOT17 / MOT20 data | **CC BY-NC-SA 3.0** | [VERIFIED]: a Wayback snapshot (2022-12-09) of `motchallenge.net/data/MOT17/` and `/MOT20/` links `creativecommons.org/licenses/by-nc-sa/3.0/` in the site footer. The current static page and the label zips contain no licence text. ⚠ Non-commercial and share-alike; fine for internal R&D benchmarking, not for training shipped models (legal review needed) [INFERENCE] |
| DanceTrack | Annotations CC BY 4.0. **Videos/images "non-commercial research purposes only"** | [VERIFIED `DanceTrack/README.md` "Agreement"; HF card says `cc-by-4.0`] |
| SOMPT22 | Site: annotations CC BY 4.0, dataset "non-commercial research purposes only". arXiv bibtex: CC BY-NC-SA 4.0 | [VERIFIED `sompt22.github.io`] [UNCERTAIN which governs → treat as NC] |
| CrowdHuman (in every ByteTrack-family detector) | not stated on the pages I fetched | [UNVERIFIED]: `crowdhuman.org` and `/download.html` gave no licence text |
| ByteTrack / OC-SORT / SparseTrack detector weights | Released in MIT repos, but trained on MOT17/MOT20 (NC-SA) + CrowdHuman (+ CityPersons/ETHZ for the full models) | [INFERENCE] ⚠ Weight licence is legally unclear because of the NC training data. OK for benchmarking only |
| FastReID SBS-S50 MOT17/MOT20 ReID weights (BoT-SORT) | Repo Apache/MIT; trained on MOT patches | ⚠ **Leakage**, see §5.6 |

### 0.3 Commits read

ByteTrack `d1bf019` (2022-12-11) · OC_SORT `8462e7e` (2026-04-21) · BoT-SORT `2519854` (2022-10-30) · Deep-OC-SORT `6bb51d0` (2023-05-06) · HybridSORT `396f8d3` (2024-04-02) · SparseTrack `499844f` (2025-03-06) · TrackEval `12c8791` (2022-11-29) · boxmot `259625c` (2026-09-18, licence check only) · YOLOX `6ddff48` (2025-06-08) · DanceTrack (latest default branch). Links below use `blob/<branch>/path`. Pin to these SHAs when you reproduce.

---

## 1. Association pipelines (verified against code and papers)

### 1.1 ByteTrack — [paper arXiv:2110.06864](https://arxiv.org/abs/2110.06864) · [repo](https://github.com/ifzhang/ByteTrack)

**Code path.** `yolox/tracker/byte_tracker.py`, class `BYTETracker.update()`, plus `yolox/tracker/matching.py` (`iou_distance`, `fuse_score`, `linear_assignment`) [VERIFIED].

1. **Score split.** `remain_inds = scores > track_thresh` (high). `inds_second = (scores > 0.1) & (scores < track_thresh)` (low). A detection with score exactly equal to `track_thresh` is in neither set [VERIFIED]. With 6-column YOLOX output the score is `obj × cls`.
2. **Step 2 (first association).** `strack_pool = tracked ∪ lost`. `STrack.multi_predict` runs the KF prediction (state `x, y, a, h` + velocities; `_std_weight_position = 1/20`, `_std_weight_velocity = 1/160`, `kalman_filter.py`). The cost is `iou_distance`. If `not args.mot20`, it becomes `fuse_score(cost, dets) = 1 − IoU·score`. Then `linear_assignment(cost, thresh=match_thresh)` via `lap.lapjv(extend_cost=True, cost_limit=thresh)`. A matched lost track is `re_activate`d with the same ID [VERIFIED].
3. **Step 3 (second association).** Only `r_tracked_stracks` (unmatched tracks whose state is `Tracked`, so lost tracks are excluded) are matched to low-score dets. Cost is plain IoU distance, `thresh = 0.5`, with no `fuse_score`. Tracks still unmatched → `mark_lost()` [VERIFIED].
4. **Unconfirmed tracks** (`not is_activated`, i.e. born last frame) are matched to the remaining high dets. Cost is IoU (+`fuse_score` unless mot20), `thresh = 0.7`. Unmatched unconfirmed tracks are **removed** [VERIFIED].
5. **Step 4 (init).** Remaining high dets with `score ≥ det_thresh = track_thresh + 0.1` start a new track. On frame 1 the track is activated immediately; otherwise it is confirmed at its next match (`STrack.activate`/`update`) [VERIFIED].
6. **Step 5.** Lost tracks with `frame_id − end_frame > max_time_lost` are removed. `buffer_size = int(frame_rate/30 · track_buffer)`, but the evaluator never passes `frame_rate`, so the default 30 is always used. `remove_duplicate_stracks` drops whichever of a tracked/lost pair with IoU dist < 0.15 is younger [VERIFIED].

**Defaults.** `tools/track.py`: `--track_thresh 0.6 --track_buffer 30 --match_thresh 0.9 --min-box-area 100`, `--mot20` off [VERIFIED].
**Paper §4.1 "Implementation details".** τ = 0.6; "if the IoU … is smaller than 0.2, the matching will be rejected"; lost tracklets are kept 30 frames [VERIFIED]. ⚠ The code default 0.9 (IoU > 0.1) differs from the paper's 0.2. Report which one you use.
**MOT20.** README test command is `--match_thresh 0.7 --mot20` [VERIFIED]. The `mot20` flag only disables `fuse_score` (in steps 2 and 4).
**Per-sequence hacks.** `yolox/evaluators/mot_evaluator.py::MOTEvaluator.evaluate` sets `track_buffer = 14` for MOT17-05/06, 25 for MOT17-13/14, and `track_thresh` = 0.65 (MOT17-01, 06), 0.7 (12), 0.67 (14), 0.3 (MOT20-06/08) [VERIFIED]. Two of these are **val-half sequences** (05 and 13). They are hand-tuned (on test, per the OC-SORT docs, which call them "adaptive detection thresholds" for test submissions). For a clean protocol, **disable them** or report both [INFERENCE].
**Output filtering.** The evaluator keeps a box only if `w·h > min_box_area` and `w/h ≤ 1.6` ("vertical"). This post-filter changes MOTA/IDF1 and must be applied identically to every tracker [VERIFIED].

### 1.2 OC-SORT — [paper arXiv:2203.14360](https://arxiv.org/abs/2203.14360) · [repo](https://github.com/noahcao/OC_SORT)

**Code path.** `trackers/ocsort_tracker/ocsort.py` (`KalmanBoxTracker`, `OCSort.update`, `k_previous_obs`, `speed_direction`), `association.py` (`associate`, `speed_direction_batch`, `iou_batch`), `kalmanfilter.py` (`KalmanFilterNew.freeze/unfreeze/update`) [VERIFIED].

- **KF.** SORT 7-dim state `[x, y, s, r, ẋ, ẏ, ṡ]` (s = area, r = aspect), with `R[2:,2:] *= 10`, `P[4:,4:] *= 1000`, `P *= 10`, `Q[-1,-1] *= 0.01`, `Q[4:,4:] *= 0.01` (`KalmanBoxTracker.__init__`) [VERIFIED].
- **OCM (velocity-direction consistency, paper §4.2, Eq. 8).** Each track keeps `velocity` = the unit direction from the observation Δt steps back to the current one (`KalmanBoxTracker.update`, Δt = `delta_t = 3`, falls back to the nearest available earlier obs). In `associate`, the direction from the track's k-th-previous observation to each detection is compared with the track velocity. `diff_angle = arccos(clip(cos))`, `angle_cost = (π/2 − |diff_angle|)/π ∈ [−0.5, 0.5]`, masked for tracks with no previous obs, times `vdc_weight = inertia = 0.2`, **times detection score**. Then `linear_assignment(−(IoU + angle_cost))`. Matches with IoU < `iou_threshold` are filtered (`_filter_matches`). Shortcut: if the binarised `IoU > thr` matrix is already one-to-one, it is used directly [VERIFIED]. Paper: λ = 0.2, Δt = 3, IoU threshold 0.3, det threshold 0.4 (MOT20) / 0.6 (others) (§5.1 "Implementations") [VERIFIED]. ⚠ The paper says "absolute difference of angles"; the code uses the `(π/2 − |θ|)/π` transform weighted by score.
- **BYTE stage (optional).** `--use_byte` (default False). It uses the KF-predicted boxes and the same `iou_threshold` [VERIFIED].
- **OCR (Observation-Centric Recovery, heuristic described at the end of §4.2; the paper has no §4.3).** Unmatched high dets vs **last observed boxes** (`last_boxes`, not the KF prediction) of unmatched tracks. IoU, gate 0.3 [VERIFIED].
- **ORU (Observation-centric Re-Update, §4.1, Eq. 6–7).** `update(None)` calls `kf.update(None)`; on the first miss `freeze()` snapshots the KF state. When the track is re-associated, `unfreeze()` restores the snapshot, builds a **linear virtual trajectory in (x, y, w, h)** between the last two real observations (`index1, index2`), and replays predict/update along it [VERIFIED]. The paper text says constant velocity, `z̃_t = z_t1 + (t−t1)/(t2−t1)·(z_t2 − z_t1)` [VERIFIED].
- **Output.** Uses the **last observation** box, not the KF state. Requires `time_since_update < 1` and (`hit_streak ≥ min_hits` or `frame_count ≤ min_hits`) [VERIFIED]. ⚠ This means coasted (unobserved) tracks are never output, so OC-SORT gets no FP/TP from extrapolated boxes during occlusion.
- **Defaults.** `utils/args.py`: `track_thresh 0.6, iou_thresh 0.3, min_hits 3, inertia 0.2, deltat 3, asso iou, use_byte False` [VERIFIED]. MOT20: `--track_thresh 0.4 --mot20` (`docs/GET_STARTED.md`) [VERIFIED]. The `mot20` flag has no effect inside `OCSort` [INFERENCE: the class never reads it].

### 1.3 BoT-SORT — [paper arXiv:2206.14651](https://arxiv.org/abs/2206.14651) · [repo](https://github.com/NirAharon/BoT-SORT)

**Code path.** `tracker/bot_sort.py` (`STrack`, `BoTSORT.update`), `tracker/gmc.py` (`GMC`), `tracker/kalman_filter.py`, `tracker/matching.py` [VERIFIED].

- **KF (paper §3.1).** 8-dim state `(x, y, w, h, vx, vy, vw, vh)` with **w, h** directly, instead of ByteTrack's aspect a. Noise is proportional to w and h (`_std_weight_position = 1/20`, `_std_weight_velocity = 1/160`) [VERIFIED].
- **CMC (paper §3.2).** Paper: "OpenCV … Video Stabilization module with affine transformation … keypoints, then sparse optical flow … translation-based local outlier rejection … RANSAC" [VERIFIED]. Code `GMC(method=…)` offers `orb`, `sift`, `ecc` (`cv2.findTransformECC`, `MOTION_EUCLIDEAN`, 5000 iterations, eps 1e-6, downscale 2), `sparseOptFlow` (`goodFeaturesToTrack(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3)` + `calcOpticalFlowPyrLK` + `estimateAffinePartial2D(RANSAC)`, downscale 2), `file` (default in `tools/track.py`; reads pre-computed VidStab affines from `tracker/GMC_files/{MOT17_ablation|MOTChallenge}/GMC-<seq>.txt`, one line per frame), and `none` [VERIFIED]. The warp is applied to the KF mean and covariance by `STrack.multi_gmc` [VERIFIED].
  - ⚠ `applyFile` reads **one line per call, sequentially**. Skipping frames or restarting a sequence silently desynchronises it [VERIFIED by reading `applyFile`].
  - `GMC_files/MOT17_ablation/GMC-MOT17-02.txt` has 299 lines, which equals the val-half length (600 → frames 302–600) [VERIFIED by `wc -l` + §5.1 split]. These files are a **CPU-free CMC cache for MOT17 val-half (7 seqs) and the full MOT17/MOT20 train+test** in an MIT repo, so they are usable for our CPU benchmark [INFERENCE].
- **IoU–ReID fusion (paper §3.3, Eq. 12–13).** `ious_dists_mask = iou_dist > proximity_thresh (0.5)`. `emb_dists = embedding_distance/2`; set to 1 where `> appearance_thresh (0.25)` or masked. `dists = min(ious_dists(fused), emb_dists)` [VERIFIED]. The same fusion is applied to unconfirmed tracks. The low-score stage is IoU-only, thresh 0.5. Paper: θ_iou = 0.5, θ_emb = 0.25, EMA α = 0.9 [VERIFIED].
- **Defaults** (`tools/track.py`): `track_high_thresh 0.6, track_low_thresh 0.1, new_track_thresh 0.7, track_buffer 30, match_thresh 0.8, aspect_ratio_thresh 1.6, min_box_area 10, cmc-method file, proximity 0.5, appearance 0.25` [VERIFIED]. `--default-parameters` applies the same per-sequence overrides as ByteTrack, plus MOT20 `match_thresh = 0.7`, `new_track_thresh = high + 0.1`, and `exp.test_conf = max(0.001, track_low_thresh − 0.01)` [VERIFIED].
- **Latency reporting.** `tools/track.py` prints separate "Detector + Tracker" and "Tracker only" FPS (`trackerTimer`) [VERIFIED]. This is the format we should copy.

### 1.4 Deep OC-SORT — [paper arXiv:2302.11813](https://arxiv.org/abs/2302.11813) · [repo](https://github.com/GerardMaggiolino/Deep-OC-SORT)

**Code path.** `trackers/integrated_ocsort_embedding/ocsort.py::OCSort.update`, `association.py::associate`, `compute_aw_new_metric`, `cmc.py::CMCComputer`, `embedding.py::EmbeddingComputer` [VERIFIED]. `main.py` imports `integrated_ocsort_embedding` [VERIFIED].

- **Detections.** Only `scores > det_thresh` are used. There is no low-score BYTE stage [VERIFIED].
- **CMC (paper §3.1 items 1–3).** Paper: OpenCV-contrib VidStab similarity transforms (§4) [VERIFIED]. Code `CMCComputer(method="file")` reads cached affines from `cache/cmc_files/{MOT17_ablation, MOT20_ablation, MOTChallenge, DanceTrack}` (shipped in the repo). The `sparse` alternative uses `goodFeaturesToTrack(maxCorners=3000, …, mask=bg mask)` + `estimateAffinePartial2D(RANSAC)`. The correction is applied to the KF (`apply_affine_correction`) and therefore to OOS/ORU, OCM and OCR points [VERIFIED].
- **Dynamic Appearance (DA, Eq. 3).** `trust = (s − det_thresh)/(1 − det_thresh)`, `dets_alpha = α_f + (1 − α_f)(1 − trust)` with `alpha_fixed_emb = 0.95`. A low-confidence det (s → σ) gives α → 1 (the embedding is ignored). Used in `KalmanBoxTracker.update_emb(emb, alpha)` as `emb ← α·emb + (1−α)·new`, then normalised [VERIFIED].
- **Adaptive Weighting (AW).** `compute_aw_new_metric(emb_cost, w_assoc_emb=0.75, max_diff=aw_param=0.5)`. Per row (and per column), it adds `min(top1 − top2, max_diff)/2` of cosine similarity to the base weight. Final cost = `−(IoU + angle_cost + w ⊙ cos_sim)` [VERIFIED]. Paper: `a_w = 0.75`, ε = 0.5 (MOT17/20); 1.25 and 1.0 (DanceTrack) [VERIFIED].
- **ReID on val-halves.** `embedding.py::_get_general_model` uses torchreid `osnet_ain_x1_0` with `osnet_ain_ms_d_c.pth.tar` "because the MOT17/20 SBS models are trained over the half-val we evaluate on" (code docstring) [VERIFIED]. ⚠ This is a leakage trap; see §5.6.
- **Detector on MOT20 "val".** `main.py`: "Just use the mot17 test model as the ablation model for 20" (`bytetrack_x_mot17.pth.tar`, 800×1440) [VERIFIED].
- **Reported runs.** These use `--post` (`utils.dti` linear interpolation, `n_min = 30, n_dti = 20`), which is **offline** post-processing [VERIFIED].

### 1.5 Hybrid-SORT — [paper arXiv:2308.00783](https://arxiv.org/abs/2308.00783) · [repo](https://github.com/ymzis69/HybridSORT)

**Code path.** `trackers/hybrid_sort_tracker/hybrid_sort.py` (`KalmanBoxTracker` with `KalmanFilterNew_score_new(dim_x=9, dim_z=5)`, `Hybrid_Sort.update`), `association.py` (`hmiou`, `associate_4_points_with_score`, `cost_vel`, `cal_score_dif_batch`, `cal_score_dif_batch_two_score`). Configs live in `exps/example/mot/*_hybrid_sort.py` [VERIFIED].

- **TCM (Tracklet Confidence Modeling).** The detection score is added to the KF state: z = `[x, y, s, score, r]` (`convert_bbox_to_z`). `predict()` returns the KF-predicted score clipped to `[track_thresh, 1]` and a **linear** extrapolation `conf − (conf_pre − conf)` clipped to `[0.1, track_thresh]` [VERIFIED].
  - First stage: `angle_diff_cost −= |s_det − s_KF| · TCM_first_step_weight`.
  - BYTE stage: `iou_left −= |s_det − s_linear| · TCM_byte_step_weight` [VERIFIED].
  - The paper says it uses a KF for high-score and Linear Prediction for low-score ("Tracklet Confidence Modeling"; ablation in its Table 5) [VERIFIED].
- **HMIoU.** `hmiou = HIoU · IoU`, where `HIoU = (min(y2) − max(y1)) / (max(y2) − min(y1))` [VERIFIED code; paper "Height Modulated IoU"]. HIoU itself is unclipped (it goes negative for vertically disjoint boxes), but there the IoU factor is 0 (`h = max(0, …)`), so HMIoU stays ≥ 0 [VERIFIED by reading `hmiou`].
- **ROCM.** Velocity directions from four corners (`speed_direction_lt/rt/lb/rb`), each **summed over intervals 1..Δt** (without renormalising after the sum). Each corner cost is `cost_vel(..., vdc_weight=inertia)`, multiplied by the det score [VERIFIED]. Paper: "fixed time interval of 3 frames → stack of multiple intervals ranging from 1 to 3 … four corners" [VERIFIED].
- **Params (MOT17 ablation exp).** `use_byte=True, inertia=0.05, iou_thresh=0.25, asso="Height_Modulated_IoU", TCM_first_step=TCM_byte_step=True, weights 1.0/1.0, ckpt pretrained/ocsort_mot17_ablation.pth.tar` [VERIFIED]. MOT20 adds `track_thresh = 0.4`. DanceTrack uses `iou_thresh = 0.15` [VERIFIED]. Paper: TCM weights 1.5/1.0 on DanceTrack and 1.0/1.0 elsewhere; ROCM weight 0.2; IoU gate 0.15/0.25 [VERIFIED]. ⚠ The DanceTrack *exp file* has first-step weight 1.0, not 1.5 (paper) [VERIFIED mismatch].
- ⚠ `exps/example/mot/yolox_x_mix_mot20_ch_valhalf.py` evaluates `val_half.json` with a training set `mix_mot20_ch/train.json`, which `tools/mix_data_test_mot20.py` builds from **full MOT20 train** (`datasets/MOT20/annotations/train.json`) [VERIFIED]. That config leaks.

### 1.6 Licence pitfalls beyond the tables

- **AGPL-3.0:** boxmot (the convenient wrapper around all of the above) and Ultralytics. Do **not** pip-install them into the product environment, copy code from them, or use Ultralytics-exported weights [VERIFIED licences; product implication = INFERENCE]. Re-implement from the MIT originals cited here.
- **GPL-3.0:** `BoT-SORT/yolov7/` [VERIFIED].
- **Training data NC:** MOT17/MOT20 (CC BY-NC-SA 3.0), DanceTrack and SOMPT22 images (NC). Weights trained on these are a legal grey area [INFERENCE].
- **`osnet_ain_ms_d_c`:** the name suggests MSMT17 + DukeMTMC + CUHK03 training. DukeMTMC was withdrawn by its authors and MSMT17 is research-only [INFERENCE / UNVERIFIED: I did not open the torchreid model-zoo page]. Avoid it in product.

---

## 2. Box-only occlusion / visibility estimation

### 2.1 What MOT `gt.txt` visibility actually is (critical)

- **Format** (MOT16 paper Tab. 5 / MOT20 paper Tab. 4) [VERIFIED]: `frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility`, 1-based pixels.
  - For GT, col 7 `conf` is a consider/ignore flag (1/0).
  - Col 8 `class`: 1 ped, 2 person on vehicle, 3 car, 4 bicycle, 5 motorbike, 6 non-motorized vehicle, 7 static person, 8 distractor, 9 occluder, 10 occluder on ground, 11 occluder full, 12 reflection, 13 crowd.
  - Col 9 is the visibility ratio in [0, 1], "due to occlusion and due to **image border cropping**".
  - Det files: col 7 is the score, cols 8–9 are −1 (MOT20 det.txt) [VERIFIED].
  - ByteTrack's `gt_val_half.txt` writer keeps col 9 as float and cols 1–8 as int [VERIFIED `convert_mot17_to_coco.py`].
- **How the visibility value was generated.** MOTChallenge IJCV paper (arXiv:2010.07548, App. C.1.5 "Occlusions"): "There is no need to explicitly annotate the level of occlusion. This value is computed automatically using the annotations. We leverage the assumption that for two or more overlapping bounding boxes the object with the lowest y-value of the bounding box is closest to the camera…" [VERIFIED]. MOT16 paper §2.5 says the same "using the ground plane assumption and the annotations" [VERIFIED]. Occluders, vehicles and static persons are annotated "for computing the occlusion level" (MOT16 §2) [VERIFIED].
- **Our reproduction** (`scripts/vis_oracle_check.py`, first 150 frames per sequence). Visibility = 1 − (fraction of the box covered by (a) any closer box, where closer = larger bottom-edge y2, or (b) the outside of the image). Compared with gt col 9 for class-1/flag-1 boxes [VERIFIED by running]:

| Seq (first 150 fr) | Occluders = all annotated classes: MAE / Spearman ρ | Occluders = pedestrians only: MAE / ρ | n |
|---|---|---|---|
| MOT17-02-FRCNN | 0.013 / 0.995 | 0.056 / 0.941 | 3852 |
| MOT17-04-FRCNN | 0.032 / 0.959 | 0.091 / 0.835 | 6157 |
| MOT17-09-FRCNN | 0.007 / 0.983 | 0.014 / 0.961 | 1110 |
| MOT20-01 | 0.070 / 0.905 | 0.114 / 0.789 | 5863 |

  **Implications.**
  1. The ground truth **is** essentially a box-overlap and bottom-edge depth-order computation. Our method rebuilds the generator, so scoring an estimator that takes **GT boxes as input** is nearly circular and must only be reported as an *implementation sanity check* [SUPPORTED].
  2. The real score has to use **detector boxes**. Static occluders (classes 9–11, cars, static persons) are invisible to a person detector, which gives an irreducible error floor. Even with perfect person boxes, the "pedestrians-only" column shows MAE ≈ 0.01–0.11 [VERIFIED on these frames].
  3. The tie-break is uncertain: "lowest y-value" is ambiguous in image coordinates. Bottom-edge-closest reproduced GT well, which favours that reading [SUPPORTED].
- **Fully invisible targets are still GT.** In MOT17-02, 29.3% of evaluated pedestrian rows have visibility exactly 0, and 20.1% in MOT17-05 [VERIFIED by counting]. TrackEval does **not** filter by visibility (§5.5). My oracle test on MOT17 val-half: submitting GT boxes only where vis > 0 caps the result at **MOTA 89.1 / IDF1 94.2 / HOTA 91.1**. Vis > 0.1 gives MOTA 84.0; vis > 0.25 gives 74.3. On MOT20 val-half, vis > 0 gives MOTA 96.9 (`scripts/run_trackeval.py`) [VERIFIED by running]. Coasting/extrapolating through occlusion (which our delayed association can do) is therefore directly rewarded by MOTA/IDF1 [INFERENCE].
- **DanceTrack has no usable visibility.** In `val/dancetrack0010/gt/gt.txt` (read by HTTP range from `huggingface.co/datasets/noahcao/dancetrack/val.zip`) all 7029 rows have col 7 = 1, col 8 = 1, col 9 = 1. IDs start at 0 and `frameRate = 20` [VERIFIED]. Score visibility only on MOT17/MOT20 (and SOMPT22 if its gt has the column [UNVERIFIED]).

### 2.2 SparseTrack pseudo-depth — [paper arXiv:2306.05238](https://arxiv.org/abs/2306.05238) · [repo](https://github.com/hustvl/SparseTrack)

- **Paper §3.2 "Pseudo-Depth Method".** The depth is `L_p = H − y_p`, where `y_p` is the y of the bottom-centre point of the box (the ground-contact projection) and H is the image height. It assumes the camera is above the ground and the objects stand on one ground plane [VERIFIED].
- **§3.3 / Algorithm 1 "Depth Cascade Matching (DCM)".** Split `[D_min, D_max]` of the dets, and separately of the tracks, into k levels. Associate level by level from near to far, carrying the unmatched items into the next level [VERIFIED].
- **Code.** `tracker/sparse_tracker.py`: `STrack.deep_vec` returns `[cx, y2, 2000 − y2]`, a hard-coded 2000 instead of H; because levels use min–max splits, ordering is unaffected [VERIFIED; equivalence = INFERENCE]. Also `SparseTracker.get_deep_range(obj, step)` (np.arange bins), `get_sub_mask`, and `DCM(detections, tracks, …, levels, thresh, is_fuse)`.
- In `update()`, high dets use `depth_levels = 1` (a single level, i.e. a plain BYTE step) and low dets use `depth_levels_low = 8` with cost thresh 0.3. GMC is `pbcvt.GMC` (OpenCV `videostab::KeypointBasedMotionEstimator`, `MM_SIMILARITY`, `down_scale = 4`) [VERIFIED].
- **Takeaway for us.** Bottom-edge depth ordering is established in the literature (SparseTrack) and is literally the MOT GT convention. Our occlusion estimate should share the bottom edge y2 as its depth key [SUPPORTED].

### 2.3 Other occlusion-aware trackers (pointers; not code-verified)

- **OccluTrack** ([arXiv:2309.10360](https://arxiv.org/abs/2309.10360)): abnormal-motion suppression in the KF under partial occlusion, pose-guided ReID, occlusion-aware IoU/appearance association [VERIFIED abstract only].
- **Hybrid-SORT HMIoU / TCM**: a confidence drop works as a weak occlusion cue (§1.5) [VERIFIED].
- **Deep OC-SORT DA**: a low-confidence det gets less appearance update, a confidence-as-occlusion proxy (§1.4) [VERIFIED].
- **UCMCTrack** ([arXiv:2312.08952](https://arxiv.org/abs/2312.08952)): a ground-plane KF with Mapped Mahalanobis Distance replaces IoU [VERIFIED abstract].
- **DepthMOT** ([arXiv:2404.05518](https://arxiv.org/abs/2404.05518)): learned depth. Not box-only [VERIFIED abstract].
- **BoostTrack++** ([arXiv:2408.13003](https://arxiv.org/abs/2408.13003)): title verified, content [UNVERIFIED].

### 2.4 Concrete protocol for scoring our visibility estimate against gt

1. **GT set.** For each frame, take rows with `class == 1 and flag == 1` (exactly TrackEval's kept set, §5.5) on the val half of each sequence. Keep all other rows (classes 2–13) only for the GT-box sanity oracle.
2. **Association of estimates to GT.** For each frame, Hungarian-match our tracker's **output boxes** (or the raw detections, a separate report) to the GT boxes with IoU ≥ 0.5. This is the same similarity and threshold TrackEval CLEAR/Identity use (`THRESHOLD 0.5`) [VERIFIED default]. Score each matched pair (v̂, v).
3. **Report coverage.** Give the matched fraction per visibility bin: matching is biased towards visible people, and unmatched low-visibility GT would otherwise disappear from the error.
4. **Metrics.**
   - MAE, RMSE.
   - **Spearman ρ** (the ranking of who is more occluded is what the association gate needs). Kendall τ-b handles ties at 0/1.
   - **Binned calibration**: GT bins `[0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]`, with a reliability table of mean v̂ against mean v per bin, an ECE-style `Σ_b n_b/N · |mean v̂_b − mean v_b|`, and bin counts.
   - **Threshold AUROC/AP** for "heavily occluded" (v < 0.5) and "nearly invisible" (v < 0.25), since the delayed-association trigger is a threshold decision.
5. **Stratify** by dataset (MOT17 vs MOT20), by moving vs static camera (MOT17-05/10/11/13 move per the MOT17 page descriptions [VERIFIED page]), and by crowd density (GT count per frame).
6. **Always report two baselines.** The *GT-box all-classes oracle* (an implementation check, expect MAE ≈ 0.01–0.03 on MOT17) and the *GT-box pedestrians-only oracle* (the floor any person-detector-based estimate can reach). Both are computed by `scripts/vis_oracle_check.py` (switch `max_frames` to all frames for the final run).
7. Do **not** tune estimator parameters on the val half you report. Tune on train-half GT boxes or detections [INFERENCE].

---

## 3. Bounded-delay / online MHT

### 3.1 Primary sources

- **Reid 1979:** D. Reid, "An algorithm for tracking multiple targets", *IEEE Trans. Automatic Control* 24(6):843–854, Dec 1979, doi:10.1109/TAC.1979.1102177 [VERIFIED via Crossref]. I did not read the full text; the claim that it originated hypothesis-oriented MHT is [SUPPORTED, standard attribution].
- **Kim, Li, Ciptadi, Rehg, "Multiple Hypothesis Tracking Revisited", ICCV 2015** ([CVF PDF](https://openaccess.thecvf.com/content_iccv_2015/papers/Kim_Multiple_Hypothesis_Tracking_ICCV_2015_paper.pdf)) [VERIFIED text]:
  - §3.1: a track tree per observation; each frame, children are spawned for the observations inside the gate, plus a dummy (missed) branch.
  - §3.2 **Gating**: Mahalanobis distance `d² ≤ d_th` around the KF-predicted location.
  - §3.3: track score `S = w_mot·S_mot + w_app·S_app` (LLR).
  - §3.4: global hypothesis = **MWIS** over the conflict graph.
  - §3.5 **N-scan pruning**: trace each selected branch back to frame k − N and prune the sibling subtrees, i.e. "consolidate the data association decisions for old observations up to frame k − (N − 1)". Also cap the branches per tree at `B_th` (keep the top by score).
  - Table 1: MHT-DAM `N-scan = 5, B_th = 100, N_miss = 15, P_D = 0.9, d_th = 12, (w_mot, w_app) = (0.1, 0.9)`; MHT `d_th = 6, (1.0, 0.0)`.
  - **N-scan pruning is a fixed-lag decision rule.** The association for frame k − N is final at frame k, which is exactly our K [SUPPORTED].
- **NOMT** (Choi, ICCV 2015, [arXiv:1504.02340](https://arxiv.org/abs/1504.02340)): "near-online". At every frame it re-solves association over the window `[t − τ, t]`, can "fix any association error made in the past" inside the window, and uses τ = 10 (~10 FPS) [VERIFIED text]. This is the closest published precedent for a fixed-lag window and for the "near-online" label.
- **MHT-bLSTM** (Kim, Li, Rehg, ECCV 2018, pp. 200–215): MHT with LSTM gating, "near-online" results on MOT16/17 [SUPPORTED via the CVF abstract].
- I found no widely used code-released *real-time* fixed-lag MHT for pedestrians. Search results surfaced a multi-camera MHT ([arXiv:1901.08787](https://arxiv.org/abs/1901.08787), real-time single-camera tracker + MHT across cameras) and a 2026 general "delayed disambiguation" paper (EviTrack, arXiv:2605.19283) [SUPPORTED by search snippets only; not read]. Neither is a drop-in baseline [UNCERTAIN].

### 3.2 Design implications for our ambiguity-triggered fixed-lag association [INFERENCE unless marked]

- **Trigger.** Branch only for *ambiguous look-alike pairs*: two tracks gated to the same detection(s), with a small margin between the best and second-best cost (exactly Deep OC-SORT's AW "top1 − top2" quantity, `compute_aw_new_metric` [VERIFIED]). Visibility from §2 gates this further. Everything else commits immediately (K = 0 path), as in BYTE/OC-SORT.
- **Bound the hypothesis count.** Per ambiguous cluster, allow at most 2 or 3 hypotheses (pair swap / no swap / one missed) and depth K. Then the worst case is `O(clusters · 2^K)` frames of bookkeeping. Cap it like `B_th` and fall back to greedy if the cluster size or time budget is exceeded, so the tracker *degrades gracefully* with density.
- **Commit rule.** At frame t, finalise the frame t − K output, i.e. N-scan with N = K. Never revise an already emitted frame.

### 3.3 Fixed lag, TrackEval and the online-only rule

- TrackEval scores whatever is in the per-sequence result file. It has no notion of when a line was produced, and it checks only that frame keys ≤ `seqLength` and that IDs are unique per timestep (`_load_raw_file`, `_check_unique_ids`) [VERIFIED]. A fixed-lag tracker therefore evaluates without modification, and **the honesty burden is on us**.
- **Protocol** [INFERENCE]:
  1. Emit frame t's boxes and IDs exactly once, at processing time t + K. Write them to the result file in emission order and log `emit_wallclock`. Do not rewrite earlier frames at all: no back-filling of IDs, no retro-merging tracklets, no interpolation.
  2. At the end of a sequence, flush the last K frames with the same decision logic and no extra look-ahead past the end.
  3. **Always report K = 0** (pure online, the same code with branching off) next to K ∈ {2, 4, 8} frames. State K in frames **and** ms per dataset: K/fps with MOT17 at 30/25/14 fps (MOT17-05 is 14 fps, MOT17-13 is 25 fps), MOT20 at 25 fps, DanceTrack at 20 fps [VERIFIED fps from the MOT pages and DanceTrack seqinfo].
  4. Add a "K-frame-delayed online" column label. Do not call it "online". NOMT's "near-online" is the closest precedent [SUPPORTED].
  5. **Interpolation** (`tools/interpolation.py::dti` in ByteTrack/OC-SORT, Deep OC-SORT `--post`) is offline and must be excluded from the main table. ByteTrack's `interpolation.py` even **grid-searches `n_min`, `n_dti` against GT MOTA** (lines ~131–142) [VERIFIED].
  6. The MOTChallenge "online" (ON) column was a method attribute in the IJCV paper tables [VERIFIED legend "ON – online method"]. I found no formal definition of the rule, and the benchmark is now offline [UNVERIFIED: no rules page was reachable].

---

## 4. Gate-event metrics (people counting, tailgating)

### 4.1 Literature found

- **Line-of-interest counting, Ma & Chan CVPR 2013** ("Crossing the Line: Crowd Counting by Integer Programming with Local Features", [CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Ma_Crossing_the_Line_2013_CVPR_paper.html)). Datasets UCSD (two directions) and LHI. §5 evaluation:
  - **Cumulative count**: MSE and absolute error between predicted and GT cumulative counts, averaged over all frames.
  - **Instantaneous crossings**: Hungarian-match GT crossing times to predicted ones by minimal temporal distance and plot a **recall–distance curve** (fraction of matches within d frames) [VERIFIED text].
- **PCDS** (Sun et al., "Benchmark data and method for real-time people counting in cluttered scenes using depth sensors", [arXiv:1804.04339](https://arxiv.org/abs/1804.04339); [repo](https://github.com/shijieS/people-counting-dataset)). Kinect-V1 depth + RGB at bus doors, 30 scenes. Labels per video are `DepthVideoName, EnteringNumber, ExitingNumber, VideoType` (video-level counts only) [VERIFIED README]. The README states "5,464 pairs … 10,908 videos", which is internally inconsistent [VERIFIED]. Licence [UNVERIFIED].
- **UNICITY** (Dumoulin et al., AVSS-W 2018; [Zenodo 10.5281/zenodo.2556679](https://zenodo.org/records/2556679)). Depth maps in security airlocks: 58k images, 65 sequences of one or two people including **tailgating** and attacks, head/shoulder annotations, evaluation code [VERIFIED Zenodo page]. Licence [UNVERIFIED: not in the fetched text]. Depth-only and top-view, so it does not transfer directly to our RGB oblique view [INFERENCE].
- I found **no standard public RGB tailgating benchmark with event-level precision/recall definitions**. What I found were theses and embedded-system reports (e.g. UTAR thesis, Turku UAS thesis) reporting accuracy/FPS on custom footage [SUPPORTED by search; not read in full].

### 4.2 Proposed definitions for PS-2 [INFERENCE; derived from the sources above]

- **Gate.** A directed line segment G in image coordinates, or a polygon pair (inside/outside). The trajectory point is the **bottom-centre** of the box (the ground-contact point, consistent with §2). A *crossing event* `e = (id, t, dir)` fires when the point's signed side of G changes and stays changed for ≥ h frames (hysteresis, so jitter does not double count) inside the segment's extent.
- **GT events** come from GT trajectories (class 1, flag 1, val half), including vis = 0 interpolated boxes. **Predicted events** come from the tracker output after the K-lag.
- **Counting error** per direction and window W (e.g. 10 s and the whole sequence):
  - absolute error `|N̂ − N|`;
  - signed bias;
  - MAE over windows (Ma & Chan's cumulative error) [SUPPORTED].
- **Event detection P/R/F1.** Hungarian-match predicted to GT events of the same direction with temporal tolerance δ (e.g. 0.5 s), then plot the Ma & Chan recall–distance curve over δ [SUPPORTED].
- **Attribution accuracy.** A matched event counts as correctly attributed if the predicted track ID maps to the crossing GT ID under the **global IDF1 bijection** (TrackEval Identity's `linear_sum_assignment` over ID pairs). Also report the per-event local majority-overlap variant. This measures whether the *right person* is credited with the crossing.
- **Tailgating/piggybacking (proxy, since MOT has no access-authorisation signal).** A GT tailgate is an ordered pair of distinct GT IDs crossing G in the same direction within Δt (e.g. ≤ 1.5 s) or within a ground-distance gap. A predicted tailgate is the same thing on predicted events. Match pairs by both member events (each within δ) and compute **tailgate precision/recall**.
  - **Failure modes to log separately**: *ID merge* (two people end up on one track, so a tailgate is missed: FN), *ID split/fragment or switch at the gate* (one person crosses as two IDs, a phantom tailgate: FP), *missed detection under occlusion*.
  - The delayed association's contribution should show up as fewer phantom tailgates and fewer merges, binned by visibility at crossing time.
- **Datasets.** MOT17/MOT20 val halves with hand-placed gates (document the coordinates per sequence and freeze them before running trackers), plus DanceTrack val (dense look-alikes, no visibility). UNICITY or PCDS only if licences permit [UNVERIFIED].

---

## 5. Benchmark protocol

### 5.1 MOT17 half-val (the ByteTrack/OC-SORT convention)

- **Split code**, identical in `ByteTrack/tools/convert_mot17_to_coco.py`, `OC_SORT/tools/…`, `HybridSORT/tools/…`, `SparseTrack/tools/…` and `Deep-OC-SORT/data/tools/…` [VERIFIED]. With `num_images = N` and 0-indexed frame i:
  - `train_half: image_range = [0, N//2]`, i.e. 1-indexed frames **1 … N//2 + 1**;
  - `val_half: [N//2 + 1, N − 1]`, i.e. frames **N//2 + 2 … N**, which are **renumbered** to start at 1 (`anns_out[:,0] -= image_range[0]`);
  - files written are `gt/gt_val_half.txt` and `det/det_val_half.txt`;
  - only `*-FRCNN` sequences are kept ("if 'mot' in DATA_PATH and … not 'FRCNN' in seq").
  - Example: MOT17-02 (600) gives val = original frames 302–600 (299 frames). This matches BoT-SORT `GMC_files/MOT17_ablation/GMC-MOT17-02.txt` (299 lines) [VERIFIED].
  - ⚠ The halves are disjoint, but frame N//2 + 1 is in train only. There is no gap frame between the halves (adjacent frames, highly correlated) [VERIFIED/INFERENCE].
  - ⚠ BoT-SORT's ReID patch generator (`fast_reid/datasets/generate_mot_patches.py`) instead splits at `f < num_frames // 2` (0-indexed), a different boundary [VERIFIED].
- **Val sequences**: MOT17-02, 04, 05, 09, 10, 11, 13 (FRCNN) (`SparseTrack/datasets/17val_seqmap.txt`) [VERIFIED].
- **`bytetrack_ablation.pth.tar` training data**: `exps/example/mot/yolox_x_ablation.py` (`train_ann = "train.json"` in `mix_mot_ch`, `val_ann = "val_half.json"`, 800×1440, 80 epochs). The training json is built by `tools/mix_data_ablation.py` = **MOT17 `train_half.json` + CrowdHuman train + CrowdHuman val** [VERIFIED]. The paper §4.1 says the same: "first half of each video … for training and the last half for validation" [VERIFIED]. README val-half result: MOTA 76.6, IDF1 79.3, IDs 159 [VERIFIED README].
- **Generating our GT.** `scripts/make_half_split.py <MOT17/train> <out> <seqs…>` reimplements the split and writes TrackEval-ready `seq/gt/gt.txt`, a `seqinfo.ini` with `seqLength` = number of val frames, and `seqmap.txt`. I ran it for MOT17 and MOT20 (val lengths: MOT17 299/524/418/262/326/449/374; MOT20 214/1390/1202/1657). A GT-as-tracker run gives HOTA = MOTA = IDF1 = 100 [VERIFIED by running].
- **Caching detections.** Run the detector on the val-half frames using the **original** frame files (`img1/{N//2+2 … N}`) and renumber the frame ids so they match the GT [INFERENCE + VERIFIED convention].

### 5.2 MOT20 half split

- The same split code exists (`convert_mot20_to_coco.py`, identical `image_range`) [VERIFIED]. Val = MOT20-01/02/03/05 val halves (`SparseTrack/datasets/20val_seqmap.txt`) [VERIFIED].
- **A public half-trained MOT20 detector does exist.** SparseTrack README: "joint training on MOT20 train half and Crowdhuman, and evaluated on MOT20 val half. The model: yolox_x_mot20_ablation" (Google Drive link; cfg `mot20_ab_track_cfg.py` refers to `bytetrack_20_ablation.pth.tar`; training mix `tools/mix_data_ablation_20.py` = MOT20 `train_half.json` + CrowdHuman train/val) [VERIFIED]. I did not download or verify the weight file itself [UNVERIFIED].
- **The alternative used by Deep OC-SORT** is `bytetrack_x_mot17.pth.tar` (MOT17 full + CrowdHuman + CityPersons + ETHZ; no MOT20 data) on the MOT20 val half [VERIFIED `main.py` comment + ByteTrack paper §4.1 training data]. This is leak-free for MOT20 but shifts domain [INFERENCE].
- **Do NOT use** `bytetrack_x_mot20` (trained on full MOT20 train, `mix_data_test_mot20.py`) or Hybrid-SORT's `yolox_x_mix_mot20_ch_valhalf.py` on the MOT20 val half: both leak [VERIFIED training-set construction].

### 5.3 DanceTrack val

- 100 videos: 40 train, 25 val (annotations public), 35 test (annotations private; evaluated on CodaBench, `codabench.org/competitions/14885`). Data on Hugging Face `noahcao/dancetrack` (`val.zip` = 4.21 GB) [VERIFIED README + HF API].
- **Official evaluation command** (README): `python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val --METRICS HOTA CLEAR Identity --GT_FOLDER dancetrack/val --SEQMAP_FILE dancetrack/val_seqmap.txt --SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER '' --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False --TRACKERS_FOLDER val/TRACKER_NAME` [VERIFIED].
- Detector: `bytetrack_dance_model.pth.tar` (OC-SORT/Hybrid-SORT/Deep OC-SORT use it for val) [VERIFIED configs]. Its training data (DanceTrack train only?) [UNVERIFIED].
- 20 fps, IDs 0-based, visibility all 1 [VERIFIED for 0010].

### 5.4 SOMPT22

- "A Surveillance Oriented Multi-Pedestrian Tracking Dataset" (Simsek, Cigla, Kayabol; ECCV-W 2022; [arXiv:2208.02580](https://arxiv.org/abs/2208.02580)).
- 14 videos from static cameras on 6–8 m poles; "by default using 9 videos as a training set and 5 as a test set"; mostly 1920×1080 at 30 fps, short (≈ 600–1800 frames) (paper §4, Table 3) [VERIFIED].
- Train videos + GT downloadable (Google Drive folder linked from `sompt22.github.io`). Test GT is private, with a GitHub-issue-based auto-evaluation challenge [VERIFIED site].
- Licence: NC (see §0.2) [VERIFIED/UNCERTAIN]. Whether its gt has a meaningful visibility column [UNVERIFIED].
- **Good fit for the surveillance/gate use case** (elevated static cameras). Use only as an extra held-out stress set, never for training [INFERENCE].

### 5.5 TrackEval settings (`JonathonLuiten/TrackEval`, `trackeval/datasets/mot_challenge_2d_box.py::MotChallenge2DBox`)

- **Defaults** (`get_default_dataset_config`) [VERIFIED]: `CLASSES_TO_EVAL ['pedestrian']`, `BENCHMARK 'MOT17'`, `SPLIT_TO_EVAL 'train'`, `DO_PREPROC True`, `TRACKER_SUB_FOLDER 'data'`, `SEQMAP_FILE None`, `SEQ_INFO None`, `GT_LOC_FORMAT '{gt_folder}/{seq}/gt/gt.txt'`, `SKIP_SPLIT_FOL False`.
- **Preprocessing** (`get_preprocessed_seq_data`) [VERIFIED]:
  1. Tracker boxes are Hungarian-matched (IoU ≥ 0.5) to **all** GT boxes; tracker boxes matched to distractor classes `['person_on_vehicle', 'static_person', 'distractor', 'reflection']` (+ `'non_mot_vehicle'` **only if BENCHMARK == 'MOT20'**) are removed.
  2. GT kept = `class == pedestrian (1)` and `zero_marked (col 7) != 0`.
  3. Tracker classes > 1 raise an error.
  4. **No visibility filter.**
  - ⚠ Set `BENCHMARK = MOT20` for MOT20 halves, otherwise non-motorized-vehicle (class 6) matches are not removed.
- **Metrics.** `scripts/run_mot_challenge.py` defaults `METRICS ['HOTA', 'CLEAR', 'Identity']`, `THRESHOLD 0.5` [VERIFIED].
  - Identity (IDF1) `THRESHOLD 0.5`; CLEAR `THRESHOLD 0.5` [VERIFIED].
  - HOTA averages over α ∈ `np.arange(0.05, 0.99, 0.05)` (19 values) [VERIFIED `metrics/hota.py`; definition in [arXiv:2009.07736](https://arxiv.org/abs/2009.07736)].
- **Custom splits.** Either (a) the SparseTrack approach: `--SEQMAP_FILE <17val_seqmap.txt> --SKIP_SPLIT_FOL True --GT_LOC_FORMAT '{gt_folder}/{seq}/gt/gt_val_half.txt'` with `GT_FOLDER = MOT17/train` (`tools/eval_hota.sh`) [VERIFIED], or (b) our `make_half_split.py` folder (§5.1).
  - `seqLength` is read from `seqinfo.ini` (or from `SEQ_INFO {seq: length}`). Any tracker frame key > `seqLength` raises an error.
  - Option (a) keeps the full-sequence `seqLength`, which is harmless (empty trailing timesteps) but hides off-by-one frame mistakes [VERIFIED `_load_raw_file` / INFERENCE]. Prefer (b).
- **Tracker file format**: `frame, id, x, y, w, h, conf, -1, -1, -1` [VERIFIED by running].
- **Environment pitfall.** TrackEval master uses `np.float`, `np.int`, `np.bool` (81 occurrences), which NumPy ≥ 1.24 removed. `requirements.txt` pins numpy 1.18.1 [VERIFIED]. `scripts/run_trackeval.py` shims these and works with NumPy 2.5.3 / Python 3.13 [VERIFIED by running].
- **Stress analysis** (TrackEval has no built-in density or occlusion stratification) [INFERENCE]:
  - (i) cut each val half into fixed windows (e.g. 100 frames) and write each window as its own pseudo-sequence (renumbered frames + `SEQ_INFO`). Compute HOTA/IDF1/IDSW per window and regress them against window crowd density (mean GT/frame, mean pairwise IoU) and mean visibility. Caveat: IDs reset at window borders, so IDSW across a border is not counted.
  - (ii) count per-GT-box ID switches and track-coverage binned by GT visibility, using the CLEAR matching (re-implement the per-timestep matching from `metrics/clear.py`).

### 5.6 Leakage traps (summary)

1. **ReID weights.** BoT-SORT `mot17_sbs_S50.pth` / `mot20_sbs_S50.pth` are flagged by the Deep OC-SORT code as "trained over the half-val we evaluate on" [VERIFIED docstring]. Yet BoT-SORT's own patch generator splits train/test at `num_frames // 2` [VERIFIED]. [UNCERTAIN] which split the *released* weights saw, so treat them as leaked.
   - For our cached embeddings, use a ReID model trained without MOT17/MOT20 val frames. The Deep OC-SORT choice was OSNet-AIN `ms_d_c` (licence concerns in §1.6).
2. **Detectors.** Only `bytetrack_ablation` (MOT17), the SparseTrack MOT20-ablation model, or the MOT17-full model (for MOT20) are clean for the val halves. `bytetrack_x_mot17` on MOT17 val leaks: its exp `yolox_x_mix_det.py` trains on `mix_det/annotations/train.json`, which `tools/mix_data_test_mot17.py` builds from **full** MOT17 `train.json` + CrowdHuman + ETHZ + CityPersons [VERIFIED].
3. **Per-sequence hand-tuned thresholds** (ByteTrack/BoT-SORT `--default-parameters`) [VERIFIED]: disable them.
4. **Interpolation tuned on GT** (`interpolation.py` grid search) [VERIFIED]: exclude it.
5. **Visibility estimator fed GT boxes** (§2.1): circular.
6. **Gate placement after looking at results**: freeze the gates first.
7. **Adjacent half boundary**: the frames either side of the split are near-duplicates. Do not train any learned gate or ReID on train-half frames within a margin of the boundary if you want strict independence [INFERENCE].

### 5.7 Latency / FPS reporting

- **Papers.** ByteTrack: "FPS is measured with FP16-precision and batch size of 1 on a single GPU" (V100), detector included; for example, the Table 2 FPS column includes the YOLOX-X detector [VERIFIED §4.1]. Hybrid-SORT: same, plus "single V100 … Intel Xeon Silver 4214R" [VERIFIED]. BoT-SORT code prints tracker-only FPS separately [VERIFIED].
- **Our CPU protocol** [INFERENCE]:
  - Measure **tracker-only** time per frame on cached dets/embeddings (exclude disk I/O and the detector) with `time.perf_counter_ns`, after a warm-up. Pin threads (`OMP_NUM_THREADS = 1`, also `MKL`/`OPENBLAS`), report the CPU model and Python/NumPy versions.
  - Report the median, p95, p99 and max ms/frame, and plot latency against detections per frame (MOT20 frames reach hundreds of GT boxes; e.g. MOT20-05 averages 226.6 boxes/frame per the MOT20 page [VERIFIED]).
  - Report the **algorithmic latency K/fps** separately from the compute latency. End-to-end = detector + ReID + tracker + K/fps.
  - Count how often the delayed branch is triggered and how many hypotheses are alive (mean/max) against density, to show bounded behaviour.
  - If CMC is used, either the cached VidStab files (BoT-SORT / Deep OC-SORT caches) with "CMC cost excluded" stated, or the measured `sparseOptFlow` time at downscale 2.

---

## 6. Implementation pitfalls (checklist)

1. **Cost vs similarity thresholds.** ByteTrack `match_thresh = 0.9` means IoU > 0.1, while OC-SORT `iou_thresh = 0.3` means IoU ≥ 0.3. Do not port numbers across families blindly [VERIFIED].
2. **`fuse_score` is on for MOT17 and off with `--mot20`** (ByteTrack, BoT-SORT, SparseTrack) [VERIFIED]. If you change the flag between datasets, document it.
3. **Strict inequalities.** A det with score exactly equal to `track_thresh` falls in neither the high nor the low set (ByteTrack/OC-SORT) [VERIFIED]. Cached detections stored with rounding (e.g. 0.6000) hit this case.
4. **New-track threshold = high + 0.1** (ByteTrack `det_thresh`, BoT-SORT `new_track_thresh`) [VERIFIED].
5. **Frame-rate scaling.** ByteTrack's `buffer_size = frame_rate/30 · track_buffer`, but the evaluator never passes the rate. MOT17-05 is 14 fps, MOT17-13 25 fps, DanceTrack 20 fps [VERIFIED]. Decide one rule, preferably time-based buffers, and apply it to every baseline.
6. **Output conventions differ.** ByteTrack outputs KF-updated boxes of activated tracks. OC-SORT outputs the *last observation* and only for tracks updated this frame after 3 hits [VERIFIED]. Both affect MOTA/HOTA independently of association quality.
7. **Post-filters.** `w/h > 1.6` removal and `min_box_area` (100 in ByteTrack/OC-SORT, 10 in BoT-SORT/Deep OC-SORT) [VERIFIED]. Apply one filter to all methods.
8. **Per-sequence threshold hacks** in `mot_evaluator.py` / `--default-parameters` [VERIFIED]: off.
9. **Interpolation / `--post` is offline** [VERIFIED].
10. **GMC "file" mode** is sequential and per-sequence (`applyFile` readline). The tracker must be reset with the file at frame 1, with no skipped frames [VERIFIED].
11. **HMIoU's HIoU factor is unclipped.** It is harmless only because it is multiplied by IoU (0 when disjoint). If you reuse HIoU alone, for example as a separate cost term, clip it to [0, 1] [VERIFIED code / INFERENCE].
12. **ORU interpolates (x, y, w, h)**, not the KF's (s, r) [VERIFIED]. Keep it consistent if you re-implement.
13. **TrackEval + NumPy ≥ 1.24** needs the np-alias shim or a fork [VERIFIED].
14. **MOT20 must be evaluated with `BENCHMARK = MOT20`** so class 6 counts as a distractor [VERIFIED].
15. **GT includes vis = 0 boxes** (≈ 11% of the MOT17 val half on our oracle). A no-coasting tracker caps at MOTA ≈ 89 there [VERIFIED by running]. Report this ceiling next to results.
16. **DanceTrack visibility is constant 1**, so visibility scoring there is impossible [VERIFIED for one sequence; assume all].
17. **Licences.** Nothing from boxmot or Ultralytics (AGPL). No `BoT-SORT/yolov7` (GPL). MOT/DanceTrack/SOMPT22 data and the derived weights are NC: benchmark only [VERIFIED / INFERENCE].
18. **Coordinate base.** MOT files are 1-based (MOT16/MOT20 papers) [VERIFIED]. The border-cropping term in the visibility estimate must use the same base. `scripts/vis_oracle_check.py` uses `x − 1`.

---

## 7. Unverified / open items (what I checked)

- CrowdHuman licence: `crowdhuman.org` and `/download.html` fetched; no licence text found [UNVERIFIED].
- UNICITY and PCDS licences: Zenodo page and GitHub README fetched; no licence text found [UNVERIFIED].
- SparseTrack MOT20-ablation weight file: link seen in the README; not downloaded or hashed [UNVERIFIED].
- Training data of `bytetrack_dance_model.pth.tar` and of Hybrid-SORT's `ocsort_mot17_ablation.pth.tar` (and whether the latter is identical to `bytetrack_ablation`) [UNVERIFIED].
- Released BoT-SORT SBS ReID weights' training split [UNCERTAIN, see §5.6].
- Formal MOTChallenge definition of "online" [UNVERIFIED; site offline, IJCV paper only has an "ON" column].
- Reid 1979 full text not read (metadata verified via Crossref). Exact MHT-bLSTM parameters not read.
- The visibility oracle numbers cover only the first 150 frames of 4 sequences. Run `scripts/vis_oracle_check.py` with full frames before quoting them as dataset-level.
- SOMPT22 gt column semantics (visibility present?) [UNVERIFIED]; not downloaded (Google Drive).
- BoostTrack++ content [UNVERIFIED beyond the title].

## 8. Artefacts produced in the box

- `/workspace/zed-ps2/src/*`: shallow clones of every repo cited.
- `/workspace/zed-ps2/papers/*.txt`: text of the papers cited (arXiv/CVF).
- `/workspace/zed-ps2/data/MOT17Labels`, `MOT20Labels`: official label zips (gt + det + seqinfo; no images).
- `/workspace/zed-ps2/data/mot17_val_half`, `mot20_val_half`: TrackEval-ready val-half GT + `seqmap.txt`.
- `/workspace/zed-ps2/scripts/make_half_split.py`: ByteTrack-convention half split.
- `/workspace/zed-ps2/scripts/run_trackeval.py`: TrackEval runner with the NumPy shim (HOTA/CLEAR/Identity).
- `/workspace/zed-ps2/scripts/vis_oracle_check.py`: GT-box visibility reproduction (bottom-edge depth + border crop).
- Python venv with numpy/scipy: `/workspace/zed-ps2/.venv`.

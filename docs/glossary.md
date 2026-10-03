# Glossary

| Term | Meaning here |
|---|---|
| **AssA / DetA** | HOTA's association and detection accuracy components; HOTA = √(DetA·AssA) per IoU threshold, averaged |
| **BENCHMARK** | TrackEval option; `MOT20` adds class 6 (non-MOT vehicle) to the distractor classes |
| **BoT-SORT** | ByteTrack + camera-motion compensation + xywh Kalman + IoU/re-ID min-fusion (Aharon et al., 2022) |
| **ByteTrack / BYTE** | tracker that associates high-score detections first, then uses low-score ones to continue existing tracks (Zhang et al., 2021) |
| **cache** | Stage A output: one `.npz` per sequence with boxes, scores, embeddings and metadata (schema v1) |
| **CLEAR** | MOTA/MOTP family; per-frame matching at IoU ≥ 0.5 |
| **coasting** | outputting a predicted box for a track that has no detection this frame |
| **conf floor** | lowest detection score kept in Stage A (0.05) |
| **confirmed / tentative / lost / removed** | track lifecycle states (see [algorithm.md](algorithm.md#5-track-lifecycle-fsm)) |
| **delayed (ambiguity-aware) assignment** | planned: hold only ambiguous decisions open for ≤ K frames, then commit; output lags by K frames |
| **density bin** | frames grouped by gt people per frame (quartile edges) |
| **distractor** | gt classes whose matched tracker boxes are ignored (person on vehicle, static person, distractor, reflection; + non-MOT vehicle in MOT20) |
| **DGX** | NVIDIA server Panshul uses for Stage A |
| **fuse_score** | ByteTrack's `1 − (1 − c_iou)·score` cost; off for MOT20 |
| **gate event** | a person's bottom-centre point crossing a virtual entry line (planned, phase 2b) |
| **HOTA** | Higher Order Tracking Accuracy (Luiten et al., 2021) |
| **IDF1** | F1 of identity-consistent matches under one global gt↔track ID bijection |
| **IDSW** | identity switch: a gt person matched to a different track id than last time |
| **IDTP / IDFP / IDFN** | Identity metric counts behind IDF1 |
| **K (lag)** | maximum frames a decision may stay open in the delayed variant; K = 0 is strictly online |
| **leakage** | evaluating on frames the detector, re-ID model or tuning has seen |
| **MOTA** | 1 − (FN + FP + IDSW) / GT |
| **MSMT17** | person re-ID dataset used to train our OSNet weights (research-only licence) |
| **OSNet-AIN** | Omni-Scale Network with instance normalisation; our re-ID model (torchreid) |
| **pending** | per-frame list of open (deferred) decisions in the JSON export; empty in phase 1 |
| **public detections** | MOTChallenge-provided `det.txt` (FRCNN for MOT17); phase-1 pipeline check only |
| **SAW_EVAL_HALF** | label for caches from a detector trained on the evaluation half (MOT20 full model) |
| **Stage A / Stage B** | GPU detection + embedding pass / CPU tracking + evaluation |
| **stress bins** | errors broken down by gt visibility and by people per frame |
| **tailgating** | two people passing a gate on one validation; proxied by two crossings within a short gap |
| **TrackEval** | the official HOTA/CLEAR/Identity implementation (vendored, MIT) |
| **val half** | frames N//2+2..N of each train sequence, renumbered from 1 |
| **visibility** | fraction of a person not occluded or outside the image (gt column 9; planned estimate from boxes) |
| **vis-0 cap** | the score ceiling when invisible gt boxes are never output (MOT17 val-half: IDF1 94.2 / HOTA 91.1 / MOTA 89.1) |
| **YOLOX** | anchor-free detector (Apache-2.0) used by ByteTrack and the other compared trackers |
| **ZIG** | Zed Digital's hands-free ticketing/validation platform; the gate scene this tracker targets |

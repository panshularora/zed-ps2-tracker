<!-- run-info -->
> Run: profile **laptop**, GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU (6144 MiB); detector yolox_x (ByteTrack) [bytetrack_ablation.pth.tar], fp16, input 800x1440; cache `MOT17/ablation/train`.
<!-- /run-info -->

# ByteTrack reproduction check (MOT17 val-half, YOLOX-X bytetrack_ablation)

Reference: ByteTrack arXiv 2110.06864v3, Table 1 (IoU|IoU) / Table 11 (Interval: No). Tolerance fixed in `configs/phase3/plan.yaml` before any GPU data existed.

| Metric | ByteTrack paper | ours (ByteTrackLike, IoU) | Δ | tolerance |
|---|---|---|---|---|
| MOTA | 76.6 | 66.4 | -10.2 | ±1.0 |
| IDF1 | 79.3 | 73.8 | -5.5 | ±1.5 |
| IDSW | 159 | 165.0 | +6.0 | (reported only) |
| FP | 3358 | 5379.0 | +2021.0 | (reported only) |
| FN | 9081 | 12552.0 | +3471.0 | (reported only) |

**Verdict: NOT REPRODUCED.**

Known differences that can move the numbers: xywh Kalman filter instead of xyah; TrackEval CLEAR instead of py-motmetrics; the common output filter (area > 100, w/h ≤ 1.6, as ByteTrack's evaluator); no per-sequence thresholds (ByteTrack's val-half run has none either).

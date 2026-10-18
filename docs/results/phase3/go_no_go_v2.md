<!-- run-info -->
> Run: profile **laptop**, GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU (6144 MiB); detector yolox_x (ByteTrack) [bytetrack_ablation.pth.tar], fp16, input 800x1440; cache `MOT17/ablation/train`.
<!-- /run-info -->

# Go/no-go v2 (sanity check, not a results table)

Pre-registered in `docs/preregistration/go_no_go_v2.md`; verdict computed by `eval/phase3.py` from the frozen rule.

| Arm | HOTA | IDF1 | IDSW | flagged frames | p95 ms |
|---|---|---|---|---|---|
| k0 | 62.5 | 73.0 | 183 | 0.0 % | 5.36 |
| k3 | 62.2 | 72.2 | 196 | 50.8 % | 116.69 |
| k5 | 61.9 | 71.9 | 198 | 50.7 % | 170.09 |

| Criterion | pass |
|---|---|
| 1_idsw | no |
| 2_identity | no |
| 3_selectivity | no |
| 4_compute | no |
| 5_replication | no |

MOT20 replication: {'status': 'fail', 'cache': 'MOT20/sparsetrack_ablation/train', 'idsw_k0': 1462.0, 'idsw_k5': 1575.0}

**Verdict: NO-GO**

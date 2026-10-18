<!-- run-info -->
> Run: profile **laptop**, GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU (6144 MiB); detector yolox_x (ByteTrack) [bytetrack_ablation.pth.tar], fp16, input 800x1440; cache `MOT17/ablation/train`.
<!-- /run-info -->

### Ablation: MOT17 val-half, YOLOX-X bytetrack_ablation + OSNet-AIN re-ID

| Group | Tracker | HOTA | IDF1 | MOTA | IDSW | AssA | DetA | p50 / p95 / max ms |
|---|---|---|---|---|---|---|---|---|
| baseline | ByteTrackLike (IoU) | 63.4 | 73.8 | 66.4 | 165 | 66.5 | 61.1 | 1.04 / 2.63 / 12.3 |
| baseline | OC-SORT | 61.8 | 72.3 | 64.0 | 204 | 65.7 | 58.6 | 0.79 / 2.48 / 15.3 |
| baseline | Deep OC-SORT (emb + CMC) | 63.3 | 74.6 | 64.6 | 125 | 68.2 | 59.2 | 1.39 / 4.60 / 15.1 |
| baseline | Hybrid-SORT | 62.4 | 72.2 | 64.2 | 240 | 65.4 | 60.2 | 1.66 / 4.75 / 15.6 |
| baseline | BoT-SORT (CMC, IoU) | 64.6 | 76.3 | 67.1 | 135 | 68.6 | 61.5 | 1.08 / 2.80 / 11.9 |
| baseline | ByteTrackLike + re-ID | 62.5 | 73.0 | 66.5 | 183 | 64.9 | 60.8 | 1.30 / 3.64 / 13.1 |
| baseline | BoT-SORT (CMC) + re-ID | 63.4 | 74.3 | 66.9 | 193 | 66.3 | 61.4 | 1.39 / 3.75 / 13.5 |
| control | ByteTrackLike + re-ID, buffer 90 (control) | 62.3 | 73.1 | 66.6 | 172 | 64.8 | 60.7 | 1.36 / 3.63 / 13.6 |
| ours | + re-ID + visibility (buffer) | 62.3 | 73.1 | 66.6 | 172 | 64.8 | 60.7 | 1.58 / 4.68 / 15.0 |
| ours | + re-ID + appearance update gate (min_visibility) | 62.0 | 72.6 | 66.2 | 201 | 64.0 | 60.8 | 1.44 / 4.52 / 14.9 |
| ours | + re-ID + appearance cost gate (cost_min_visibility) | 63.2 | 73.3 | 66.3 | 174 | 66.2 | 61.0 | 1.50 / 4.64 / 12.9 |
| ours | + re-ID + visibility-gated appearance (full) | 63.2 | 73.9 | 66.2 | 167 | 66.2 | 61.0 | 1.57 / 4.63 / 13.6 |
| control | BoT-SORT + re-ID, buffer 90 (control) | 63.3 | 74.4 | 66.9 | 183 | 66.1 | 61.4 | 1.43 / 3.71 / 13.1 |
| ours | BoT-SORT + re-ID + visibility-gated appearance | 63.9 | 75.1 | 66.8 | 156 | 67.1 | 61.6 | 1.60 / 4.72 / 40.7 |

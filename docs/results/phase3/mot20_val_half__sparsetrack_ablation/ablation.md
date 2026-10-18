<!-- run-info -->
> Run: profile **laptop**, GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU (6144 MiB); detector yolox_x (ByteTrack) [sparsetrack_mot20_ablation.pth.tar], fp16, input 896x1600; cache `MOT20/sparsetrack_ablation/train`.
<!-- /run-info -->

### Ablation: MOT20 val-half, sparsetrack_ablation + OSNet-AIN re-ID

| Group | Tracker | HOTA | IDF1 | MOTA | IDSW | AssA | DetA | p50 / p95 / max ms |
|---|---|---|---|---|---|---|---|---|
| baseline | ByteTrackLike (IoU) | 69.4 | 84.1 | 87.9 | 1011 | 66.8 | 72.4 | 9.65 / 13.66 / 85.9 |
| baseline | OC-SORT | 68.0 | 82.7 | 86.7 | 1293 | 64.9 | 71.5 | 11.18 / 16.78 / 375.9 |
| baseline | Deep OC-SORT (emb + CMC) | 68.7 | 83.8 | 86.8 | 1171 | 66.1 | 71.6 | 25.44 / 54.95 / 228.5 |
| baseline | Hybrid-SORT | 68.8 | 83.4 | 87.2 | 1365 | 66.0 | 71.9 | 20.42 / 29.62 / 115.6 |
| baseline | BoT-SORT (CMC, IoU) | 69.3 | 83.9 | 87.9 | 1162 | 66.5 | 72.4 | 9.96 / 13.51 / 90.7 |
| baseline | ByteTrackLike + re-ID | 67.5 | 81.1 | 87.7 | 1278 | 63.5 | 72.0 | 21.03 / 57.38 / 228.1 |
| baseline | BoT-SORT (CMC) + re-ID | 67.6 | 81.2 | 87.8 | 1429 | 63.6 | 72.1 | 30.45 / 65.31 / 223.5 |
| control | ByteTrackLike + re-ID, buffer 90 (control) | 67.4 | 81.2 | 87.7 | 1272 | 63.2 | 72.0 | 28.36 / 57.88 / 224.0 |
| ours | + re-ID + visibility-gated appearance (full) | 69.5 | 84.5 | 87.9 | 1019 | 66.9 | 72.4 | 50.60 / 95.16 / 245.6 |
| ours | BoT-SORT + re-ID + visibility-gated appearance | 69.2 | 84.0 | 87.9 | 1162 | 66.3 | 72.4 | 48.45 / 87.48 / 179.0 |

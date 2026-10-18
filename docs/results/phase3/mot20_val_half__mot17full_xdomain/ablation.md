<!-- run-info -->
> Run: profile **laptop**, GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU (6144 MiB); detector yolox_x (ByteTrack) [bytetrack_x_mot17.pth.tar], fp16, input 896x1600; cache `MOT20/mot17full_xdomain/train`.
<!-- /run-info -->

### Ablation: MOT20 val-half, mot17full_xdomain + OSNet-AIN re-ID

| Group | Tracker | HOTA | IDF1 | MOTA | IDSW | AssA | DetA | p50 / p95 / max ms |
|---|---|---|---|---|---|---|---|---|
| baseline | ByteTrackLike (IoU) | 60.8 | 76.8 | 79.6 | 1070 | 58.6 | 63.3 | 10.71 / 21.45 / 1211.8 |
| baseline | OC-SORT | 59.8 | 77.0 | 76.4 | 1286 | 58.8 | 61.1 | 12.57 / 25.66 / 1260.1 |
| baseline | Deep OC-SORT (emb + CMC) | 60.4 | 78.0 | 76.4 | 1227 | 59.8 | 61.2 | 22.99 / 84.46 / 308.6 |
| baseline | Hybrid-SORT | 62.9 | 79.9 | 82.6 | 1471 | 60.6 | 65.5 | 22.52 / 44.74 / 2057.4 |
| baseline | BoT-SORT (CMC, IoU) | 60.9 | 77.1 | 79.8 | 1248 | 58.8 | 63.4 | 9.09 / 14.55 / 119.1 |
| baseline | ByteTrackLike + re-ID | 60.3 | 75.9 | 79.6 | 1160 | 57.7 | 63.2 | 30.96 / 74.85 / 227.8 |
| baseline | BoT-SORT (CMC) + re-ID | 60.4 | 76.0 | 79.7 | 1367 | 57.9 | 63.3 | 31.60 / 72.95 / 217.2 |
| control | ByteTrackLike + re-ID, buffer 90 (control) | 60.5 | 76.7 | 79.8 | 1118 | 58.1 | 63.3 | 30.17 / 71.54 / 210.3 |
| ours | + re-ID + visibility-gated appearance (full) | 60.9 | 77.5 | 79.8 | 1039 | 58.8 | 63.4 | 56.17 / 106.96 / 229.9 |
| ours | BoT-SORT + re-ID + visibility-gated appearance | 61.1 | 77.7 | 80.0 | 1193 | 58.9 | 63.5 | 57.36 / 107.89 / 273.0 |

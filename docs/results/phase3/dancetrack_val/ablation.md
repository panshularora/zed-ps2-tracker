<!-- run-info -->
> Run: profile **laptop**, GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU (6144 MiB); detector yolox_x (ByteTrack) [dancetrack_bytetrack_x.pth.tar], fp16, input 800x1440; cache `DanceTrack/dancetrack/val`.
<!-- /run-info -->

### Ablation: DanceTrack val, DanceTrack YOLOX-X + OSNet-AIN re-ID (MOT17 tracker settings)

| Group | Tracker | HOTA | IDF1 | MOTA | IDSW | AssA | DetA | p50 / p95 / max ms |
|---|---|---|---|---|---|---|---|---|
| baseline | ByteTrackLike (IoU) | 52.5 | 54.9 | 88.1 | 1775 | 36.3 | 76.1 | 1.62 / 3.78 / 51.6 |
| baseline | OC-SORT | 51.2 | 51.3 | 85.1 | 1871 | 34.8 | 75.6 | 1.38 / 3.21 / 39.1 |
| baseline | Deep OC-SORT (emb) | 55.0 | 55.6 | 85.2 | 1738 | 40.2 | 75.6 | 2.39 / 6.05 / 53.4 |
| baseline | Hybrid-SORT | 57.4 | 58.5 | 87.3 | 1667 | 42.9 | 77.3 | 2.47 / 6.12 / 60.7 |
| baseline | BoT-SORT (IoU) | 52.3 | 53.2 | 88.1 | 1934 | 35.9 | 76.6 | 1.77 / 4.15 / 44.3 |
| baseline | ByteTrackLike + re-ID | 53.3 | 54.0 | 87.8 | 1883 | 37.6 | 76.1 | 2.16 / 4.64 / 41.8 |
| baseline | BoT-SORT + re-ID | 53.4 | 53.8 | 87.7 | 2127 | 37.5 | 76.5 | 1.75 / 3.38 / 34.1 |
| control | ByteTrackLike + re-ID, buffer 90 (control) | 54.3 | 55.0 | 87.8 | 1887 | 38.9 | 76.0 | 1.60 / 3.22 / 29.6 |
| ours | + re-ID + visibility-gated appearance (full) | 53.4 | 55.3 | 88.2 | 1681 | 37.7 | 75.9 | 1.75 / 3.78 / 28.9 |
| ours | BoT-SORT + re-ID + visibility-gated appearance | 54.3 | 55.8 | 88.1 | 1897 | 38.8 | 76.6 | 1.77 / 3.51 / 24.4 |

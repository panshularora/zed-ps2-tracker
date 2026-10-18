<!-- run-info -->
> Run: profile **laptop**, GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU (6144 MiB); detector bytetrack_ablation.pth.tar; bytetrack_x_mot17.pth.tar; dancetrack_bytetrack_x.pth.tar; sparsetrack_mot20_ablation.pth.tar, see detectors, input see detectors.
<!-- /run-info -->

### Ablation: MOT20 val-half, mot20full_SAW_EVAL_HALF + OSNet-AIN re-ID (DETECTOR SAW THE EVAL HALF: not a clean result)

| Group | Tracker | HOTA | IDF1 | MOTA | IDSW | AssA | DetA | p50 / p95 / max ms |
|---|---|---|---|---|---|---|---|---|
| baseline | ByteTrackLike (IoU) | 77.9 | 90.0 | 93.3 | 643 | 75.4 | 80.5 | 9.83 / 15.22 / 96.9 |
| baseline | OC-SORT | 77.2 | 89.1 | 93.0 | 833 | 74.4 | 80.3 | 10.57 / 15.74 / 112.9 |
| baseline | Deep OC-SORT (emb + CMC) | 77.3 | 89.2 | 93.0 | 790 | 74.5 | 80.3 | 33.18 / 58.70 / 170.4 |
| baseline | Hybrid-SORT | 77.4 | 89.6 | 92.7 | 954 | 74.9 | 80.0 | 19.33 / 28.85 / 120.8 |
| baseline | BoT-SORT (CMC, IoU) | 78.1 | 90.3 | 93.4 | 707 | 75.7 | 80.6 | 9.90 / 13.42 / 90.1 |
| baseline | ByteTrackLike + re-ID | 77.0 | 88.9 | 93.2 | 877 | 74.0 | 80.3 | 22.11 / 66.35 / 207.2 |
| baseline | BoT-SORT (CMC) + re-ID | 76.7 | 88.3 | 93.3 | 967 | 73.3 | 80.3 | 29.29 / 59.12 / 208.2 |
| control | ByteTrackLike + re-ID, buffer 90 (control) | 77.1 | 89.1 | 93.2 | 885 | 74.1 | 80.3 | 33.56 / 69.85 / 210.0 |
| ours | + re-ID + visibility-gated appearance (full) | 77.8 | 90.1 | 93.3 | 659 | 75.2 | 80.5 | 51.50 / 86.94 / 232.4 |
| ours | BoT-SORT + re-ID + visibility-gated appearance | 78.1 | 90.4 | 93.4 | 721 | 75.7 | 80.6 | 51.56 / 84.31 / 288.3 |

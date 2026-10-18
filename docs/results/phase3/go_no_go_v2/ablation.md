<!-- run-info -->
> Run: profile **laptop**, GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU (6144 MiB); detector yolox_x (ByteTrack) [bytetrack_ablation.pth.tar], fp16, input 800x1440; cache `MOT17/ablation/train`.
<!-- /run-info -->

### Ablation: go/no-go v2 (sanity check): delayed decision with re-ID, MOT17 val-half

| Group | Tracker | HOTA | IDF1 | MOTA | IDSW | AssA | DetA | p50 / p95 / max ms |
|---|---|---|---|---|---|---|---|---|
| sanity check | k0 | 62.5 | 73.0 | 66.5 | 183 | 64.9 | 60.8 | 1.74 / 5.36 / 17.6 |
| sanity check | k3 | 62.2 | 72.2 | 66.5 | 196 | 64.5 | 60.7 | 5.10 / 116.69 / 323.9 |
| sanity check | k5 | 61.9 | 71.9 | 66.6 | 198 | 63.9 | 60.7 | 4.52 / 170.09 / 521.2 |

<!-- run-info -->
> Run: profile **laptop**, GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU (6144 MiB); detector yolox_x (ByteTrack) [bytetrack_ablation.pth.tar], fp16, input 800x1440; cache `MOT17/ablation/train`.
<!-- /run-info -->

| Tracker | gt / trk crossings | count error % | event P / R % | attribution error % | tailgates gt / trk | tailgate P / R % |
|---|---|---|---|---|---|---|
| gt as tracker (check) | 25 / 25 | 0.0 | 100.0 / 100.0 | 0.0 | 9 / 9 | 100.0 / 100.0 |
| botsort | 25 / 28 | 12.0 | 67.9 / 76.0 | 31.6 | 9 / 7 | 85.7 / 66.7 |
| botsort_reid | 25 / 31 | 24.0 | 64.5 / 80.0 | 40.0 | 9 / 9 | 77.8 / 77.8 |
| botsort_reid_buffer90 | 25 / 36 | 44.0 | 58.3 / 84.0 | 38.1 | 9 / 12 | 66.7 / 88.9 |
| botsort_reid_vis_gated | 25 / 32 | 28.0 | 62.5 / 80.0 | 25.0 | 9 / 9 | 77.8 / 77.8 |
| bytetrack | 25 / 28 | 12.0 | 67.9 / 76.0 | 26.3 | 9 / 7 | 85.7 / 66.7 |
| bytetrack_reid | 25 / 30 | 20.0 | 66.7 / 80.0 | 40.0 | 9 / 10 | 80.0 / 88.9 |
| deep_ocsort | 25 / 25 | 0.0 | 72.0 / 72.0 | 16.7 | 9 / 5 | 100.0 / 55.6 |
| hybrid_sort | 25 / 33 | 32.0 | 60.6 / 80.0 | 35.0 | 9 / 14 | 50.0 / 77.8 |
| ocsort | 25 / 25 | 0.0 | 72.0 / 72.0 | 22.2 | 9 / 5 | 100.0 / 55.6 |
| reid_buffer90 | 25 / 35 | 40.0 | 62.9 / 88.0 | 40.9 | 9 / 12 | 75.0 / 100.0 |
| reid_costgate | 25 / 29 | 16.0 | 65.5 / 76.0 | 26.3 | 9 / 8 | 75.0 / 66.7 |
| reid_minvis | 25 / 33 | 32.0 | 63.6 / 84.0 | 38.1 | 9 / 12 | 66.7 / 88.9 |
| reid_vis | 25 / 35 | 40.0 | 62.9 / 88.0 | 40.9 | 9 / 12 | 75.0 / 100.0 |
| reid_vis_gated | 25 / 37 | 48.0 | 56.8 / 84.0 | 23.8 | 9 / 11 | 72.7 / 88.9 |

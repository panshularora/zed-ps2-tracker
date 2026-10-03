### MOT17 val_half, public FRCNN detections

| Estimate | n | MAE | RMSE | bias | Spearman ρ | Kendall τ-b | ECE | AUROC v<0.5 | AP v<0.5 | AUROC v<0.25 | AP v<0.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ours (geometry × score), on detector boxes** | 27176 | 0.074 | 0.140 | 0.034 | 0.613 | 0.521 | 0.034 | 0.938 | 0.460 | 0.952 | 0.475 |
| ours, geometry only | 27176 | 0.073 | 0.142 | 0.052 | 0.550 | 0.488 | 0.052 | 0.806 | 0.388 | 0.851 | 0.501 |
| detector score as visibility (naive) | 27176 | 0.107 | 0.175 | 0.092 | 0.429 | 0.366 | 0.092 | 0.947 | 0.293 | 0.931 | 0.071 |
| constant (mean gt vis of matched) | 27176 | 0.129 | 0.160 | 0.000 | – | – | 0.109 | 0.500 | 0.031 | 0.500 | 0.028 |
| gt-box oracle, pedestrians only (same boxes) | 27176 | 0.032 | 0.079 | 0.029 | 0.824 | 0.764 | 0.029 | 0.941 | 0.831 | 0.975 | 0.920 |
| gt-box oracle, pedestrians only (all gt) | 53890 | 0.060 | 0.159 | 0.056 | 0.905 | 0.839 | 0.056 | 0.938 | 0.955 | 0.964 | 0.956 |
| gt-box oracle, all classes (implementation check) | 53890 | 0.023 | 0.066 | 0.017 | 0.984 | 0.943 | 0.017 | 0.990 | 0.990 | 0.990 | 0.984 |

Calibration of ours, geometry only (by gt-visibility bin):

| gt bin | n | mean v̂ | mean v |
|---|---|---|---|
| [0.00, 0.25) | 159 | 0.431 | 0.069 |
| [0.25, 0.50) | 378 | 0.681 | 0.420 |
| [0.50, 0.75) | 5137 | 0.831 | 0.652 |
| [0.75, 1.00) | 21502 | 0.968 | 0.952 |

Coverage: fraction of gt pedestrians matched by a detector box (IoU ≥ 0.5):

| gt bin | gt boxes | matched |
|---|---|---|
| [0.00, 0.25) | 13794 | 0.012 |
| [0.25, 0.50) | 9550 | 0.040 |
| [0.50, 0.75) | 8198 | 0.627 |
| [0.75, 1.00) | 22348 | 0.962 |

Stratified (ours, geometry only):

| Stratum | n | MAE | Spearman ρ | AUROC v<0.5 |
|---|---|---|---|---|
| static | 16738 | 0.066 | 0.571 | 0.900 |
| moving | 10438 | 0.084 | 0.517 | 0.753 |
| dets/frame <= 12 | 9363 | 0.081 | 0.583 | 0.781 |
| 12 < dets/frame <= 23 | 10544 | 0.063 | 0.574 | 0.851 |
| dets/frame > 23 | 7269 | 0.078 | 0.461 | 0.805 |
| MOT17-02 | 3236 | 0.046 | 0.789 | 0.925 |
| MOT17-04 | 11983 | 0.073 | 0.490 | 0.815 |
| MOT17-05 | 1912 | 0.107 | 0.534 | 0.806 |
| MOT17-09 | 1519 | 0.054 | 0.651 | 0.758 |
| MOT17-10 | 3645 | 0.072 | 0.429 | 0.714 |
| MOT17-11 | 2751 | 0.054 | 0.707 | 0.855 |
| MOT17-13 | 2130 | 0.123 | 0.374 | 0.640 |

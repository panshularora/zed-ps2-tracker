# Delayed decision: variant selection on MOT17 train_half (public FRCNN)

Selection only; not a result. Measured with `eval.run`, `delayed.*` overrides on `configs/baselines/bytetrack.yaml`.

| Variant | HOTA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|
| baseline (K=0) | 52.67 | 59.48 | 51.39 | 299 |
| K=5, margin 0.05 | 51.85 | 58.31 | 51.27 | 361 |
| K=5, margin 0.02 | 51.85 | 58.31 | 51.30 | 355 |
| K=5, min_gain 0.1 | 51.89 | 58.48 | 51.32 | 341 |
| K=5, min_gain 0.3 | 51.99 | 58.48 | 51.34 | 331 |
| K=5, min_gain 0.6 | 52.03 | 58.86 | 51.40 | 312 |
| K=5, margin 0.02, min_gain 0.6 | 52.03 | 58.87 | 51.41 | 313 |
| K=5, confirmed_only | 52.81 | 59.74 | 51.33 | 315 |
| K=5, same_tracks | 52.12 | 58.93 | 51.38 | 307 |
| K=5, confirmed_only + same_tracks | 52.59 | 59.41 | 51.36 | 325 |

## v2: look-ahead scoring, MOT17 train_half, corrupted gt (seed 100, not used in the go/no-go), K = 3

All rows use `configs/ours/delayed.yaml` v1 (confirmed_only) plus the listed overrides.

| Variant | HOTA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|
| K = 0 (baseline) | 57.9 | 69.9 | 65.2 | 363 |
| score over cluster tracks (v1) | 48.0 | 57.0 | 64.1 | 820 |
| score over all first-association tracks | 55.4 | 66.2 | 65.2 | 482 |
| all + birth penalty 0.5 | 55.0 | 65.9 | 65.1 | 480 |
| all + birth penalty 1.0 | 55.1 | 66.1 | 64.9 | 492 |
| **all + birth penalty 1.0 + same_tracks (v2, frozen)** | 56.1 | 67.8 | 64.9 | 424 |

Scoring only the cluster's tracks was biased. A hypothesis that gives a cluster track a detection
belonging to a track outside the cluster paid nothing for the damage it did to that other track.
Scoring every track removes most of the harm, but no variant beats K = 0.

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

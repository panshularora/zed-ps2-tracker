# scripts/research

Research Lab scripts kept as-is for provenance (see `docs/research/implementation_research.md`).
They are not part of the package and are excluded from lint.

* `vis_oracle_check.py`: recomputes MOT gt visibility from gt boxes alone (bottom-edge depth
  order + image-border crop) and compares it with gt column 9. This is the basis of the planned
  depth-order visibility estimator (phase 2) and its two reference baselines.
  `python scripts/research/vis_oracle_check.py <max_frames> data/MOT17/train/MOT17-02-FRCNN ...`

The half-split and TrackEval wrappers from the same brief became `eval/splits.py` and
`eval/trackeval_runner.py`. Their output is identical: same gt rows, same `seqLength` and same
seqmap as the Research Lab's `data/mot17_val_half`. The vis > 0 oracle reproduces the brief's
MOTA 89.1 / IDF1 94.2 / HOTA 91.1 on MOT17 val-half and MOTA 96.9 on MOT20 val-half.

# Visibility estimation (PR 7)

All numbers are measured on this box, CPU-only, on MOT17 **public FRCNN** detections. Parameters
were chosen on **train_half** and run once on **val_half**. One caveat: the val_half table for the
estimator (below) was printed in the same run that compared geometry-only and geometry × score,
after the fusion had been chosen on train_half.

## 1. How good is the estimate? (scoring protocol: `docs/research/implementation_research.md` §2.4)

Reproduce: `python -m eval.visibility_eval --split val_half --out docs/results/visibility`
(full table with calibration, coverage and strata: `visibility_MOT17_val_half.md`).
Detections are matched to gt pedestrians (class 1, conf 1) at IoU ≥ 0.5; n = matched pairs.

| Estimate (val_half) | n | MAE | Spearman ρ | AUROC v<0.5 | AP v<0.5 |
|---|---|---|---|---|---|
| **ours: depth order × detector score** | 27176 | 0.074 | 0.613 | 0.938 | 0.460 |
| ours: depth order only | 27176 | 0.073 | 0.550 | 0.806 | 0.388 |
| detector score alone | 27176 | 0.107 | 0.429 | 0.947 | 0.293 |
| constant (mean) | 27176 | 0.129 | – | 0.500 | 0.031 |
| gt-box oracle, pedestrians only, same boxes | 27176 | 0.032 | 0.824 | 0.941 | 0.831 |
| gt-box oracle, all classes (implementation check) | 53890 | 0.023 | 0.984 | 0.990 | 0.990 |

* The all-classes oracle reproduces MOT gt visibility to MAE 0.023 / ρ 0.984, which checks that
  the depth-order rule and the coverage computation are right.
* Pedestrians-only oracle versus all classes shows the floor caused by occluders a person
  detector cannot see (cars, poles and so on).
* Coverage: the detector finds only 1.2 % of gt boxes with v < 0.25 and 4.0 % with 0.25 ≤ v < 0.5,
  so the scored pairs are mostly visible people (prevalence of v<0.5 among pairs: 2.0 %).
* The detector score alone has a slightly higher AUROC, but a much worse MAE and AP. The product
  is the best or close to the best on every column except AP v<0.25 (0.475 vs 0.501).
* train_half is much harder for the estimate (geometry only: AUROC 0.632, product 0.724;
  `tune/` and `tune_MOT17_train_half.json`). The val_half numbers should not be read as typical.
* occluder_min_score is insensitive on FRCNN (its scores are mostly ≈ 1). 0.1 was chosen on
  train_half by AUROC.

## 2. Does it help tracking? (MOT17 val_half, public FRCNN, no re-ID)

| Config | HOTA | IDF1 | MOTA | IDSW | p50 / p95 / max ms |
|---|---|---|---|---|---|
| ByteTrackLike (`configs/baselines/bytetrack.yaml`) | 50.1 | 56.3 | 47.5 | 191 | 0.88 / 1.86 / 2.91 |
| **+ visibility** (`configs/ours/visibility.yaml`) | 51.0 | 57.3 | 47.6 | 183 | 1.09 / 2.22 / 3.18 |
| control: track_buffer 90, no visibility | 51.0 | 57.2 | 47.6 | 183 | 1.01 / 2.38 / 3.26 |

**Verdict: on public FRCNN detections without re-ID, visibility adds nothing measurable beyond a
longer lost-track buffer.** The tuned config keeps lost tracks 3× longer when their last
detection had v̂ < 0.7. That gives +0.9 HOTA, and giving every track the 3× buffer gives the same.
On train_half (`tune/tune_stage2.json`), the gated buffer had slightly fewer ID switches than the
plain one (292 vs 301) with equal HOTA (53.80 vs 53.81). On val_half both had 183.

The visibility-softened IoU (β > 0) lowered train_half HOTA at every value tried (0.25, 0.5, 1.0),
so the frozen config has β = 0. The other planned use, freezing the appearance update when
v̂ is low (`appearance.min_visibility`, `appearance.cost_min_visibility`), needs re-ID
embeddings, which need a GPU. It is not measured here.

Latency is the single-threaded tracker `update` time from `run.json`; it includes the estimator.

## Files

* `visibility_MOT17_val_half.{md,json}`: estimator report.
* `tune_MOT17_train_half.json`: occluder_min_score sweep.
* `tune/tune_stage{1,2}.json`: tracker grid search on train_half
  (`scripts/research/tune_visibility.py`).

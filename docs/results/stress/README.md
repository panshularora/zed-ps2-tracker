# Stress suite · one person, one ID (CPU, synthetic)

Generated 2026-10-17 on the box CPU. No GPU numbers anywhere in this folder.

## What this measures

* **Generator** `eval/stress_sim.py`: 960×540 @ 30 fps scene with a counting line at y = 300; person-shaped
  boxes walking through it, each walker scripted to be fully hidden for `hide_s` seconds (plus 6-frame
  fade in / out) centred on the line, so they re-emerge on screen. Detector noise is seeded (box jitter,
  missed boxes, false boxes, score noise, truncation of partly hidden boxes). Each (scenario, seed) is
  deterministic (`tests/test_stress_suite.py::test_render_is_deterministic_and_seeded`).
* **Trackers**: ours = `configs/ours/visibility_botsort.yaml` (BoT-SORT + depth-order visibility,
  appearance off), baseline = `configs/baselines/bytetrack.yaml`. Both are the real `tracker/` code; no
  relabelling or display-side fixing anywhere.
* **Seeds**: reported seeds 1000–1019 (20 per cell, all of them, nothing dropped). Tuning used seeds
  0–9 only.
* **Per cell**: people, strict kept, ID changes, gate count error, IDF1 / AssA / HOTA / IDSW / Frag
  (vendored TrackEval, `third_party/trackeval`, in memory), per-seed values, the per-seed difference
  ours − ByteTrack in ID changes, its bootstrap 95 % CI (4 000 resamples) and exact sign-test p.
* **Verdict**: `better` if CI upper < 0 and p < 0.05; `worse` if CI lower > 0 and p < 0.05; `tie` if
  all differences are 0 or the CI is within ±0.5; else `unclear`.
* **Strict kept**: a person counts as kept if one tracker ID covers ≥ 80 % of the frames in which the
  person is visible (truth box visibility ≥ 0.25, IoU ≥ 0.5 Hungarian match) and that ID is not used on
  anyone else. `kept ≤ people` is asserted.
* **ID changes**: number of times the ID matched to a person differs from the previous matched ID.
* **Gate error**: tracker IN+OUT minus true IN+OUT (8 px hysteresis on box bottom-centre).
* **Explainability**: for every seed where the two trackers' ID histories diverge, `first_disagreement`
  stores the frame, the person, which tracker changed, and a trace from the tracker itself
  (`ByteTrackLike(trace_enabled=True)`, off by default, outputs identical either way: tested): when the
  old ID lost its box, its visibility, its memory, when it expired, and the matching candidates
  (IoU, cost, threshold, stage) for the person's box at the decision frame. Reasons: `expired`,
  `cost_above_threshold`, `taken_by_other_box`, `low_score_box`, `transfer`, `not_a_candidate`.

Example (grid, 10 people, 1 s hidden, seed 1000, ByteTrack), copied verbatim from `stress_map.json`:

> #3 was deleted at frame 154 after 31 frames without a box (memory 30); the person's next box started
> #9. When #3 lost its box its visibility was 1.00 (threshold 0.5), so its memory was not extended:
> 30 frames.

Reproduce: `python -m eval.stress_run grid --seeds 20 --seed0 1000 --workers 6` (and `scenarios`,
`noise`; add `--ours configs/ours/visibility_botsort_mem6.yaml --tag _mem6` for the after runs).
Runtimes on the box: grid 268 s base / 404 s mem6 (6 workers, other jobs running).

## Where ours loses or does not help (read this first)

* **0.5 s hides**: no advantage; at 20 people ours is worse (8.4 vs 7.3 ID changes per clip,
  CI [+0.1, +2.0], p = 0.022). Unchanged by the mem6 tweak.
* **Hides ≥ 3 s with the default config**: both trackers give nearly every hidden person a new number
  (strict kept ≈ 0) and ours is slightly worse than ByteTrack in 13 of 18 such cells. Ours keeps an occluded
  track for 90 frames (3 s) instead of 30, so at ≥ 3 s it expires too, and its first-disagreement
  traces are dominated by `expired` and `cost_above_threshold`.
* **Abrupt dropouts** (person's box vanishes without partial occlusion, 5/15/30 frames): ours is not
  better (+0.3 / +0.4 / +0.4 ID changes per clip, `unclear`), because the visibility rule only extends
  memory when the last box looked occluded.
* **Gate count error is negative for both trackers in every grid cell**: when the ID changes while the
  person is hidden on the line, no single ID crosses, so the crossing is lost. At hides ≥ 3 s the error
  is ≈ −people for the base config. Counting is only as good as ID continuity across the line.
* This is a synthetic scene with boxes, not pixels: there is no appearance signal and the detector is a
  noise model. It shows the association logic's behaviour, not real-video accuracy.

## Tracker change tried (tuned on seeds 0–9, checked on MOT17, reported on seeds 1000–1019)

Tuning subset (seeds 0–9): grid cells 10/30 people × 0.5/1.5/3/5 s plus group, look-alike crossing,
stop-and-go and 15-frame dropout. Total ID changes of ours (sum of per-cell means):

| config | change | tuning total ID changes |
|---|---|---|
| base | `visibility_botsort.yaml` | 156.0 |
| A · mem6 | `occluded_buffer_scale` 3 → 6 | 143.9 |
| B · BIoU 0.3 | buffered IoU for lost tracks (`lost_buffer_per_s` 0.3, max 0.6) | 145.1 |
| C · mem6 + BIoU 0.3 | | 118.0 |
| D · BIoU 0.5 | | 144.0 |
| E · mem6 + BIoU 0.5 | | 114.7 |

Guard-rail on real data, MOT17 val-half, public FRCNN detections, CPU (`eval.run`):

| config | HOTA | IDF1 | IDSW |
|---|---|---|---|
| base | 52.5 | 59.8 | 180 |
| A · mem6 | 52.5 | 59.7 | 181 |
| B · BIoU 0.3 | 52.2 | 59.0 | 172 |
| C · mem6 + BIoU 0.3 | 52.3 | 59.1 | 164 |
| E · mem6 + BIoU 0.5 | 51.8 | 58.3 | 176 |

Decision: BIoU variants beat mem6 alone on the synthetic tuning set but cost 0.7–1.5 IDF1 on MOT17
(B, C, E; D was not run on MOT17), so BIoU
was rejected and its code removed. **mem6 (A)** is neutral on MOT17 (−0.1 IDF1, +1 IDSW: noise level) and
is shipped as an opt-in config, `configs/ours/visibility_botsort_mem6.yaml`. The default ours config
is **not** changed. Before/after on the held-out seeds is below: mem6 fixes the ≥ 3 s regime (13 of 14
"worse" cells go to "better"/"unclear") and changes below 3 s by at most 0.2 ID changes per clip, but strict kept at ≥ 3 s is still
low and the 0.5 s / 20-people loss remains.

## Files

`stress_map.json/.csv` (grid, base), `stress_scenarios.json/.csv`, `stress_noise.json/.csv`, and the
same three with `_mem6`. Every json carries generator, git commit, tracker configs, seeds, rules and
elapsed time. The CSVs have one row per cell with ours_* and bytetrack_* columns.

## Tables
<!-- generated tables below by /workspace helper from the json files in this folder -->

### Grid · ours = base config (`configs/ours/visibility_botsort.yaml`)

Seeds 1000–1019 (20 per cell), CPU only, git `67c603c`, ours = `configs/ours/visibility_botsort.yaml`, baseline = `configs/baselines/bytetrack.yaml`.

**ID changes per clip, mean ours / ByteTrack · verdict on per-seed (ours − ByteTrack)**

| people \ hidden | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s | 4 s | 5 s |
|---|---|---|---|---|---|---|---|---|
| 5 | 1.0 / 1.1 · tie | 1.4 / 5.0 · better | 1.7 / 5.0 · better | 2.8 / 5.0 · better | 3.0 / 5.0 · better | 5.1 / 5.0 · tie | 5.2 / 5.0 · tie | 5.1 / 4.9 · tie |
| 10 | 2.8 / 2.5 · unclear | 3.5 / 10.5 · better | 4.2 / 10.6 · better | 5.3 / 10.3 · better | 7.6 / 10.3 · better | 10.9 / 10.2 · WORSE | 10.7 / 10.1 · WORSE | 10.4 / 10.0 · unclear |
| 15 | 4.5 / 3.9 · unclear | 5.0 / 16.4 · better | 7.1 / 16.1 · better | 9.8 / 15.8 · better | 10.0 / 15.8 · better | 16.2 / 15.8 · unclear | 16.1 / 15.2 · WORSE | 15.3 / 14.7 · WORSE |
| 20 | 8.4 / 7.3 · WORSE | 9.1 / 22.5 · better | 12.4 / 22.4 · better | 12.6 / 21.4 · better | 16.6 / 21.9 · better | 22.7 / 21.6 · WORSE | 21.9 / 20.5 · WORSE | 20.7 / 19.4 · WORSE |
| 30 | 15.9 / 15.0 · unclear | 16.6 / 35.0 · better | 20.2 / 35.1 · better | 24.0 / 34.5 · better | 26.8 / 33.5 · better | 35.3 / 33.6 · WORSE | 33.9 / 31.8 · WORSE | 31.6 / 29.6 · WORSE |
| 40 | 23.8 / 24.0 · unclear | 27.4 / 51.1 · better | 31.6 / 50.6 · better | 33.6 / 48.9 · better | 37.5 / 48.1 · better | 49.3 / 46.9 · WORSE | 46.8 / 44.1 · WORSE | 43.4 / 40.5 · WORSE |

**Mean paired difference ours − ByteTrack in ID changes [bootstrap 95% CI], sign-test p**

| people \ hidden | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s | 4 s | 5 s |
|---|---|---|---|---|---|---|---|---|
| 5 | -0.1 [-0.3, +0.2] p=1 | -3.6 [-4.2, -3.0] p<1e-4 | -3.3 [-3.7, -2.9] p<1e-4 | -2.2 [-2.8, -1.6] p<1e-4 | -2.0 [-2.6, -1.5] p=0.0001 | +0.1 [+0.0, +0.2] p=0.5 | +0.2 [+0.1, +0.4] p=0.12 | +0.2 [+0.1, +0.4] p=0.12 |
| 10 | +0.3 [+0.0, +0.7] p=0.22 | -7.0 [-7.7, -6.4] p<1e-4 | -6.3 [-7.2, -5.5] p<1e-4 | -5.0 [-5.8, -4.2] p<1e-4 | -2.8 [-3.3, -2.1] p<1e-4 | +0.8 [+0.2, +1.4] p=0.039 | +0.6 [+0.2, +1.0] p=0.039 | +0.5 [+0.1, +0.9] p=0.062 |
| 15 | +0.7 [+0.1, +1.2] p=0.11 | -11.3 [-12.1, -10.7] p<1e-4 | -8.9 [-9.8, -8.0] p<1e-4 | -6.0 [-6.8, -5.2] p<1e-4 | -5.8 [-6.9, -4.8] p<1e-4 | +0.5 [+0.0, +0.9] p=0.23 | +0.9 [+0.5, +1.4] p=0.001 | +0.7 [+0.5, +0.8] p=0.0002 |
| 20 | +1.1 [+0.1, +2.0] p=0.022 | -13.4 [-14.8, -11.8] p<1e-4 | -10.0 [-11.3, -8.6] p<1e-4 | -8.8 [-9.8, -8.0] p<1e-4 | -5.3 [-6.5, -4.3] p<1e-4 | +1.1 [+0.6, +1.7] p=0.0023 | +1.4 [+0.7, +2.2] p=0.001 | +1.3 [+0.9, +1.7] p<1e-4 |
| 30 | +0.9 [-0.1, +1.9] p=0.035 | -18.4 [-20.1, -16.7] p<1e-4 | -14.9 [-16.1, -13.8] p<1e-4 | -10.5 [-11.8, -9.2] p<1e-4 | -6.7 [-8.3, -5.1] p<1e-4 | +1.7 [+0.8, +2.6] p=0.0044 | +2.1 [+1.1, +3.1] p=0.0005 | +2.0 [+1.5, +2.6] p<1e-4 |
| 40 | -0.2 [-1.6, +1.2] p=0.8 | -23.8 [-25.7, -21.6] p<1e-4 | -19.1 [-21.4, -16.6] p<1e-4 | -15.2 [-17.1, -13.4] p<1e-4 | -10.7 [-12.6, -8.8] p<1e-4 | +2.5 [+1.2, +3.7] p=0.0074 | +2.6 [+1.6, +3.6] p=0.0003 | +3.0 [+1.9, +4.0] p=0.0003 |

**Strict kept (≥80% of visible frames, one ID, not shared), mean ours / ByteTrack / people**

| people \ hidden | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s | 4 s | 5 s |
|---|---|---|---|---|---|---|---|---|
| 5 | 4.0 / 3.9 / 5.0 | 3.6 / 0.0 / 5.0 | 3.2 / 0.0 / 5.0 | 2.3 / 0.0 / 5.0 | 2.0 / 0.0 / 5.0 | 0.0 / 0.0 / 5.0 | 0.0 / 0.0 / 5.0 | 0.0 / 0.0 / 5.0 |
| 10 | 7.2 / 7.4 / 10.0 | 6.6 / 0.0 / 10.0 | 6.0 / 0.0 / 10.0 | 4.5 / 0.0 / 10.0 | 2.6 / 0.0 / 10.0 | 0.0 / 0.0 / 10.0 | 0.0 / 0.0 / 10.0 | 0.0 / 0.0 / 10.0 |
| 15 | 10.2 / 10.4 / 15.0 | 9.4 / 0.1 / 15.0 | 7.8 / 0.0 / 15.0 | 5.3 / 0.0 / 15.0 | 5.0 / 0.0 / 15.0 | 0.1 / 0.1 / 15.0 | 0.1 / 0.1 / 15.0 | 0.1 / 0.2 / 15.0 |
| 20 | 12.4 / 12.2 / 20.0 | 11.1 / 0.0 / 20.0 | 9.0 / 0.0 / 20.0 | 7.8 / 0.0 / 20.0 | 5.1 / 0.0 / 20.0 | 0.0 / 0.0 / 20.0 | 0.0 / 0.0 / 20.0 | 0.1 / 0.1 / 20.0 |
| 30 | 15.2 / 14.9 / 30.0 | 14.4 / 0.0 / 30.0 | 11.8 / 0.0 / 30.0 | 8.9 / 0.0 / 30.0 | 6.9 / 0.0 / 30.0 | 0.1 / 0.0 / 30.0 | 0.0 / 0.0 / 30.0 | 0.3 / 0.2 / 30.0 |
| 40 | 17.6 / 16.5 / 40.0 | 14.9 / 0.0 / 40.0 | 11.9 / 0.1 / 40.0 | 10.5 / 0.1 / 40.0 | 7.5 / 0.1 / 40.0 | 0.1 / 0.1 / 40.0 | 0.1 / 0.1 / 40.0 | 0.3 / 0.1 / 40.0 |

**IDF1 range over seeds (min–max), ours / ByteTrack**

| people \ hidden | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s | 4 s | 5 s |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.56–0.96 / 0.56–0.95 | 0.62–0.96 / 0.47–0.56 | 0.69–0.94 / 0.47–0.56 | 0.52–0.94 / 0.47–0.57 | 0.54–0.94 / 0.47–0.58 | 0.47–0.59 / 0.47–0.58 | 0.50–0.60 / 0.49–0.60 | 0.48–0.63 / 0.47–0.62 |
| 10 | 0.72–0.95 / 0.74–0.95 | 0.71–0.94 / 0.49–0.56 | 0.66–0.90 / 0.49–0.55 | 0.66–0.88 / 0.50–0.55 | 0.57–0.77 / 0.50–0.56 | 0.50–0.57 / 0.50–0.56 | 0.51–0.58 / 0.51–0.58 | 0.51–0.60 / 0.51–0.60 |
| 15 | 0.69–0.92 / 0.69–0.92 | 0.72–0.90 / 0.50–0.55 | 0.66–0.87 / 0.50–0.55 | 0.64–0.83 / 0.50–0.55 | 0.61–0.80 / 0.50–0.56 | 0.51–0.56 / 0.51–0.56 | 0.51–0.59 / 0.51–0.58 | 0.52–0.61 / 0.51–0.60 |
| 20 | 0.75–0.91 / 0.71–0.90 | 0.72–0.88 / 0.50–0.55 | 0.68–0.84 / 0.51–0.54 | 0.63–0.86 / 0.51–0.54 | 0.58–0.76 / 0.50–0.55 | 0.51–0.55 / 0.51–0.55 | 0.51–0.57 / 0.50–0.57 | 0.53–0.59 / 0.52–0.59 |
| 30 | 0.67–0.83 / 0.69–0.84 | 0.66–0.84 / 0.49–0.53 | 0.67–0.79 / 0.50–0.54 | 0.61–0.76 / 0.50–0.54 | 0.56–0.71 / 0.50–0.55 | 0.51–0.56 / 0.50–0.55 | 0.52–0.57 / 0.51–0.56 | 0.52–0.58 / 0.52–0.58 |
| 40 | 0.70–0.83 / 0.66–0.81 | 0.69–0.80 / 0.49–0.54 | 0.66–0.77 / 0.50–0.54 | 0.61–0.75 / 0.50–0.54 | 0.60–0.68 / 0.49–0.55 | 0.50–0.55 / 0.50–0.55 | 0.50–0.57 / 0.50–0.56 | 0.51–0.58 / 0.51–0.57 |

**AssA range over seeds (min–max), ours / ByteTrack**

| people \ hidden | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s | 4 s | 5 s |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.49–0.89 / 0.49–0.88 | 0.55–0.89 / 0.39–0.45 | 0.61–0.86 / 0.40–0.45 | 0.41–0.85 / 0.40–0.45 | 0.41–0.86 / 0.40–0.45 | 0.39–0.46 / 0.39–0.46 | 0.38–0.46 / 0.38–0.45 | 0.37–0.47 / 0.36–0.47 |
| 10 | 0.62–0.87 / 0.62–0.87 | 0.59–0.86 / 0.39–0.45 | 0.56–0.80 / 0.39–0.45 | 0.57–0.79 / 0.41–0.44 | 0.45–0.68 / 0.39–0.44 | 0.38–0.45 / 0.38–0.45 | 0.39–0.45 / 0.37–0.45 | 0.38–0.45 / 0.38–0.44 |
| 15 | 0.58–0.84 / 0.57–0.84 | 0.64–0.81 / 0.37–0.44 | 0.54–0.78 / 0.38–0.44 | 0.55–0.73 / 0.38–0.44 | 0.50–0.70 / 0.39–0.44 | 0.39–0.44 / 0.39–0.44 | 0.39–0.44 / 0.38–0.44 | 0.38–0.45 / 0.38–0.45 |
| 20 | 0.64–0.82 / 0.59–0.81 | 0.61–0.79 / 0.38–0.43 | 0.57–0.74 / 0.40–0.43 | 0.51–0.77 / 0.40–0.43 | 0.48–0.67 / 0.39–0.43 | 0.39–0.44 / 0.39–0.43 | 0.40–0.44 / 0.38–0.43 | 0.40–0.44 / 0.39–0.44 |
| 30 | 0.57–0.74 / 0.58–0.75 | 0.54–0.76 / 0.37–0.43 | 0.56–0.70 / 0.37–0.42 | 0.50–0.66 / 0.39–0.42 | 0.44–0.59 / 0.39–0.42 | 0.39–0.43 / 0.39–0.42 | 0.40–0.43 / 0.39–0.42 | 0.38–0.43 / 0.38–0.43 |
| 40 | 0.57–0.72 / 0.52–0.70 | 0.55–0.70 / 0.36–0.42 | 0.52–0.66 / 0.37–0.41 | 0.47–0.65 / 0.35–0.41 | 0.47–0.57 / 0.36–0.41 | 0.37–0.42 / 0.37–0.42 | 0.37–0.43 / 0.37–0.42 | 0.38–0.43 / 0.37–0.41 |

**Gate count error (tracker − truth), mean ours / ByteTrack**

| people \ hidden | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s | 4 s | 5 s |
|---|---|---|---|---|---|---|---|---|
| 5 | -0.6 / -0.8 | -1.1 / -4.8 | -1.4 / -4.8 | -2.4 / -5.0 | -2.8 / -5.0 | -5.0 / -5.0 | -5.0 / -5.0 | -5.0 / -5.0 |
| 10 | -0.8 / -0.8 | -1.9 / -8.9 | -2.8 / -9.6 | -4.2 / -9.7 | -6.4 / -9.9 | -9.6 / -9.9 | -10.0 / -10.0 | -9.9 / -10.0 |
| 15 | -1.1 / -1.1 | -2.4 / -13.3 | -5.0 / -14.4 | -7.2 / -14.8 | -8.2 / -14.8 | -14.8 / -15.0 | -14.9 / -15.0 | -15.0 / -15.0 |
| 20 | -1.9 / -2.5 | -3.9 / -17.6 | -6.7 / -18.8 | -9.7 / -19.9 | -12.8 / -19.9 | -19.8 / -19.9 | -20.0 / -20.0 | -20.0 / -20.0 |
| 30 | -4.0 / -4.9 | -6.5 / -25.6 | -11.2 / -28.1 | -15.5 / -29.0 | -18.9 / -29.6 | -29.1 / -29.9 | -29.8 / -30.0 | -29.9 / -30.0 |
| 40 | -4.2 / -5.8 | -10.0 / -34.1 | -16.1 / -37.3 | -21.4 / -39.0 | -26.3 / -39.6 | -39.1 / -39.6 | -39.6 / -40.0 | -39.6 / -40.0 |

Verdicts over 48 cells: WORSE 14, better 24, tie 4, unclear 6

Seeds where the two trackers' ID histories diverge: 951 of 960. Reasons in the trace at the first disagreement (tracker, reason → count): bytetrack/expired 558, ours/expired 454, ours/cost_above_threshold 192, bytetrack/taken_by_other_box 111, ours/taken_by_other_box 96, bytetrack/cost_above_threshold 50, bytetrack/transfer 17, ours/transfer 12

### Scenarios · base config

| scenario | people | ID changes ours / BT | diff [95% CI] · verdict | kept ours / BT | IDF1 range ours / BT | AssA range ours / BT | gate err ours / BT |
|---|---|---|---|---|---|---|---|
| lookalike_cross | 6.9 | 2.15 / 4.00 | -1.85 [-2.45, -1.25] · better | 4.5 / 2.7 | 0.62–0.93 / 0.61–0.78 | 0.51–0.84 / 0.50–0.68 | -2.7 / -4.8 |
| stop_and_go | 10.0 | 8.75 / 11.25 | -2.50 [-3.10, -1.90] · better | 2.9 / 0.0 | 0.63–0.80 / 0.56–0.64 | 0.50–0.71 / 0.42–0.50 | -1.6 / -4.0 |
| step_back | 10.0 | 3.60 / 10.45 | -6.85 [-7.45, -6.25] · better | 6.5 / 0.0 | 0.75–0.90 / 0.53–0.59 | 0.66–0.83 / 0.41–0.47 | -2.9 / -9.8 |
| group | 9.3 | 4.00 / 5.70 | -1.70 [-2.35, -1.00] · better | 3.5 / 2.6 | 0.63–0.89 / 0.61–0.78 | 0.47–0.80 / 0.44–0.71 | -1.6 / -3.8 |
| camera_shake | 10.0 | 7.25 / 10.75 | -3.50 [-4.50, -2.55] · better | 3.6 / 0.0 | 0.56–0.85 / 0.48–0.55 | 0.43–0.75 / 0.37–0.43 | -5.0 / -9.2 |
| linger | 10.0 | 5.30 / 10.95 | -5.65 [-6.45, -4.85] · better | 5.5 / 0.0 | 0.65–0.96 / 0.54–0.64 | 0.52–0.90 / 0.39–0.51 | -2.0 / -7.6 |
| fast_walker | 10.0 | 3.45 / 10.65 | -7.20 [-8.00, -6.35] · better | 6.8 / 0.0 | 0.71–0.95 / 0.48–0.55 | 0.59–0.87 / 0.39–0.45 | -1.9 / -8.9 |
| dropout_5f | 10.0 | 1.00 / 0.70 | +0.30 [+0.10, +0.55] · unclear | 8.8 / 8.9 | 0.82–0.96 / 0.79–0.96 | 0.70–0.90 / 0.67–0.89 | +0.1 / +0.1 |
| dropout_15f | 10.0 | 1.40 / 1.00 | +0.40 [+0.10, +0.75] · unclear | 8.3 / 8.6 | 0.85–0.96 / 0.80–0.96 | 0.74–0.90 / 0.69–0.89 | +0.1 / -0.1 |
| dropout_30f | 10.0 | 2.05 / 1.65 | +0.40 [-0.10, +0.85] · unclear | 7.8 / 8.1 | 0.80–0.95 / 0.74–0.95 | 0.71–0.88 / 0.65–0.88 | -0.6 / -1.0 |

### Detector-noise sweep · base config

| scenario | people | ID changes ours / BT | diff [95% CI] · verdict | kept ours / BT | IDF1 range ours / BT | AssA range ours / BT | gate err ours / BT |
|---|---|---|---|---|---|---|---|
| noise 0.05 · 10 people · hide 1 s | 10.0 | 3.40 / 10.45 | -7.05 [-7.50, -6.60] · better | 6.5 / 0.0 | 0.72–0.89 / 0.48–0.56 | 0.66–0.83 / 0.41–0.46 | -1.7 / -8.8 |
| noise 0.05 · 10 people · hide 3 s | 10.0 | 10.60 / 10.15 | +0.45 [+0.15, +0.80] · WORSE | 0.0 / 0.0 | 0.50–0.57 / 0.50–0.57 | 0.40–0.46 / 0.41–0.46 | -9.8 / -9.9 |
| noise 0.05 · 30 people · hide 1 s | 30.0 | 17.00 / 35.80 | -18.80 [-20.45, -17.25] · better | 14.6 / 0.0 | 0.67–0.88 / 0.50–0.54 | 0.57–0.80 / 0.39–0.44 | -7.1 / -26.0 |
| noise 0.05 · 30 people · hide 3 s | 30.0 | 34.10 / 32.60 | +1.50 [+0.60, +2.50] · WORSE | 0.0 / 0.0 | 0.51–0.56 / 0.50–0.56 | 0.41–0.45 / 0.40–0.43 | -29.2 / -29.9 |
| noise 0.15 · 10 people · hide 1 s | 10.0 | 3.45 / 10.50 | -7.05 [-7.65, -6.45] · better | 6.6 / 0.0 | 0.71–0.94 / 0.49–0.56 | 0.59–0.86 / 0.39–0.45 | -1.9 / -8.9 |
| noise 0.15 · 10 people · hide 3 s | 10.0 | 10.90 / 10.15 | +0.75 [+0.25, +1.35] · WORSE | 0.0 / 0.0 | 0.50–0.57 / 0.50–0.56 | 0.38–0.45 / 0.38–0.45 | -9.6 / -9.9 |
| noise 0.15 · 30 people · hide 1 s | 30.0 | 16.60 / 35.05 | -18.45 [-20.10, -16.70] · better | 14.4 / 0.0 | 0.66–0.84 / 0.49–0.53 | 0.54–0.76 / 0.37–0.43 | -6.5 / -25.6 |
| noise 0.15 · 30 people · hide 3 s | 30.0 | 35.30 / 33.60 | +1.70 [+0.75, +2.60] · WORSE | 0.1 / 0.0 | 0.51–0.56 / 0.50–0.55 | 0.39–0.43 / 0.39–0.42 | -29.1 / -29.9 |
| noise 0.4 · 10 people · hide 1 s | 10.0 | 4.50 / 10.35 | -5.85 [-6.50, -5.15] · better | 5.4 / 0.0 | 0.69–0.89 / 0.48–0.54 | 0.57–0.75 / 0.36–0.41 | -2.5 / -9.2 |
| noise 0.4 · 10 people · hide 3 s | 10.0 | 11.25 / 10.10 | +1.15 [+0.75, +1.55] · WORSE | 0.0 / 0.0 | 0.50–0.55 / 0.50–0.55 | 0.38–0.41 / 0.37–0.41 | -9.9 / -9.9 |
| noise 0.4 · 30 people · hide 1 s | 30.0 | 22.40 / 36.65 | -14.25 [-15.60, -12.80] · better | 9.8 / 0.0 | 0.62–0.79 / 0.48–0.53 | 0.47–0.66 / 0.35–0.39 | -9.7 / -26.6 |
| noise 0.4 · 30 people · hide 3 s | 30.0 | 35.95 / 33.70 | +2.25 [+1.05, +3.45] · WORSE | 0.0 / 0.0 | 0.50–0.54 / 0.49–0.54 | 0.36–0.40 / 0.36–0.39 | -29.2 / -29.9 |

## Tracker change: before / after (`visibility_botsort_mem6.yaml`)

**Grid ID changes: ours base → ours mem6 (same seeds; ByteTrack unchanged)**

| people \ hidden | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s | 4 s | 5 s |
|---|---|---|---|---|---|---|---|---|
| 5 | 1.0 → 1.0 | 1.4 → 1.4 | 1.7 → 1.7 | 2.8 → 2.8 | 3.0 → 3.0 | 5.1 → 4.0 | 5.2 → 4.5 | 5.1 → 4.5 |
| 10 | 2.8 → 2.8 | 3.5 → 3.5 | 4.2 → 4.2 | 5.3 → 5.3 | 7.6 → 7.6 | 10.9 → 7.4 | 10.7 → 9.1 | 10.4 → 9.3 |
| 15 | 4.5 → 4.5 | 5.0 → 5.0 | 7.1 → 7.1 | 9.8 → 9.8 | 10.0 → 10.0 | 16.2 → 11.9 | 16.1 → 13.3 | 15.3 → 14.4 |
| 20 | 8.4 → 8.4 | 9.1 → 9.1 | 12.4 → 12.4 | 12.6 → 12.6 | 16.6 → 16.6 | 22.7 → 17.4 | 21.9 → 18.1 | 20.7 → 19.4 |
| 30 | 15.9 → 15.9 | 16.6 → 16.6 | 20.2 → 20.1 | 24.0 → 24.1 | 26.8 → 26.6 | 35.3 → 27.9 | 33.9 → 29.2 | 31.6 → 29.5 |
| 40 | 23.8 → 23.8 | 27.4 → 27.2 | 31.6 → 31.6 | 33.6 → 33.5 | 37.5 → 37.3 | 49.3 → 39.1 | 46.8 → 41.0 | 43.4 → 40.7 |

Sum of mean ID changes over all cells: 786.7 → 726.1. Verdicts vs ByteTrack: {'tie': 4, 'better': 24, 'unclear': 6, 'WORSE': 14} → {'tie': 1, 'better': 36, 'unclear': 10, 'WORSE': 1}.

**Scenarios: ours base → ours mem6 (same seeds; ByteTrack unchanged)**

| scenario | ID changes base → mem6 | verdict base → mem6 | IDF1 mean base → mem6 |
|---|---|---|---|
| lookalike_cross | 2.15 → 2.10 | better → better | 0.822 → 0.824 |
| stop_and_go | 8.75 → 8.75 | better → better | 0.721 → 0.721 |
| step_back | 3.60 → 3.65 | better → better | 0.839 → 0.839 |
| group | 4.00 → 3.95 | better → better | 0.783 → 0.783 |
| camera_shake | 7.25 → 7.25 | better → better | 0.696 → 0.696 |
| linger | 5.30 → 5.30 | better → better | 0.807 → 0.807 |
| fast_walker | 3.45 → 3.50 | better → better | 0.836 → 0.836 |
| dropout_5f | 1.00 → 1.00 | unclear → unclear | 0.933 → 0.933 |
| dropout_15f | 1.40 → 1.40 | unclear → unclear | 0.923 → 0.923 |
| dropout_30f | 2.05 → 2.05 | unclear → unclear | 0.899 → 0.899 |

**Noise: ours base → ours mem6 (same seeds; ByteTrack unchanged)**

| scenario | ID changes base → mem6 | verdict base → mem6 | IDF1 mean base → mem6 |
|---|---|---|---|
| noise 0.05 · 10 people · hide 1 s | 3.40 → 3.40 | better → better | 0.828 → 0.828 |
| noise 0.05 · 10 people · hide 3 s | 10.60 → 7.90 | WORSE → better | 0.539 → 0.649 |
| noise 0.05 · 30 people · hide 1 s | 17.00 → 16.95 | better → better | 0.764 → 0.764 |
| noise 0.05 · 30 people · hide 3 s | 34.10 → 25.40 | WORSE → better | 0.536 → 0.641 |
| noise 0.15 · 10 people · hide 1 s | 3.45 → 3.50 | better → better | 0.835 → 0.835 |
| noise 0.15 · 10 people · hide 3 s | 10.90 → 7.40 | WORSE → better | 0.536 → 0.663 |
| noise 0.15 · 30 people · hide 1 s | 16.60 → 16.55 | better → better | 0.779 → 0.780 |
| noise 0.15 · 30 people · hide 3 s | 35.30 → 27.85 | WORSE → better | 0.528 → 0.622 |
| noise 0.4 · 10 people · hide 1 s | 4.50 → 4.50 | better → better | 0.791 → 0.791 |
| noise 0.4 · 10 people · hide 3 s | 11.25 → 8.95 | WORSE → better | 0.524 → 0.599 |
| noise 0.4 · 30 people · hide 1 s | 22.40 → 22.75 | better → better | 0.714 → 0.714 |
| noise 0.4 · 30 people · hide 3 s | 35.95 → 30.80 | WORSE → better | 0.516 → 0.578 |

### Grid · ours = mem6 (full tables)

Seeds 1000–1019 (20 per cell), CPU only, git `b9dd300`, ours = `configs/ours/visibility_botsort_mem6.yaml`, baseline = `configs/baselines/bytetrack.yaml`.

**ID changes per clip, mean ours / ByteTrack · verdict on per-seed (ours − ByteTrack)**

| people \ hidden | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s | 4 s | 5 s |
|---|---|---|---|---|---|---|---|---|
| 5 | 1.0 / 1.1 · tie | 1.4 / 5.0 · better | 1.7 / 5.0 · better | 2.8 / 5.0 · better | 3.0 / 5.0 · better | 4.0 / 5.0 · better | 4.5 / 5.0 · unclear | 4.5 / 4.9 · unclear |
| 10 | 2.8 / 2.5 · unclear | 3.5 / 10.5 · better | 4.2 / 10.6 · better | 5.3 / 10.3 · better | 7.6 / 10.3 · better | 7.4 / 10.2 · better | 9.1 / 10.1 · better | 9.3 / 10.0 · better |
| 15 | 4.5 / 3.9 · unclear | 5.0 / 16.4 · better | 7.1 / 16.1 · better | 9.8 / 15.8 · better | 10.0 / 15.8 · better | 11.9 / 15.8 · better | 13.3 / 15.2 · better | 14.4 / 14.7 · unclear |
| 20 | 8.4 / 7.3 · WORSE | 9.1 / 22.5 · better | 12.4 / 22.4 · better | 12.6 / 21.4 · better | 16.6 / 21.9 · better | 17.4 / 21.6 · better | 18.1 / 20.5 · better | 19.4 / 19.4 · unclear |
| 30 | 15.9 / 15.0 · unclear | 16.6 / 35.0 · better | 20.1 / 35.1 · better | 24.1 / 34.5 · better | 26.6 / 33.5 · better | 27.9 / 33.6 · better | 29.2 / 31.8 · better | 29.5 / 29.6 · unclear |
| 40 | 23.8 / 24.0 · unclear | 27.2 / 51.1 · better | 31.6 / 50.6 · better | 33.5 / 48.9 · better | 37.3 / 48.1 · better | 39.1 / 46.9 · better | 41.0 / 44.1 · better | 40.7 / 40.5 · unclear |

**Mean paired difference ours − ByteTrack in ID changes [bootstrap 95% CI], sign-test p**

| people \ hidden | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s | 4 s | 5 s |
|---|---|---|---|---|---|---|---|---|
| 5 | -0.1 [-0.3, +0.2] p=1 | -3.6 [-4.2, -3.0] p<1e-4 | -3.3 [-3.7, -2.9] p<1e-4 | -2.2 [-2.8, -1.6] p<1e-4 | -2.0 [-2.6, -1.5] p=0.0001 | -0.9 [-1.4, -0.5] p=0.0074 | -0.5 [-0.9, -0.1] p=0.065 | -0.4 [-0.8, -0.1] p=0.07 |
| 10 | +0.3 [+0.0, +0.7] p=0.22 | -7.0 [-7.6, -6.3] p<1e-4 | -6.3 [-7.2, -5.5] p<1e-4 | -5.0 [-5.8, -4.3] p<1e-4 | -2.8 [-3.3, -2.1] p<1e-4 | -2.8 [-3.6, -1.9] p=0.0001 | -1.1 [-1.6, -0.5] p=0.0013 | -0.7 [-1.1, -0.2] p=0.0063 |
| 15 | +0.7 [+0.1, +1.2] p=0.11 | -11.3 [-12.1, -10.7] p<1e-4 | -8.9 [-9.8, -8.0] p<1e-4 | -6.0 [-6.8, -5.2] p<1e-4 | -5.8 [-6.9, -4.8] p<1e-4 | -3.8 [-4.9, -2.8] p=0.0004 | -1.9 [-2.7, -1.1] p=0.0044 | -0.2 [-0.8, +0.2] p=0.58 |
| 20 | +1.1 [+0.1, +2.0] p=0.022 | -13.4 [-14.8, -11.8] p<1e-4 | -10.0 [-11.3, -8.6] p<1e-4 | -8.8 [-9.8, -8.0] p<1e-4 | -5.3 [-6.5, -4.2] p<1e-4 | -4.2 [-5.3, -3.1] p<1e-4 | -2.5 [-3.6, -1.2] p=0.0007 | -0.1 [-0.7, +0.6] p=0.8 |
| 30 | +0.9 [-0.1, +1.9] p=0.057 | -18.5 [-20.2, -16.7] p<1e-4 | -15.0 [-16.2, -13.8] p<1e-4 | -10.4 [-11.7, -9.2] p<1e-4 | -6.8 [-8.5, -5.2] p<1e-4 | -5.8 [-7.3, -4.2] p<1e-4 | -2.5 [-3.6, -1.3] p=0.0007 | -0.1 [-0.7, +0.6] p=0.61 |
| 40 | -0.2 [-1.6, +1.1] p=0.61 | -23.9 [-25.9, -21.8] p<1e-4 | -19.1 [-21.4, -16.6] p<1e-4 | -15.4 [-17.4, -13.6] p<1e-4 | -10.8 [-12.8, -8.8] p<1e-4 | -7.7 [-9.4, -6.1] p<1e-4 | -3.1 [-4.7, -1.6] p=0.0075 | +0.2 [-1.0, +1.6] p=0.48 |

**Strict kept (≥80% of visible frames, one ID, not shared), mean ours / ByteTrack / people**

| people \ hidden | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s | 4 s | 5 s |
|---|---|---|---|---|---|---|---|---|
| 5 | 4.0 / 3.9 / 5.0 | 3.6 / 0.0 / 5.0 | 3.2 / 0.0 / 5.0 | 2.3 / 0.0 / 5.0 | 2.0 / 0.0 / 5.0 | 0.8 / 0.0 / 5.0 | 0.6 / 0.0 / 5.0 | 0.5 / 0.0 / 5.0 |
| 10 | 7.2 / 7.4 / 10.0 | 6.6 / 0.0 / 10.0 | 6.0 / 0.0 / 10.0 | 4.5 / 0.0 / 10.0 | 2.6 / 0.0 / 10.0 | 2.6 / 0.0 / 10.0 | 1.1 / 0.0 / 10.0 | 0.7 / 0.0 / 10.0 |
| 15 | 10.2 / 10.4 / 15.0 | 9.4 / 0.1 / 15.0 | 7.8 / 0.0 / 15.0 | 5.3 / 0.0 / 15.0 | 5.0 / 0.0 / 15.0 | 3.2 / 0.1 / 15.0 | 1.4 / 0.1 / 15.0 | 0.8 / 0.2 / 15.0 |
| 20 | 12.4 / 12.2 / 20.0 | 11.1 / 0.0 / 20.0 | 9.0 / 0.0 / 20.0 | 7.8 / 0.0 / 20.0 | 5.2 / 0.0 / 20.0 | 3.6 / 0.0 / 20.0 | 2.0 / 0.0 / 20.0 | 0.7 / 0.1 / 20.0 |
| 30 | 15.2 / 14.9 / 30.0 | 14.4 / 0.0 / 30.0 | 11.8 / 0.0 / 30.0 | 8.8 / 0.0 / 30.0 | 7.0 / 0.0 / 30.0 | 4.7 / 0.0 / 30.0 | 2.5 / 0.0 / 30.0 | 1.2 / 0.2 / 30.0 |
| 40 | 17.6 / 16.5 / 40.0 | 14.9 / 0.0 / 40.0 | 11.8 / 0.1 / 40.0 | 10.4 / 0.1 / 40.0 | 7.5 / 0.1 / 40.0 | 5.4 / 0.1 / 40.0 | 3.2 / 0.1 / 40.0 | 1.2 / 0.1 / 40.0 |

**IDF1 range over seeds (min–max), ours / ByteTrack**

| people \ hidden | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s | 4 s | 5 s |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.56–0.96 / 0.56–0.95 | 0.62–0.96 / 0.47–0.56 | 0.69–0.94 / 0.47–0.56 | 0.52–0.94 / 0.47–0.57 | 0.54–0.94 / 0.47–0.58 | 0.51–0.82 / 0.47–0.58 | 0.54–0.74 / 0.49–0.60 | 0.49–0.78 / 0.47–0.62 |
| 10 | 0.72–0.95 / 0.74–0.95 | 0.71–0.94 / 0.49–0.56 | 0.66–0.90 / 0.49–0.55 | 0.66–0.88 / 0.50–0.55 | 0.57–0.77 / 0.50–0.56 | 0.57–0.77 / 0.50–0.56 | 0.54–0.67 / 0.51–0.58 | 0.51–0.70 / 0.51–0.60 |
| 15 | 0.69–0.92 / 0.69–0.92 | 0.72–0.90 / 0.50–0.55 | 0.66–0.87 / 0.50–0.55 | 0.64–0.83 / 0.50–0.55 | 0.61–0.80 / 0.50–0.56 | 0.57–0.77 / 0.51–0.56 | 0.52–0.68 / 0.51–0.58 | 0.53–0.65 / 0.51–0.60 |
| 20 | 0.75–0.91 / 0.71–0.90 | 0.72–0.88 / 0.50–0.55 | 0.68–0.84 / 0.51–0.54 | 0.63–0.86 / 0.51–0.54 | 0.58–0.76 / 0.50–0.55 | 0.56–0.71 / 0.51–0.55 | 0.53–0.68 / 0.50–0.57 | 0.53–0.64 / 0.52–0.59 |
| 30 | 0.67–0.84 / 0.69–0.84 | 0.66–0.84 / 0.49–0.53 | 0.67–0.79 / 0.50–0.54 | 0.61–0.76 / 0.50–0.54 | 0.55–0.71 / 0.50–0.55 | 0.52–0.68 / 0.50–0.55 | 0.53–0.65 / 0.51–0.56 | 0.54–0.64 / 0.52–0.58 |
| 40 | 0.70–0.83 / 0.66–0.81 | 0.69–0.80 / 0.49–0.54 | 0.66–0.77 / 0.50–0.54 | 0.61–0.76 / 0.50–0.54 | 0.60–0.69 / 0.49–0.55 | 0.56–0.66 / 0.50–0.55 | 0.53–0.63 / 0.50–0.56 | 0.53–0.62 / 0.51–0.57 |

**AssA range over seeds (min–max), ours / ByteTrack**

| people \ hidden | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s | 4 s | 5 s |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.49–0.89 / 0.49–0.88 | 0.55–0.89 / 0.39–0.45 | 0.61–0.86 / 0.40–0.45 | 0.41–0.85 / 0.40–0.45 | 0.41–0.86 / 0.40–0.45 | 0.40–0.74 / 0.39–0.46 | 0.42–0.64 / 0.38–0.45 | 0.40–0.65 / 0.36–0.47 |
| 10 | 0.62–0.87 / 0.62–0.87 | 0.59–0.86 / 0.39–0.45 | 0.56–0.80 / 0.39–0.45 | 0.57–0.79 / 0.41–0.44 | 0.45–0.68 / 0.39–0.44 | 0.45–0.66 / 0.38–0.45 | 0.44–0.55 / 0.37–0.45 | 0.37–0.57 / 0.38–0.44 |
| 15 | 0.58–0.84 / 0.57–0.84 | 0.64–0.81 / 0.37–0.44 | 0.54–0.78 / 0.38–0.44 | 0.55–0.73 / 0.38–0.44 | 0.50–0.70 / 0.39–0.44 | 0.47–0.68 / 0.39–0.44 | 0.41–0.57 / 0.38–0.44 | 0.39–0.52 / 0.38–0.45 |
| 20 | 0.64–0.82 / 0.59–0.81 | 0.61–0.79 / 0.38–0.43 | 0.57–0.74 / 0.40–0.43 | 0.51–0.77 / 0.40–0.43 | 0.47–0.67 / 0.39–0.43 | 0.45–0.60 / 0.39–0.43 | 0.41–0.56 / 0.38–0.43 | 0.38–0.50 / 0.39–0.44 |
| 30 | 0.57–0.74 / 0.58–0.75 | 0.54–0.76 / 0.37–0.43 | 0.56–0.70 / 0.37–0.42 | 0.49–0.65 / 0.39–0.42 | 0.43–0.59 / 0.39–0.42 | 0.40–0.57 / 0.39–0.42 | 0.41–0.53 / 0.39–0.42 | 0.41–0.50 / 0.38–0.43 |
| 40 | 0.57–0.72 / 0.52–0.70 | 0.54–0.70 / 0.36–0.42 | 0.52–0.66 / 0.37–0.41 | 0.47–0.65 / 0.35–0.41 | 0.46–0.57 / 0.36–0.41 | 0.43–0.55 / 0.37–0.42 | 0.41–0.50 / 0.37–0.42 | 0.38–0.49 / 0.37–0.41 |

**Gate count error (tracker − truth), mean ours / ByteTrack**

| people \ hidden | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s | 4 s | 5 s |
|---|---|---|---|---|---|---|---|---|
| 5 | -0.6 / -0.8 | -1.1 / -4.8 | -1.4 / -4.8 | -2.4 / -5.0 | -2.8 / -5.0 | -3.6 / -5.0 | -4.1 / -5.0 | -4.3 / -5.0 |
| 10 | -0.8 / -0.8 | -1.9 / -8.9 | -2.8 / -9.6 | -4.1 / -9.7 | -6.3 / -9.9 | -6.0 / -9.9 | -8.2 / -10.0 | -8.6 / -10.0 |
| 15 | -1.1 / -1.1 | -2.4 / -13.3 | -5.0 / -14.4 | -7.1 / -14.8 | -8.2 / -14.8 | -10.2 / -15.0 | -12.0 / -15.0 | -13.4 / -15.0 |
| 20 | -1.9 / -2.5 | -3.9 / -17.6 | -6.7 / -18.8 | -9.6 / -19.9 | -12.6 / -19.9 | -13.8 / -19.9 | -16.1 / -20.0 | -18.2 / -20.0 |
| 30 | -4.0 / -4.9 | -6.5 / -25.6 | -10.9 / -28.1 | -14.9 / -29.0 | -18.1 / -29.6 | -20.9 / -29.9 | -24.4 / -30.0 | -26.7 / -30.0 |
| 40 | -4.2 / -5.8 | -9.8 / -34.1 | -15.8 / -37.3 | -21.1 / -39.0 | -25.5 / -39.6 | -27.9 / -39.6 | -32.3 / -40.0 | -36.0 / -40.0 |

Verdicts over 48 cells: WORSE 1, better 36, tie 1, unclear 10

Seeds where the two trackers' ID histories diverge: 951 of 960. Reasons in the trace at the first disagreement (tracker, reason → count): bytetrack/expired 558, ours/cost_above_threshold 389, ours/expired 217, bytetrack/taken_by_other_box 111, ours/taken_by_other_box 95, bytetrack/cost_above_threshold 50, bytetrack/transfer 17, ours/transfer 13

## Crossing recovery

[`CROSSING_RECOVERY.md`](CROSSING_RECOVERY.md) covers four things:
- the counting step that pairs an ID ending at the gate with a new ID starting across it;
- its limits (gap 4.0 s, distance 2.0 box heights, near 3.0 box heights), tuned on simulator seeds 0–9 only;
- the held-out checks: simulator seeds 1000–1019, MOT17-04, MOT17-09 and the MOT17-09 upload;
- why recovered crossings are listed but not counted by default.

# Crossing recovery: limits, how they were picked, held-out checks

**What it is** (`eval/crossing_recovery.py`). When the tracker loses a person while they are
hidden *at* the gate line, the old number ends on one side and a new number begins on the other.
Neither number crosses, so plain line counting misses that crossing. Crossing recovery runs after
the plain count, as a separate step. For each such pair it adds **one** crossing, built from the
old number's last foot point and the new number's first one. Track numbers on screen are never
changed. It is separate from hysteresis (the only double-count protection in this repo; there is
no "held back double" guard or ledger row on any branch).

A pair counts when all of these hold:
- number *a* ends with its foot point within `near_rel` box heights of the line (inside the
  segment);
- number *b* starts on the other hysteresis side, `0 < gap ≤ max_gap_s` later;
- the two foot points are at most `max_dist_rel` box heights apart;
- each end and each start is used once, with the closest in time paired first, then the closest
  in space.

## Chosen limits

| limit | value |
|---|---|
| `max_gap_s` | **4.0 s** |
| `max_dist_rel` | **2.0** box heights |
| `near_rel` | **3.0** box heights |

**How they were picked** (`python -m eval.crossing_recovery_tune tune`). The limits came from
**simulator tuning seeds 0–9 only**. MOT17-04 and MOT17-09 were never used, and neither were the
reported seeds 1000–1019.
- **Tuning cells:** stress grid people {5, 10, 20, 40} × hidden {0.5, 1, 2, 3, 5} s, plus the
  lookalike_cross, group, stop_and_go, step_back, linger and dropout_15f scenarios. Both trackers
  (ours default and ByteTrack) were run on each cell.
- **Search grid:** gap {0.5, 1, 2, 3, 4, 6} s × distance {0.5, 1, 1.5, 2, 3, 4} × near
  {0.5, 1, 1.5, 2, 3}.
- **Objective:** Σ over both trackers of (|gate count error| after recovery + false recoveries).
  A recovery is false when the ended and the new number belong to different true people (each
  number is assigned the person it matched most often). Ties go to the smaller limits.
- **Caveat:** `near_rel = 3.0` sits at the edge of the search grid.
- **On the tuning seeds:**
  - ours: |error| 2315 → 1478, with 955 recovered, of which 419 were false;
  - ByteTrack: |error| 3358 → 1633, with 1813 recovered, of which 880 were false.

## Held-out checks (not used to pick anything)

**Simulator, reported seeds 1000–1019** (full grid of 6 crowd sizes × 8 hidden times, plus all
scenarios):

| | gate \|error\| before → after | recovered | of which false |
|---|---|---|---|
| ours, grid | 11480 → 7339 | 4397 | 1929 (44 %) |
| ByteTrack, grid | 16582 → 8332 | 8380 | 4257 (51 %) |
| ours, scenarios | 381 → 147 | 306 | 78 |
| ByteTrack, scenarios | 990 → 176 | 896 | 263 |

On the grid, nothing is recovered at 4–5 s hidden. Those cells stay at their full error, because
the hide plus the ramp is longer than `max_gap_s`.

**MOT17 val_half, frozen gates** (`configs/gates/mot17_val_half.yaml`, h60/v50, hysteresis 10)
(`python -m eval.crossing_recovery_mot`). Each cell is IN/OUT; true values come from ground truth:

| sequence · gate | truth | ours before → after | ByteTrack before → after |
|---|---|---|---|
| MOT17-04 · h60 | 3/2 | 3/2 → 3/2 (0 recovered) | 3/2 → 3/2 (0) |
| MOT17-04 · v50 | 1/0 | 1/0 → 1/0 (0) | 1/0 → 1/0 (0) |
| MOT17-09 · h60 | 0/4 | 3/2 → **5/4** (4 recovered; \|error\| 1 → 5) | 2/2 → **5/4** (5; \|error\| 0 → 5) |
| MOT17-09 · v50 | 1/2 | 1/2 → 2/2 (1; \|error\| 0 → 1) | 1/2 → 3/2 (2; 0 → 2) |

**Upload of MOT17-09** (job `7e62233a122c`; **superseded**: this job tracked the head-blurred
copy, see the operator demo below for the canonical raw-tracked run):
- Setup: the user-drawn gate at y = 0.649, CPU, YOLOX-S, our tracker. The true count is
  3 IN / 10 OUT, from gt.txt on the same line.
- Before: **0/7**. After: **4/12**, with 9 recovered, so the total |error| goes from 6 to 3.
- Only **3 of the 9** recovered crossings match a true crossing within 15 frames. Most of the
  improvement in the count is therefore coincidental: wrong events that happen to fill missing
  ones.

## Decision: moments worth checking, never a count

The method holds up on the simulator, where it was tuned, and on held-out seeds. It does **not**
hold up on real footage: on the frozen MOT17-09 gates it makes the count worse for both trackers,
and on the MOT17-09 upload only 1 of 8 suggestions matches a true crossing the tracker missed
(canonical raw-tracked run; 3 of 9 in the superseded head-blur run). The limits are relative to
box height, so the large boxes in MOT17-09 widen "near the line" to most of the frame. Box heights
in the simulator are 46–130 px; in MOT17-09 they are up to about 400 px.

Product call: a recovered crossing is a **moment worth checking**, never added to IN/OUT, and
there is no "adjusted" count on screen.
- The API emits `recovered_crossing` events with `from_track_id`. The job metrics report
  `possible_missed_crossings`, separate from `crossings_in` / `crossings_out`. There is no option
  to count them automatically.
- Studio step 4 shows one 32 px **Tracker count** card. Below the player, **"Moments worth
  checking (N)"** holds two groups, both in crossing-time order with no cap: "Possible missed
  crossings (n)" and "Unsure crossings (n)". Each row has a still, an mm:ss link, ID chips,
  crops, and Confirm / Dismiss; suggestion rows also show the pairing score in grey. Clicking the
  row plays the 2 s around the moment.
- Confirm / Dismiss / Undo are **review notes**: they change no number on screen and no row
  disappears. They are kept on the client (localStorage per job) and saved to the exports:
  "Crossings + review notes CSV" (`in_tracker_count`, `unsure`, `review_status` = pending /
  confirmed / dismissed) and "Review notes JSON" (every listed moment with its status).
- Why no adjusted count: suggestions can only add crossings, and false counted crossings stay in.
  On the canonical MOT17-09 run, confirming the one real suggestion moved IN 4 → 5, away from
  the true 3 (below). An earlier version of this branch showed an "Operator-adjusted" card; it
  was removed.
- A double-count guard is out of scope.

The limits were **not** re-tuned on MOT17 after seeing these results, because that would turn the
held-out check into tuning.

Raw numbers: `docs/results/stress/crossing_recovery.json`.

## Pairing score (a hint on each row, never an order or a filter)

Each suggestion carries `gap_s`, `distance_rel` and
`pair_score = 1 − ½·min(1, gap_s / max_gap_s) − ½·min(1, distance_rel / max_dist_rel)`:
1 means the new number started at once, on the spot; 0 means it sits at both limits. The score
is a hint shown as small grey text on each row. It never counts anything and never hides or
orders a row: Studio step 4 shows every suggestion, in crossing-time order, with no top-N cap.
On the superseded head-blur MOT17-09 upload run the real crossings ranked 2, 4 and 9 of 9 by score (below), so a top-3 cap
would have hidden two of the three real ones, and a score order would have put a false one
first.

## Operator demo on the MOT17-09 upload

`operator_demo_mot17_09.json` (`python -m eval.operator_demo`, ground truth offline only; truth
3 IN / 10 OUT). **Canonical run: job 03bcd2ab3ace**, detection and tracking on the raw upload
frames, face-only oval blur applied only to what is shown (the raw file was deleted when the job
ended):

| | Value |
|---|---|
| Tracker count | 4 IN / 11 OUT (truth 3 / 10) |
| Tracker crossings that are real | 9 of 15 (precision 0.60) |
| True crossings the tracker missed | 4 |
| Possible missed crossings (suggestions) | 8 |
| Real suggestions | 1 (pairing-score rank 2 of 8) |
| Adjusted count if the real one is added (recorded experiment, not shown in the UI) | 5 IN / 11 OUT |

Confirming the real suggestion moved IN 4 → 5, away from the true 3, because the tracker already
over-counts (6 of its 15 counted crossings are not real) and suggestions can only add. Adjusted
counts are not shown in the UI; the result is kept here only as a recorded experiment
(`adjusted_count_experiment` in the JSON). By crossing: 9 of the 15 counted crossings are real
and 4 true crossings are missed; of the 8 suggestions, 1 is one of those 4.

**Superseded** (both tracked on the *blurred* copy, so the blur changed the counts; kept in the
JSON under `superseded`, not to be quoted): head-square blur, job 367de74b7ea3: tracker 0 / 7,
9 suggestions, 3 real (score ranks 2, 4, 9), adjusted then 1 / 9. Face-oval blur, job
d90a457d1cab: tracker 3 / 9, 6 suggestions, 1 real (rank 1), adjusted then 4 / 9.

Why the list shows every suggestion in clip order, with no top-3 cap and no score sort: in the
superseded head-blur run the real ones ranked 2, 4 and 9 of 9 by score, so a top-3 cap would
have hidden two of three, and in the canonical run the top-scored suggestion is not real. The
score is a hint, not a filter; a reviewer still has to look at every row. (An earlier version
of this branch had the cap and the score order.)

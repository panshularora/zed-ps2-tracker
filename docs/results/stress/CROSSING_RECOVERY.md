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

**Upload of MOT17-09** (job `7e62233a122c`):
- Setup: the user-drawn gate at y = 0.649, CPU, YOLOX-S, our tracker. The true count is
  3 IN / 10 OUT, from gt.txt on the same line.
- Before: **0/7**. After: **4/12**, with 9 recovered, so the total |error| goes from 6 to 3.
- Only **3 of the 9** recovered crossings match a true crossing within 15 frames. Most of the
  improvement in the count is therefore coincidental: wrong events that happen to fill missing
  ones.

## Decision: suggestions an operator confirms, never an automatic count

The method holds up on the simulator, where it was tuned, and on held-out seeds. It does **not**
hold up on real footage: on the frozen MOT17-09 gates it makes the count worse for both trackers,
and on the MOT17-09 upload (head-blur run) only 3 of 9 recoveries match a true crossing. The limits are relative to
box height, so the large boxes in MOT17-09 widen "near the line" to most of the frame. Box heights
in the simulator are 46–130 px; in MOT17-09 they are up to about 400 px.

Product call: a recovered crossing is a **suggestion** for an operator, never added to IN/OUT.
- The API emits `recovered_crossing` events with `from_track_id`. The job metrics report
  `possible_missed_crossings`, separate from `crossings_in` / `crossings_out`. There is no option
  to count them automatically.
- Studio step 4 lists every one, in crossing-time order with no cap, under "Possible missed
  crossings (N)" (the "Unsure crossings (N)" list follows). Each row has a still, an mm:ss link,
  `#a → #b` chips, crops of both IDs, the pairing score in grey and Confirm / Dismiss. Clicking
  the row plays the 2 s around the moment.
- IN/OUT stays the tracker's plain count. Confirmed suggestions go into a separate,
  labelled **Operator-adjusted** count. Dismissed suggestions disappear from the list.
- Decisions are kept on the client (localStorage per job).
- "Crossings + your decisions CSV" lists every crossing and every suggestion, with an
  `operator_decision` column (confirmed / dismissed / pending) and whether the row is in the plain
  count and in the operator-adjusted count.
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
On the head-blur MOT17-09 upload run the real crossings ranked 2, 4 and 9 of 9 by score (below), so a top-3 cap
would have hidden two of the three real ones, and a score order would have put a false one
first.

## Operator demo on the MOT17-09 upload

`operator_demo_mot17_09.json` (`python -m eval.operator_demo`, ground truth offline only; truth
3 IN / 10 OUT). The tracker runs on the blurred copy of an upload, so the blur changes the
result:

| Upload blur | Tracker count | Plain matched to truth | Suggestions | Real | Real ranks by score | Operator-adjusted (real ones confirmed) |
|---|---|---|---|---|---|---|
| Head squares (earlier run, job 367de74b7ea3) | 0 IN / 7 OUT | 6 of 7 | 9 | 3 | 2, 4, 9 | 1 IN / 9 OUT |
| Face-only oval (current, job d90a457d1cab) | 3 IN / 9 OUT | 8 of 12 | 6 | 1 | 1 | 4 IN / 9 OUT |

With the face-only blur the tracker counts more crossings, but 4 of its 12 do not match a true
crossing (8 of 12 do). So its IN of 3 is not three real INs, and confirming the one real
suggestion (an IN) takes the operator-adjusted IN to 4, one above the true 3. On the head-blur run the real ones ranked 2, 4 and 9 of 9 by score: a top-3 cap would
have hidden two of three, and a score order would have put a false one first. That is why the
list shows all suggestions in clip order instead of the top 3 by score with a "Show N more"
toggle (an earlier version of this branch had that cap and order). The score does not reliably
surface the real ones, and an operator still has to look at every row.

# Kept or swapped? Checking "came back with the same number"

When a tracker ID disappears for some frames and the **same ID** is drawn again, the web app
used to show a 1 s **"kept #N"** chip. The tracker can, however, hand the number to a
*different* person (identity transfer, a "false keep"). On the subway-turnstile clip
`#4` did exactly that: last seen at tracked frame **95** on a woman in a white top with long
hair, then drawn again at frame **128** on a different woman in an orange top. The chip said "kept #4".

Now the chip only says **"kept #N"** when a check agrees it is the same person. Otherwise it says
**"possible swap: #N given to someone else"**, there is no dashed "still #N (hidden)" bridge over
that gap, the timeline shows a break with a `swap? #N` tag, and the summary counts the person as
**uncertain**. With no verdict at all the chip says `#N back (not checked)`.

Code: `scripts/demo/verify_keeps.py` (writes `web/public/verify/<clip>.json`),
`web/src/identity/verify.ts` (UI). Re-run: `python -m scripts.demo.verify_keeps`.

## Pexels clips: a heuristic, not ground truth

There are no labels on these clips, so this is a **heuristic check**. It can be wrong both ways.

For each reappearance (ID last seen at frame `a`, seen again at frame `b`):

1. **Appearance.** HSV colour histogram (12 hue × 4 saturation × 3 value bins) of the torso
   region of the box (x 20–80 %, y 18–60 % of the box; faces are blurred in the demo cuts, torso
   colour is not). Histograms are averaged over up to **5** visible frames before the gap and 5
   after, decoded from the blurred demo cut at half of 720p. Similarity = Bhattacharyya
   coefficient. **Pass if ≥ 0.80.**
2. **Path.** Constant-velocity extrapolation of the box bottom-centre from the last ≤ 10 frames
   before the gap to frame `b`. Error in box heights. **Pass if ≤ 0.6 + 0.9 × gap seconds.**
3. **Size.** Box height ratio after/before (or before/after). **Pass if ≤ 1.6.**

All three pass → **kept**. Any fails → **possible swap** (counted as uncertain). The repo has an
OSNet re-ID embedder (`models/osnet_embedder.py`), but it needs torch and weights that are not
on the CPU demo box, so the check uses colour histograms.

### How the thresholds were chosen (`scripts/demo/calibrate_keeps.py` → `docs/results/keep_check_calibration.json`)

* Appearance, on the Pexels cuts: 333 "same ID, 15 frames later, no gap" pairs vs 437 "different
  ID at the same later frames" pairs. Medians 0.898 vs 0.533. At 0.80, **16.5 %** of the same-ID
  pairs fall below (so the check is cautious, sometimes calling a real keep "possible swap") and
  **7.3 %** of different-person pairs pass. Neither set is ground truth.
* Path + size, on MOT17 with human labels (53 reappearances with a known answer): all **44**
  real keeps pass, but all **9** real swaps pass too. **The path check alone does not catch
  swaps**; it only rules out impossible jumps. The appearance signal does most of the work, and it
  could not be validated against labels because MOTChallenge frames are not in this repo.

## MOT17 (human labels): ground truth instead

On MOT17-04 and MOT17-09 the box before and after the gap is matched to the labelled person
(IoU ≥ 0.5). Same person → kept; different → swap; no match → unknown. The MOT17 "ours" export is
plain BoT-SORT (our visibility variant was not exported for those clips). No MOT20 tracker
export exists yet, so there is nothing to check on MOT20.

## Results (reappearances: kept / possible swap / unknown)

| Clip | Method | Ours | ByteTrack |
|---|---|---|---|
| subway_turnstile | heuristic | 26 / 16 / 0 | 31 / 16 / 0 |
| milan_turnstiles | heuristic | 1 / 0 / 0 | 1 / 0 / 0 |
| mexico_city_entrance | heuristic | 10 / 0 / 0 | 10 / 1 / 0 |
| rabat_stadium | heuristic | 3 / 3 / 0 | 3 / 5 / 0 |
| shibuya_crossing | heuristic | 67 / 71 / 0 | 65 / 84 / 0 |
| man_entering_metro | heuristic | 0 / 0 / 0 | 0 / 0 / 0 |
| MOT17-09 | ground truth | 11 / 2 / 0 | 11 / 2 / 1 |
| MOT17-04 | ground truth | 11 / 1 / 0 | 11 / 4 / 0 |

Turnstile `#4` (frames 95 → 128): appearance 0.731 (< 0.80), size ×1.61 (> 1.6), path 0.19 ≤ 1.59
→ **possible swap**. Its earlier gap (33 → 75) passes (0.911, path 0.84 ≤ 1.86, ×1.31).
Regression tests: `tests/test_verify_keeps.py`, `web/tests/keep_check.test.tsx`.

## Story moments (Scenarios / Studio)

Listed in `web/src/identity/moments.ts`, shown only if the check file says "kept"; captions are
built from the file. Before/after crops of the Pexels ones were also looked at by eye.

| Clip | Person | Hidden | Check |
|---|---|---|---|
| subway_turnstile | #10, behind #20, frames 63–85 (0.8 s) | back still #10 | heuristic: 0.926, path 0.31 |
| subway_turnstile | #13, behind #15, frames 99–117 (0.6 s) | back still #13 | heuristic: 0.911, path 0.07 |
| mexico_city_entrance | #1, missed by the detector, frames 404–423 (0.7 s) | back still #1 | heuristic: 0.975, path 0.03 |
| rabat_stadium | #9, missed by the detector, frames 91–99 (0.3 s) | back still #9 | heuristic: 0.973, path 0.12 |
| MOT17-09 | #14, behind #7, frames 215–234 (0.7 s) | back still #14 | ground truth: labelled person 3 both sides |

The "behind #M" occluder is the other ID whose box covers the hidden person's interpolated box
most during the gap (≥ 30 % on average); otherwise the caption says "missed by the detector".
Failures stay visible: every flagged swap is listed under the player and marked on the timeline.

# Full review · zed-ps2-tracker · 2026-10-17

Scope: `main` at `7559df8` ("Keep-ID: one body, one number in Gate scene and Studio"), compared with
the fixes on `feat/review-stress`. Line numbers marked **main:** refer to `7559df8`; everything else
refers to the branch. Everything here was run on the box CPU. No GPU numbers are quoted; the MOT17
figures below are the public-FRCNN val-half CPU runs (`make eval`), the same setting as the
ablation table.

## 1. Test results

| Check | main `7559df8` | feat/review-stress |
|---|---|---|
| pytest (`python -m pytest -q`, as `make test-py`) | 274 passed | 297 passed (+23: upload/gate/CSV/annotated/unsure/reappearance tests and the stress suite; the 2 Keep-ID tests are gone) |
| ruff check | **7 errors** (`tests/test_gate_sim.py` PT018×4 + RUF015, `tests/test_keep_id.py` B007, `tests/test_dgx_helpers.py` E501) | clean |
| ruff format --check | **3 files would be reformatted** | clean |
| mypy --strict tracker | clean | clean |
| mypy eval models serve | **1 error**: `serve/gate_sim.py:288` `_make_tracker` missing return type | clean |
| vitest | 121 passed (17 files) | 119 passed (18 files; keepId.test.tsx removed, upload_studio + your_video added) |
| tsc --noEmit | clean (only after `npm ci` in `viewer/`: `web/` imports `viewer/src` and needs its `node_modules`) | clean |
| eslint | clean | clean |
| vite build | builds (389 kB JS) | builds (398 kB JS) |

So `make check` (what CI is meant to run) **fails on main** (ruff check, ruff format, mypy).

## 2. Architecture in one paragraph

`tracker/` is a from-scratch ByteTrack/BoT-SORT re-implementation (`ByteTrackLike`) with a
pluggable visibility estimator (`tracker/visibility.py`, ours = `DepthOrderVisibility`: box overlap
with closer boxes × detector score) that only changes **how long a lost track is remembered**
(`occluded_buffer_scale`, `tracker/bytetrack.py` lifecycle). `eval/` runs MOT17/MOT20 through the
vendored TrackEval. `serve/` is the FastAPI product: clips (only blurred copies served), uploads →
face blur → YOLOX on CPU → tracker → `serve/analysis.py` (gate events, "recoveries", heuristic
switch flags). `serve/gate_sim.py` is the synthetic gate scene streamed to the web app. `web/` is
React/Vite: landing, Gate scene, Scenarios, Studio (`/dashboard/{studio,compare,evidence,experiments}`),
Results, Analytics.

## 3. Findings

### P0

1. **Keep-ID relabelling was active by default on main** (Research Lab: the display must never fix
   identity).
   * Gate scene, "One body, one number" play (the default tab): **main:** `serve/gate_sim.py:39`
     labels `ours` as `keep_id` and **main:** `serve/gate_sim.py:289-290` swaps in `KeepIdTracker`, a
     separate nearest-centre tracker that extends lost tracks instead of minting new IDs. So the
     pane called "Ours" was not our tracker.
   * Studio step 1 and Scenarios: **main:** `web/src/guided/model.ts:77`
     `applyKeep = opts?.keepId ?? m.kind === "illustrative"` → every illustrative (Pexels) clip ran
     `collapseRenumbers` (**main:** `web/src/identity/keepId.ts`), rewriting ours' new numbers back to
     the old ones, and `keepIdVerdict` (**main:** `keepId.ts:178`) returned `verdict: "kept"` for
     every reappearance. Captions said "Illustrative only: Keep-ID" in small print
     (**main:** `GuidedStudio.tsx:51`, `Scenarios.tsx:115`), but the boxes looked like ours.
   * **Fixed on branch** (commit "Remove Keep-ID relabelling"): deleted `tracker/keep_id.py`,
     `tests/test_keep_id.py`, `web/src/identity/keepId.ts`, `web/tests/keepId.test.tsx`; the gate scene
     runs `configs/ours/visibility_botsort.yaml` in every play (`serve/gate_sim.py:35-40`,
     `_make_tracker`), Studio/Scenarios show ours exactly as tracked (`web/src/guided/model.ts`
     `fromScenario`), recordings regenerated. Honest consequence, now pinned in
     `tests/test_gate_sim.py::test_keep_play_runs_the_real_ours_tracker_no_relabel`: on the scripted
     seed-7 hide **both** trackers give the hidden person a new number (ours too), because the hidden
     person's last box was still ≥ 0.7 visible, so the visibility rule never extended its memory.
2. **No upload in the main flow.** **main:** `web/src/App.tsx:75` redirects `/dashboard/demo` to
   Scenarios; `web/src/pages/Demo.tsx` (the only page wired to `POST /api/uploads` + jobs) is not
   routed; Studio step 4 ("Drop your video", **main:** `Experiments.tsx:118-120`) says it "does not
   find people in a new recording" and only overlays saved tracks or runs a Stage-A `.npz` cache with
   a typed gate. **Fixed:** step 4 is now "Your video"
   (`web/src/workstation/screens/YourVideo.tsx`, mounted in `Experiments.tsx`), the old tools are under
   "Advanced".
3. **Uploads counted IN 0 / OUT 0 on many clips**: no way to set the gate, so every upload used the
   level line at 60 % height (`serve/pipeline.py:162-164` `gate_from_norm(None) → "h60"`), which on a
   real scene often misses the walking direction (API E2E on main: Shibuya 0/0, Yonge-Dundas 0/1).
   **Fixed:** `GET /api/clips/{id}/first-frame.jpg` (face-blurred frame 0, original still never
   served, `serve/api.py` `first_frame`), the user clicks two points, the job carries `gate`; tested
   end to end (`tests/test_serve_jobs.py::test_user_drawn_gate_counts_crossings`,
   `web/tests/your_video.test.tsx`).
4. **"2 people, 4 kept their number"** is a unit error: reappearance counts are **per event**, shown
   next to **per person** counts.
   * `serve/analysis.py` `metrics()` → `id_preserving_recoveries = len(rec)` counts every
     reappearance; one track that drops out three times is 3. API E2E on main: subway clip 40 tracks
     / 63 "recoveries", Shibuya 136 / 375, Yonge-Dundas 64 / 98. **main:** `Demo.tsx:423` shows it as
     "ID-preserving recoveries" beside the track count.
   * Studio Compare: **main:** `RealCompare.tsx:32-38` `backConfirmed = counts(doc).kept` counts all
     reappearance events in the check file, including those of brief tracks that are not listed as
     people, under a "People N" counter.
   * With Keep-ID on, `keepIdVerdict` made every one of those events "kept".
   * **Fixed:** backend adds `reappeared_ids` (distinct IDs, ≤ tracks;
     `test_reappearances_are_events_not_people`); RealCompare counts only listed people's
     reappearances and labels them as events (`runSummary`, test in `keep_check.test.tsx`); the upload
     page never shows a kept tally (timeline summary says "N track numbers on screen · no ground
     truth"); the stress tally is `kept ≤ people` by construction (`eval/stress_metrics.py` `tally`).
5. **Upload result could not open in a Studio-style view** (no per-person timeline, no unsure list,
   no downloads; neutral clips never computed unsure crossings: `serve/analysis.py` `gate_output`
   returned bare crossings). **Fixed:** result opens in the Studio `IdentityPlayer` with per-number
   timelines, IN/OUT at the user's line, an "Unsure moments" list (the `ambiguous_hold` rule; on
   neutral clips crossings carry `unsure` + `reasons`) that seeks to each moment, CSV
   (`/api/jobs/{id}/tracks.csv`), annotated clip (`/api/jobs/{id}/annotated.mp4`, `serve/render.py`),
   labels "Your clip · no ground truth" and "Re-identification off on CPU"
   (`result.tracker_info.reid = false`).

### P1

6. **Ours is worse than ByteTrack on the shipped crowd recording.** `docs/results/synthetic_gate/seed7.json`
   (crowd play, seed 7, 30 s): ID changes ours 25, BoT-SORT 26, ByteTrack 21; counted 25/34 for all.
   One seed, but it is the one on screen; the gate-scene copy on main promised ours keeps the number
   (finding 11).
7. **Ours' advantage is a memory-length effect only, and it ends at 3 s.** The visibility rule only
   multiplies the lost-track buffer (30 → 90 frames) when the last box was < 0.7 visible
   (`tracker/bytetrack.py` lifecycle loop, `occluded = t.visibility < occluded_thresh`). Stress grid
   (section 4 and `docs/results/stress/README.md`): ours is clearly better for hides of 1–2.5 s, but
   for hides ≥ 3 s both trackers give essentially every hidden person a new number and ours is
   slightly worse than ByteTrack (base config: 14 of 48 grid cells "ours worse", all at 0.5 s / n20 or
   ≥ 3 s). Doubling the occluded memory (`configs/ours/visibility_botsort_mem6.yaml`, tuned on seeds
   0-9, no MOT17 regression) turns 13 of those 14 cells into "better"/"unclear"; it is opt-in on the
   branch, not the default. A person who vanishes abruptly (no partial-occlusion ramp) never gets the
   longer memory: in the dropout scenarios ours is not better than ByteTrack (5/15/30-frame dropouts:
   +0.3/+0.4/+0.4 ID changes per clip, verdict "unclear").
8. **Gate counting needs the same ID on both sides of the line.** When the ID changes while the person
   is hidden at the line, neither the old nor the new number crosses, so the crossing is lost; at
   hides ≥ 3 s gate error ≈ −(number of people) for both trackers (stress grid). `eval/gate_events.py`
   `crossings()` has no track-stitching across the line.
9. **"ID-preserving recovery" is a claim without ground truth.** `serve/analysis.py:45` `recoveries()`
   labels every reappearance of the same number as ID-preserving; on a real clip that number can be
   on a different person. Kept for API compatibility, but no UI should present it as "kept".
10. **CI does not run.** `.github/workflows/ci` has no `.yml/.yaml` extension, so GitHub Actions
    ignores it. Not changed here (the brief forbids pushing workflow files); rename to `ci.yml` on
    main.
11. **Gate-scene copy promised the outcome** ("Ours must keep the same number", "Same body keeps the
    same number from approach to exit", **main:** `GateScene.tsx` and `lib/gateSim.ts` `beatCopy`).
    Rewritten to describe the rule and to say the outcome is shown, not assumed.
12. **Overview row "Keep-ID (gate scene)… BoT-SORT + visibility is not the keep-ID contract"**
    (**main:** `Overview.tsx:20-21`) contradicted the product story. Replaced with one row for ours.

13. **Upload counts are well below truth on the one clip that has truth.** MOT17-09 through the
    upload path (CPU YOLOX-s, user line at 65 % height): counted 0 IN / 7 OUT, truth 3 IN / 10 OUT,
    3 crossings attributed to the right person; 45 IDs for 26 people
    (`docs/review/UPLOAD_E2E_2026-10-17.md`). The upload page rightly shows no accuracy, but nothing
    in the product should present upload counts as reliable.
14. **The gate editor gives no hint where people's foot points are.** Counting uses the box
    bottom-centre (`eval/gate_events.py:75` `bottom_centre`); on Shibuya the foot points all sit at the
    bottom edge and people walk sideways, so a natural horizontal line counts 0 / 0 (a vertical line:
    0 / 4). Show foot-point trails on the first frame or warn when no track crosses the line.

### P2

15. `serve/blur.py` `blur_video`: on cancel the `.tmp.mp4` is left behind (`enc.kill()` then raise);
    harmless (never served) but leaks disk.
16. `web/src/pages/Demo.tsx` is dead code (not routed); its metrics panel is the "recoveries" display
    in finding 4. Superseded by "Your video"; delete in a follow-up.
17. `ByteTrackLike.update`: a detection whose score equals `high_thresh` is in neither the high nor
    the low set (`s > high` / `s < high`). Same as upstream ByteTrack, so kept for parity; noted.
18. `tsc` in `web/` silently depends on `viewer/node_modules` (imports `viewer/src`); document it in
    the Makefile / CI (`npm ci` in both).
19. Lint debt on main (section 1). Fixed on the branch.
20. `serve/gate_sim.py` "keep" play is a two-person script designed around ByteTrack's 30-frame
    buffer; it is a demo of one failure, not evidence. The stress grid is the evidence.
21. "Your video" does not resume a running job after a page reload (state is component-local); the
    backend job keeps running and finishes, the user just loses it.
22. Two "track" counts: the upload page shows every ID drawn (Shibuya 167), `metrics.tracks` counts
    IDs with ≥ 3 frames (`serve/analysis.py` `min_track_frames`, 136). Label one of them differently.

## 4. Stress suite (one person, one ID)

Generator `eval/stress_sim.py`, runner `eval/stress_run.py`, scoring `eval/stress_metrics.py`, tests
`tests/test_stress_suite.py`. Grid 8 hidden durations × 6 densities × 20 seeds (seeds 1000-1019;
tuning used seeds 0-9), plus 10 scenario kinds × 20 seeds and a noise sweep. Results:
`docs/results/stress/stress_map.json` / `.csv` (ours = `visibility_botsort.yaml`),
`stress_scenarios.json`, `stress_noise.json`, and the `_mem6` variants for the candidate change.
Tables and the before/after are in `docs/results/stress/README.md`.

Headline (base config, 48 cells, verdict = bootstrap 95 % CI of per-seed ours − ByteTrack ID changes
excludes 0 **and** sign test p < 0.05): ours better in 24 cells (every density at 1–2.5 s; e.g.
10 people, 1 s hidden: 3.5 vs 10.5 ID changes per clip, strict kept 6.6 vs 0.0 of 10, IDF1 range
0.71–0.94 vs 0.49–0.56), worse in 14 (20 people at 0.5 s: 8.4 vs 7.3; most ≥ 3 s cells), tie 4,
unclear 6. With mem6: better 36, worse 1 (20 people, 0.5 s, unchanged), tie 1, unclear 10. Gate
count error is negative everywhere for both trackers (finding 8). In 951 of 960 grid seeds the two
trackers' ID histories diverge; the stored trace at the first divergence says why (most often
ByteTrack "expired": the 30-frame memory ran out while the person was hidden).

## 5. Upload E2E

See `docs/review/UPLOAD_E2E_2026-10-17.md`.

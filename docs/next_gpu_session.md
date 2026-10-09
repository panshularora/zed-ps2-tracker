# Next GPU session (DGX, 5 Oct 2026): what to run, what comes back, what it unblocks

Phase 2 ran on CPU only, with public detections and no embeddings. The CPU-side tooling for
phase 3 is ready and tested on synthetic bundles (`make phase3-selftest`). What is still missing
is the real Stage A output.

**Runbook (step by step, checked): [dgx_runbook.md](dgx_runbook.md).** Background:
[dgx_guide.md](dgx_guide.md).

## On the DGX

1. Run `setup.sh` and wait for **SETUP OK**. Then run `run.sh` per dataset (MOT17 ablation + full,
   MOT20, DanceTrack val, SOMPT22 if downloaded, the gate clip if recorded), then `finalize.sh`.
2. **Pack:** `bash scripts/dgx/pack.sh` writes uncompressed tars from `bring_back/`:
   - `core.tar` (manifest, SHA256SUMS, gt, logs)
   - one `<dataset>__<label>.tar` per cache folder
   - `clips.tar` (demo MP4s)

   Each tar has its own MANIFEST.json and SHA256SUMS. Copy them with `scp`, one at a time; the
   CPU box has about 5 GB free, so not everything fits at once.

## Back on the CPU box

3. **Ingest, one tar at a time** (core, MOT17 and MOT20 first). This checks sha256, validates
   the caches and the SAW_EVAL_HALF label, checks free disk, and rebuilds
   `data/stagea/index.json`:
   ```bash
   make ingest BRING_BACK=/workspace/uploads/zed-bring-back/<tar> DELETE=1
   ```
4. **Phase 3:** `make phase3`. It runs the frozen plan `configs/phase3/plan.yaml` and writes
   `docs/results/phase3/<id>/`:
   - the tune on MOT17 train_half (tau_update / tau_cost)
   - ablation tables for MOT17 val_half, MOT20 val_half and DanceTrack val
   - stress plots
   - the ByteTrack **reproduction check** (`reproduction.md`; paper: MOTA 76.6 / IDF1 79.3 /
     IDs 159, tolerance ±1.0 MOTA, ±1.5 IDF1)
   - the pre-registered **go/no-go v2** verdict (`go_no_go_v2.md`, rule in
     [preregistration/go_no_go_v2.md](preregistration/go_no_go_v2.md); without MOT20 the best
     possible verdict is "GO (MOT17 only)")

   Conditions without data are listed as "waiting". Re-run after each ingest.
5. **Gate events on the GPU runs (manual, not in `make phase3` yet).** Run `python -m
   eval.gate_events --gates configs/gates/mot17_val_half.yaml` on the MOT17 val_half tracker
   outputs under `outputs/phase3/`. These fill the ⟨GPU⟩ cells in
   [integration.md](integration.md) §3 and slides 5–7 of [deck_outline.md](deck_outline.md).
6. **Gate clip.** Freeze a line config for it *before* running trackers, then compute the gate
   events. MOT17 has only 25 crossings, too few to rank trackers.
7. **Viewer on real frames.** Copy the MP4s from `clips.tar` next to `viewer/dist/index.html`.
   Then export with `python -m eval.export_viewer ... --dets <cache dir> --media SEQ=clips/<seq>.mp4`
   (frame_offset comes from the cache), and re-shoot `docs/media/` (`npm run screenshots`).
8. **Video and deck:** pick ⟨OURS⟩ from the phase-3 tables, read the go/no-go v2 verdict (it
   decides the conditional "dashed hold" shot), and fill the placeholders in
   [demo_video_script.md](demo_video_script.md) and [deck_outline.md](deck_outline.md).

## Blocked on Panshul (not on the GPU)

- **CI workflow:** the token has no `workflow` scope, so add `docs/ci/github-actions.yml` as
  `.github/workflows/ci.yml` through the GitHub website.
- **Gate clip:** it needs recording, plus consent.
- **SOMPT22:** a manual download (Drive), extracted before `scp` (see runbook).
- **Mentor questions:** `03_mentor_questions_phase1.md` (in the project notes, not in this repo).

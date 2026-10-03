# Next GPU session: what to run and what it unblocks

Everything in phase 2 was done on CPU with public detections and no embeddings. The list below
is ordered by how much each item unblocks. Runbook: [dgx_guide.md](dgx_guide.md).

1. **Stage A caches.** YOLOX-X detections (ByteTrack `bytetrack_ablation` / `bytetrack_x_mot17`
   / MOT20 checkpoints, score ≥ 0.05) plus OSNet-AIN re-ID embeddings for MOT17, MOT20,
   DanceTrack val, SOMPT22 and the gate clip. Bring back `bring_back/` and check
   `sha256sum -c SHA256SUMS`.
2. **Reproduction check.** Run ByteTrackLike on the YOLOX ablation caches (MOT17 val-half) and
   compare with ByteTrack's README (MOTA 76.6 / IDF1 79.3). Expect a small gap from the xywh KF
   and from not using `--post` (see implementation_notes.md).
3. **Re-run `make ablate` and the go/no-go protocol (`python -m eval.go_no_go`) on the real detections** (add YOLOX conditions to its condition list), with
   the re-ID baselines (`configs/baselines/*_reid.yaml`), and with Deep OC-SORT embeddings switched on. Report the
   numbers next to the public-detection ones, labelled by detector.
4. **Visibility-gated appearance (untested so far).** Sweep `appearance.min_visibility` (skip
   EMA gallery updates from occluded boxes) and `appearance.cost_min_visibility` (ignore the
   appearance cost for occluded detections) on train-half and report on val-half. This is the
   use the visibility estimate was designed for, and the one experiment that could show it does
   more than a longer buffer.
5. **Delayed look-ahead with re-ID.** Re-score the K-frame hypotheses with appearance as well as
   IoU and tighten the trigger (margin + visibility) so far fewer frames are flagged. Pre-register
   a go/no-go before running, as in #9. Without a clear IDSW win at p95 ≤ 20 ms, drop it.
6. **MOT20 and DanceTrack** with YOLOX caches (MOT20 public detections are unusable; see
   implementation_notes.md). Check delayed K > 0 runtime on MOT20 first.
7. **Gate clip.** Freeze a line config for the custom gate clip *before* running trackers, then
   run `make gates` on it. MOT17 has only 25 crossings, too few to rank trackers.
8. **Viewer on real frames.** Export frame JPEGs (or a downscaled MP4) for the demo clips and
   draw them under the boxes in `viewer/`. Today it draws on a blank canvas.

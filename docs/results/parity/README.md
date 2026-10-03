# Upstream parity of the baseline ports

`scripts/research/parity_upstream.py` gives the **same** MOT17 val-half public FRCNN detections
(7 sequences, 2,652 frames) to the upstream reference tracker and to our port. The output filter is
off on both sides. It then compares every frame's reported `(track id, box)` set, with boxes
compared at atol 1e-4 px (BoT-SORT: 1e-3 px on 4-decimal output).

| Port | Upstream (commit) | Settings compared | Frames that differ |
|---|---|---|---|
| `ocsort` | noahcao/OC_SORT `trackers/ocsort_tracker` (8462e7e) | det_thresh 0.6, IoU 0.3, Δt 3, inertia 0.2, no BYTE | **0 / 2652** |
| `deep_ocsort` | GerardMaggiolino/Deep-OC-SORT `trackers/integrated_ocsort_embedding` (6bb51d0) | as above + CMC from `cache/cmc_files` (file), `--emb_off --aw_off --new_kf_off --grid_off` (no embeddings in public dets) | **0 / 2652** |
| `hybrid_sort` | ymzis69/HybridSORT `trackers/hybrid_sort_tracker` (396f8d3) | ablation exp: HMIoU, IoU 0.25, inertia 0.05, BYTE, TCM 1.0/1.0 | **0 / 2652** |
| `botsort` | NirAharon/BoT-SORT `tracker/bot_sort.py` (2519854) | tools/track.py defaults, `--cmc-method file` (MOT17_ablation), no re-ID | **0 / 2652** |

What was stubbed or patched upstream, and why it doesn't affect the comparison:
* `linear_assignment` is patched to scipy's `linear_sum_assignment`, our backend. `lapjv` finds
  the same optimum but can break exact ties differently.
* `cv2`, `torch`, `torchvision`, `torchreid` and FastReID are stubbed. They are only used for
  images and re-ID, which are off here. BoT-SORT's file-mode GMC does not touch the image.
* `cython_bbox.bbox_overlaps` is a numpy re-implementation of the same pixel-inclusive (+1)
  formula.
* Hybrid-SORT needs numpy < 1.24, because it assigns a 1-element array into a row. The scratch venv
  uses numpy 1.23.5 / scipy 1.10.1.

Getting BoT-SORT to exact parity needed four switches on `ByteTrackLike`, all off by default and on
in `botsort.yaml`:
* `association.iou_plus_one`: cython_bbox IoU.
* `lifecycle.predict_tentative: false`: only tracked + lost tracks are predicted.
* `output.report_tentative`: BoT-SORT reports unconfirmed tracks.
* `lifecycle.removal_lag`: the `sub_stracks`/`removed_stracks` ordering quirk. A track marked
  removed can be re-found one frame later, and is then dropped the next time it is lost.

The duplicate-pruning rule was also fixed for every ByteTrack-family config. It now includes
tentative tracks, uses `last_frame − start_frame`, and judges pairs independently, as in ByteTrack
and BoT-SORT. ByteTrackLike's MOT17 val-half numbers did not change (HOTA 50.1, IDSW 191).

The JSON files here are the reports. `tests/test_baselines.py` asserts that they record 0
mismatches.

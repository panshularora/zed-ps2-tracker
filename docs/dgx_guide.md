# DGX guide: Stage A (detections + re-ID embeddings)

This is the copy-paste runbook for the one GPU session. Stage A runs the detector and the re-ID
model **once** over every sequence and writes a compact cache. Everything else (tracking,
TrackEval, stress bins, ablations) runs later on CPU from that cache.

> **Status.** The scripts were exercised end to end on the CPU box with `DEVICE=cpu
> VARIANT=yolox_s` and ByteTrack's real `bytetrack_s_mot17` + OSNet-AIN MSMT17 weights: the
> SOMPT22 path (with a synthetic archive), the custom-video path, resumability, the
> SOMPT22 test-format export and `finalize.sh`. On a real crowd photo the detector gave sensible
> person boxes and the same person's embedding across frames had cosine 0.97 against 0.52 for
> different people. **Not yet run:** anything on a GPU, the YOLOX-X checkpoints (the GPU self-test
> covers loading them), the full dataset downloads, `setup.sh` itself, and the optional MOT20
> fine-tune.

## 0. Before you go (2 minutes)

* `git clone https://github.com/panshularora/zed-ps2-tracker && cd zed-ps2-tracker`
* You need about **60 GB free** for everything (about 25 GB for MOT17 + MOT20 alone), plus `wget`,
  `unzip` and `git`. Use a scratch disk: `export ZED_ROOT=/raid/$USER/zed` (default
  `~/zed-ps2-stagea`).
* Optional: copy your gate clip (1080p30 MP4) onto the DGX.

## 1. Setup (one command, ~10–20 min, mostly downloads)

```bash
bash scripts/dgx/setup.sh
```

This does the following:

1. Checks the GPU, the driver and the free disk.
2. Creates a Python 3.11 env under `$ZED_ROOT/env`, using uv, conda or venv (whichever exists), with
   torch 2.7.1 (`cu126`, or `cu128` on Blackwell GPUs).
3. Clones ByteTrack (`d1bf019`) and torchreid (`f8cd150`) at pinned commits.
4. Downloads every checkpoint with resumable `wget -c` and checks its byte size. It records
   `weights/SHA256SUMS`, and the OSNet hash is checked against the value measured on the CPU box.
5. Runs `selftest.py`, which loads every YOLOX-X checkpoint and OSNet on the GPU and runs a dummy
   frame. The run ends with **`SELFTEST OK`**.

| Weight file | Source | Trained on | Used for | Leak-free for |
|---|---|---|---|---|
| `bytetrack_ablation.pth.tar` | ByteTrack README (Drive) | CrowdHuman + MOT17 train **first half** | MOT17 | MOT17 val-half ✅ |
| `sparsetrack_mot20_ablation.pth.tar` | SparseTrack README (Drive, file `bytetrack_20_ablation.pth.tar`) | CrowdHuman + MOT20 train **first half** (per README, unverified) | MOT20 | MOT20 val-half ✅ |
| `bytetrack_x_mot17.pth.tar` | ByteTrack README (Drive) | CrowdHuman + MOT17 (full) + CityPersons + ETHZ | MOT20 cross-domain, SOMPT22, videos | MOT20 ✅, SOMPT22 ✅, **not** MOT17 |
| `bytetrack_x_mot20.pth.tar` | ByteTrack README (Drive) | CrowdHuman + MOT20 train (full) | MOT20, labelled `SAW_EVAL_HALF` | ❌ (saw the eval half) |
| `dancetrack_bytetrack_x.pth.tar` | DanceTrack HF `noahcao/dancetrack_models` | DanceTrack train | DanceTrack val | DanceTrack val ✅ |
| `osnet_ain_x1_0_msmt17.pth` | torchreid model zoo (Drive) | MSMT17 (combineall) | every embedding | all ✅ (no MOT/DanceTrack/SOMPT22 data) |

We do **not** use BoT-SORT's FastReID `mot17_sbs_S50` / `mot20_sbs_S50` weights. They are trained
on MOT patches and may have seen the val halves (unverified).

If `setup.sh` fails, fix the cause and re-run it. Completed steps are skipped.

## 2. Run (one command per dataset; resumable)

Run these in priority order. Every command can be re-run after a crash or a dropped SSH session:
a finished sequence is skipped and partial files are never left behind. Use `tmux` so a dropped
SSH connection doesn't kill the run.

```bash
tmux new -s zed
bash scripts/dgx/run.sh mot17                      # 1st: MOT17 (headline)
bash scripts/dgx/run.sh mot20                      # 2nd: MOT20 (3 detector variants)
bash scripts/dgx/run.sh video /path/to/gate.mp4 gate   # your own gate clip
bash scripts/dgx/run.sh sompt22                    # SOMPT22 (see the manual download note)
bash scripts/dgx/run.sh dancetrack                 # DanceTrack val
bash scripts/dgx/run.sh pexels                     # optional demo clips
# or everything in one go (each target gets its own log; one failure does not stop the rest):
bash scripts/dgx/run.sh all /path/to/gate.mp4
```

| Target | Downloads | Sequences / frames | Detector runs (cache folder under `cache/`) | GPU time (est.) |
|---|---|---|---|---|
| `mot17` | MOT17.zip 5.9 GB (only the 7 `*-FRCNN` train sequences are extracted) | 7 / 5,316 | `MOT17/ablation/train` | 10–20 min |
| `mot20` | MOT20.zip 5.0 GB (train only) | 4 / 8,931 | `MOT20/sparsetrack_ablation`, `MOT20/mot17full_xdomain`, `MOT20/mot20full_SAW_EVAL_HALF` (+ `mot20half` if fine-tuned) | 1–2 h |
| `sompt22` | Drive folder (manual if gdown is refused) | 9 train + 5 test | `SOMPT22/mot17full/{train,test}` + baseline test-format results | 30–60 min |
| `dancetrack` | val.zip 4.2 GB | 25 val | `DanceTrack/dancetrack/val` | 30–60 min |
| `video` | – | your clip | `video/mot17full`, `video/ablation` | ~1 min per minute of video |
| `pexels` | 5 clips (one is 0.7 GB at 4K) | 5 | `video/mot17full` | 10 min |

GPU times are estimates. YOLOX-X at 1440 px with fp16 runs at roughly 25–40 fps on an A100-class
GPU, plus OSNet on every box. They have not been measured.

Notes:

* **All detections are kept down to `CONF_FLOOR=0.05`, each with its embedding.** Nothing is
  filtered by score on the GPU; only zero-area boxes (after clipping to the image) are dropped.
  The tracker chooses its own thresholds on CPU (ByteTrack uses 0.1).
* **Test splits of MOT17/MOT20 are skipped**: there is no ground truth and the MOTChallenge server
  is offline, so they are useless to us.
* **SOMPT22 manual step.** When tested from the CPU box, its Google Drive folder refused anonymous
  listing (HTTP 401). The script tries `gdown --folder` first. If that fails, it prints the
  instructions: download every file from the folder in a browser into
  `$ZED_ROOT/downloads/sompt22/`, then re-run `run.sh sompt22`. The archive layout is unverified,
  so `models/prepare.py` treats any `<seq>/img1/` folder as a sequence: one with `gt/gt.txt` goes to
  train, the rest to test. If an archive has no `seqinfo.ini`, one is written with
  `SOMPT22_FPS=30`; the paper says "mostly 30 fps", so check this. The run also writes
  `results/SOMPT22/bytetrack_like_test_results.zip` in the challenge's format. **Nothing is
  submitted.** The organisers allow one submission per week through a GitHub issue, so a
  submission is a separate, deliberate decision.
* **Optional MOT20 fine-tune** (untested): `bash scripts/dgx/finetune_mot20_half.sh` (≈1.5–3 h
  per 10 epochs on one GPU, estimated). It fine-tunes from `bytetrack_ablation` on MOT20 train
  first half only, then `run.sh mot20` picks up the new weights. Only do this if time is left over.
* Tunables (export before running): `GPU=1`, `BATCH=4` (if you hit OOM), `CONF_FLOOR=0.05`,
  `ZED_ROOT=...`.

## 3. Integrity check + bundle (one command)

```bash
bash scripts/dgx/finalize.sh
```

This checks every cache:

* it loads and passes the schema validator;
* its frame count equals `seqinfo.ini`;
* it has 512-d unit embeddings;
* its confidence floor is ≤ 0.1;
* the `SAW_EVAL_HALF` folder name agrees with the flag stored inside the cache.

It then builds `$ZED_ROOT/bring_back/` with hard links (no extra disk), `MANIFEST.json` and
`SHA256SUMS`, and prints the size of each category. It ends with **`INTEGRITY OK: N caches
verified`**, or a list of what is wrong.

## 4. What to bring back

```bash
rsync -avhP --partial dgx:$ZED_ROOT/bring_back/ ./zed-stagea-bring-back/   # resumable
cd zed-stagea-bring-back && sha256sum -c SHA256SUMS
```

Checklist (sizes are **estimates**: about 770 bytes per detection, measured on the CPU smoke run,
times an assumed 1.3–1.5 detections per gt box at conf ≥ 0.05):

| ✓ | What | Path in `bring_back/` | Est. size |
|---|---|---|---|
| ☐ | MOT17 cache (ablation detector) | `cache/MOT17/ablation/train/*.npz` | 0.1–0.2 GB |
| ☐ | MOT20 caches, 3 variants | `cache/MOT20/*/train/*.npz` | 3–4 GB (≈1.1 GB each) |
| ☐ | SOMPT22 caches | `cache/SOMPT22/mot17full/{train,test}/*.npz` | 0.5–1 GB |
| ☐ | DanceTrack val cache | `cache/DanceTrack/dancetrack/val/*.npz` | 0.2–0.4 GB |
| ☐ | Gate clip + Pexels caches | `cache/video/*/*.npz` | 0.1–0.5 GB |
| ☐ | Demo clips (720p H.264): MOT17-04, MOT20-02, the first SOMPT22 train sequence, the gate clip | `clips/*.mp4` | 0.2–0.3 GB |
| ☐ | gt + `seqinfo.ini` for every cached sequence (DanceTrack and SOMPT22 labels can't be downloaded separately) | `gt/**` | < 0.1 GB |
| ☐ | SOMPT22 baseline test-format results (not submitted) | `results/SOMPT22/...` | < 5 MB |
| ☐ | Run logs + per-sequence Stage A stats (fps, detections/frame) | `logs/*.log`, `cache/**/stage_a_log.jsonl` | < 5 MB |
| ☐ | Weight checksums | `weights_SHA256SUMS` | 1 KB |
| | **Total** | | **≈ 5–6.5 GB** |

The CPU box has about 10 GB free, so the whole bundle fits. If you're short on space or time,
bring back in this order: MOT17 → MOT20 `sparsetrack_ablation` → MOT20 `mot17full_xdomain` →
SOMPT22 → clips → DanceTrack → MOT20 `SAW_EVAL_HALF`. **Always bring `logs/` and `gt/`.** Do not
bring images, zips or weights.

## 5. Cache format (schema v1)

There is one compressed `.npz` per sequence. It is defined and validated in `tracker/io.py`.

| Key | dtype | Shape | Meaning |
|---|---|---|---|
| `frame` | int32 | (N,) | 1-based frame index of the **original** sequence, sorted |
| `tlwh` | float32 | (N, 4) | box in original-image pixels, clipped to the image |
| `score` | float32 | (N,) | YOLOX `obj × cls` confidence, every box ≥ `conf_floor` |
| `emb` | float16 | (N, 512) | L2-normalised OSNet-AIN embedding of that box |
| `meta` | str (JSON) | () | sequence, dataset, split, frame rate, length, image size, detector info (weights file, sha256, `trained_on`, test size, conf floor, NMS), re-ID info, **`detector_saw_eval_half`**, provenance (git commit, GPU, torch, runtime) |

Caches cover **full sequences**. The half split (train frames 1..N//2+1, val frames N//2+2..N,
renumbered from 1) is applied on CPU, so the same cache serves tuning on the train half and
reporting on the val half. To load one: `tracker.io.load_cache(path)`; `cache.frames()` yields
`(frame_id, Detections)` for every frame, including empty ones.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| download stops | re-run the same command (`wget -c` resumes) |
| `CUDA out of memory` | `BATCH=2 bash scripts/dgx/run.sh <target>` |
| `selftest` fails on `sparsetrack_mot20_ablation` | that checkpoint is unverified; carry on with the other two MOT20 variants and note it |
| `torch cannot see a GPU` | wrong CUDA wheel: `rm -rf $ZED_ROOT/env && TORCH_CUDA=cu121 bash scripts/dgx/setup.sh` |
| gdown refuses SOMPT22 | manual download (section 2), then re-run |
| `INTEGRITY ERRORS` | the message names the file; delete it and re-run its `run.sh` target |

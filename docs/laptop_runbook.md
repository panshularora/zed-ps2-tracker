# Laptop runbook: the whole GPU pipeline on Windows + WSL2 (RTX 3050 Laptop, 6 GB)

One command runs everything that needs the GPU and everything after it: detector + re-ID caches
for MOT17, MOT20 and DanceTrack, every tracker experiment with TrackEval, the ByteTrack
reproduction check, the frozen pre-registered go/no-go v2 verdict, and one results tar per
dataset. The results stay on the laptop. The command is `bash run.sh all --profile laptop`
([`scripts/dgx/run_all.sh`](../scripts/dgx/run_all.sh)). The DGX version is
[dgx_runbook.md](dgx_runbook.md).

> **What has been tested.** Nothing in this runbook has run on a GPU yet. Tested on a CPU box:
> `--dry-run`, the stage engine (skip, resume, `--force`, fail-fast, STATUS), and a full
> `run.sh all --profile laptop` with a stubbed GPU (fake `nvidia-smi`, fake torch, synthetic
> caches). That stubbed run included the CUDA out-of-memory fallback, ingest, `make phase3`, run-info
> stamping, the results tar and `make ingest` of that tar
> ([`tests/test_run_all.py`](../tests/test_run_all.py)). Not tested: `setup.sh`, the real
> detector, real datasets, and the real out-of-memory behaviour on 6 GB.
> **Run times are unknown.** Nothing has been measured on this card. After the first MOT17
> sequence, STATUS shows the measured frames per second and an estimate for the rest.

What the laptop profile does: YOLOX-X `bytetrack_ablation` at the standard 800x1440 input, FP16,
batch 1. This is the detector the ByteTrack paper used, so the reproduction check stays valid.
MOT17 runs first, and each dataset's results are finalized as soon as that dataset finishes. MOT20
caches only its two clean detector variants. It switches to the small YOLOX-S
(`bytetrack_s_mot17`) **only** after a real CUDA out-of-memory error. From then on it uses YOLOX-S
for that dataset and every later one, and every tracker (baselines included) runs on that
detector's output. Those rows are tagged *"laptop detector, not comparable to published
numbers"*, and the reproduction check and go/no-go v2 are marked **PENDING** instead of reported.
Every results file and the tar manifest record the detector, precision, input size, GPU name and
memory, and profile.

Every block below says where it runs: 🪟 **WINDOWS** (Settings or PowerShell) or 🐧 **UBUNTU**
(the WSL2 Ubuntu 24.04 terminal). It starts from what you already have: WSL2 Ubuntu 24.04 with
the repo cloned at `~/zed-ps2-tracker`.

## 1 · Windows: power, driver, WSL

🪟 **WINDOWS**
* Keep the laptop **plugged in** for the whole run.
* Settings → System → Power & battery → **Power mode: Best performance**. Set **"When plugged
  in, put my device to sleep after": Never**. A sleeping laptop pauses the run. Turning off only
  the screen is fine.
* The GPU reaches Ubuntu through the **Windows NVIDIA driver** (yours: 581.86). **Do NOT install
  an NVIDIA driver or `nvidia-*` / `cuda-drivers` packages inside WSL Ubuntu**, because they break
  the WSL GPU passthrough. The CUDA runtime comes with PyTorch.
* In PowerShell: `wsl --update` (once). Then reopen Ubuntu.

## 2 · Ubuntu: tools, Python, GPU check

🐧 **UBUNTU**
```bash
sudo apt update && sudo apt install -y git wget unzip tmux make build-essential curl
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt install -y nodejs   # Node 22 (Ubuntu's apt nodejs is 18, end-of-life)
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env   # uv brings Python 3.11
nvidia-smi                       # must list "NVIDIA GeForce RTX 3050 Laptop GPU"
df -h ~                          # want >= 40 GB free (refuses below 25 GB)
cd ~/zed-ps2-tracker && git pull
```
If `nvidia-smi` fails inside Ubuntu, fix it **on Windows**: update the NVIDIA driver, run
`wsl --update` and then `wsl --shutdown`, and reopen Ubuntu. Don't install a driver in Ubuntu.
Keep the repo and data under `~` (Linux disk). `/mnt/c/...` is several times slower.

## 3 · Environment, weights, GPU self-test (once)

🐧 **UBUNTU**
```bash
cd ~/zed-ps2-tracker
bash run.sh all --profile laptop --until gpucheck
```
This runs the preflight check and `setup.sh`: a Python 3.11 env with torch 2.7.1 cu126, the
pinned ByteTrack and torchreid, about 4 GB of weights, and a GPU self-test. Then it runs the torch
GPU check. It ends with the STATUS table and `state: DONE`. On a 6 GB card the self-test may
print `[OOM ]` for a checkpoint. That is allowed in the laptop profile. If torch cannot see the
GPU, the script itself rebuilds the env once with the CUDA 11.8 wheel (the documented fallback).

Everything lives in `~/zed-ps2-stagea` (`ZED_ROOT`): the env, downloads, data, caches, logs,
STATUS and the results.

## 4 · Download the datasets (≈15 GB, resumable)

🐧 **UBUNTU**
```bash
tmux new -s zed                                 # or: tmux attach -t zed  (a closed terminal would kill the download)
cd ~/zed-ps2-tracker
bash run.sh download mot17 mot20 dancetrack     # re-run the same line if it stops; wget resumes
```
MOT17 is 5.9 GB, MOT20 5.0 GB and DanceTrack val 4.2 GB. They are extracted after download.
`run.sh all` would also download anything missing, so this step is only so the GPU run doesn't
wait on the network. After extraction you can free about 15 GB with
`rm ~/zed-ps2-stagea/downloads/*.zip`.

## 5 · Dry run, then the real run in tmux

🐧 **UBUNTU**
```bash
cd ~/zed-ps2-tracker
bash run.sh all --profile laptop --dry-run      # prints every stage; runs nothing
tmux new -s zed
bash run.sh all --profile laptop
```
Detach with **Ctrl+b, then d**. The run keeps going, and you can close the terminal. Reattach
with `tmux attach -t zed`.

Stage order: preflight → setup → gpucheck → **MOT17** (Stage A → ingest → phase 3 → results
tar) → **MOT20** (same) → **DanceTrack** (+ SOMPT22 if its archives are in
`~/zed-ps2-stagea/downloads/sompt22/`) → final phase 3 → final tar.

## 6 · Check on it

🐧 **UBUNTU** (any terminal, no need to attach)
```bash
cat ~/zed-ps2-stagea/STATUS                 # state, every stage, notes (fps + ETA, fallback, verdicts)
tail -f ~/zed-ps2-stagea/run_all.log        # the live log (Ctrl+C stops tail, not the run)
nvidia-smi                                  # GPU busy?
ls -lh ~/zed-ps2-stagea/final/              # results tars so far
```
* **Failed?** STATUS says `FAILED at stage X` and the log names the command. Fix the cause and
  run **the same command again** (`bash run.sh all --profile laptop`). Finished stages are
  skipped, and Stage A keeps every finished sequence.
* **Redo something:** `bash run.sh all --profile laptop --force=phase3_mot17`. The stages after
  it are redone too. `--force` alone redoes all stages (finished cache sequences are still
  reused).
* **Out of memory:** handled automatically once (the switch to YOLOX-S, shown in STATUS notes).
  To force the small detector from the start: `--detector s`. To never fall back: `--detector x`.

## 7 · Results: ingest and view them

The first tar exists as soon as MOT17 is done; each later dataset adds a newer one.
`final/zed_results_latest.tar.gz` always points to the newest. Phase 3 has already run inside
`run.sh all`, so ingest just verifies and installs its report.

🐧 **UBUNTU**
```bash
cd ~/zed-ps2-tracker
make setup                                  # once: the repo's CPU venv (.venv, Python 3.11)
make ingest BRING_BACK=$HOME/zed-ps2-stagea/final/zed_results_latest.tar.gz
cat docs/results/phase3/README.md           # index: tables, reproduction, go/no-go v2 verdict
```
`make ingest` checks every file against the tar's SHA256SUMS and installs the report as
`docs/results/phase3/`. It also copies everything else (configs, logs, run info) to
`data/dgx_results/<tar name>/`. You don't need `make phase3`. (To recompute phase 3 from the
local caches anyway: `make phase3 STAGEA=$HOME/zed-ps2-stagea/stagea`. Add the `PHASE3_ARGS`
shown in the log if the YOLOX-S fallback happened.)

**Web app** (Results page):
```bash
cd ~/zed-ps2-tracker
make web && make serve                      # builds web/, serves it on port 8000
```
Open **http://localhost:8000** in the Windows browser (WSL forwards localhost). Results lists
the phase-3 files. `make demo` also works, as a dev server on :5173 with the API on :8000.

To copy the results tar to Windows: `cp ~/zed-ps2-stagea/final/zed_results_latest.tar.gz /mnt/c/Users/$USER/`
(or open `\\wsl$\Ubuntu-24.04\home\<you>\zed-ps2-stagea\final` in Explorer).

## Troubleshooting

| Problem | Fix |
|---|---|
| `nvidia-smi does not work inside WSL` | Windows side: update the NVIDIA driver, `wsl --update`, `wsl --shutdown`; never a Linux driver in Ubuntu |
| `a Linux NVIDIA driver package is installed inside WSL` | `sudo apt remove --purge 'nvidia-driver-*' 'nvidia-dkms-*'`, then `wsl --shutdown` |
| `torch cannot use the GPU even after the cu118 rebuild` | close other GPU apps (games, browsers with HW acceleration), reboot Windows, re-run |
| `only N GB free` | free space on C: (the WSL disk lives there); delete `~/zed-ps2-stagea/downloads/*.zip` after extraction |
| `another 'run.sh all' is already running` | it is: `tmux attach -t zed` |
| `the frozen files ... differ from git HEAD` | `git checkout -- configs/phase3/plan.yaml docs/preregistration configs/ours/delayed_v2_reid.yaml` |
| `$ZED_ROOT was run with profile=dgx` | use the same `--profile`, or `export ZED_ROOT=~/another-dir` |
| laptop slept / rebooted | re-run the same command; it resumes |

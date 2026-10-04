# DGX runbook for the Stage A session (Windows laptop)

This is the checked, copy-paste version of the day-of runbook. The reference with all the
background is [dgx_guide.md](dgx_guide.md); the scripts are in [`scripts/dgx/`](../scripts/dgx/).

Every block says **where** it runs:

* 💻 **LAPTOP**: Windows PowerShell on your laptop. Uses Windows' built-in `ssh` / `scp` (OpenSSH).
  There is no `rsync` on Windows and none is needed.
* 🖥️ **DGX**: inside the SSH session on `srmist1@172.16.0.32`.
* 🌐 **BROWSER**: your laptop browser.

Expect 3–5 h, mostly unattended. You need about **60 GB** free on the DGX (25 GB for MOT17 +
MOT20 alone) and about 7 GB on the laptop. All GPU times are estimates; nothing has run on a GPU
yet.

> `srmist1` looks like a shared account, so everything here lives under **your own folder**
> `~/zed-panshul` (code, env, data, caches), and the tmux session is called `zed-panshul` so it
> can't clash with anyone else's.

---

## Step 0 · Before you leave (optional)

🌐 **BROWSER:** if you want SOMPT22, open the
[SOMPT22 Drive folder](https://drive.google.com/drive/folders/1Z_gnFmX-EKUe4yLBQPa2pxXkyqYbxkhX)
and download everything into `C:\Users\Panshul\sompt22\`. Automatic download of this folder was
refused when tested, so this has to be done by hand. If Drive hands you a
`sompt22-<date>-001.zip` of the whole folder, **right-click → Extract All** into
`C:\Users\Panshul\sompt22\` first, so the folder holds SOMPT22's own archives. (The DGX script
only unpacks archives one level deep.)

Gate clip (if filmed): keep it at `C:\Users\Panshul\gate.mp4`.

## Step 1 · Get the code onto the DGX

💻 **LAPTOP:**
```powershell
cd C:\Users\Panshul
git clone https://github.com/panshularora/zed-ps2-tracker    # first time only
cd zed-ps2-tracker; git pull; cd ..                          # every time: latest scripts
ssh srmist1@172.16.0.32 "mkdir -p zed-panshul"
scp -r zed-ps2-tracker srmist1@172.16.0.32:zed-panshul/
```
The repo has a `.gitattributes` that keeps `*.sh` with Linux line endings even on a Windows
clone. If you cloned **before** 4 Oct 2026 evening, delete the old clone and clone again;
otherwise bash may fail with `$'\r': command not found`.

Optional, same window:
```powershell
scp C:\Users\Panshul\gate.mp4 srmist1@172.16.0.32:zed-panshul/
```
(Remote paths without `~/` are relative to your DGX home folder; this works with every Windows
`scp` version.)

## Step 2 · Log in and start a session that survives disconnects

💻 **LAPTOP:**
```powershell
ssh srmist1@172.16.0.32
```
🖥️ **DGX:**
```bash
tmux new -s zed-panshul
```
If Wi-Fi drops: SSH back in, then `tmux attach -t zed-panshul`; the jobs keep running.
If `tmux` is not installed, use `screen -S zed-panshul` (re-attach with `screen -r zed-panshul`).

## Step 3 · Pick a free GPU and set your paths

🖥️ **DGX** (inside tmux):
```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```
Pick the row with the lowest `memory.used` **and** ~0 % utilization. Put its `index` in place
of `3` below (plain number, no `<` `>`):
```bash
export GPU=3
export ZED_ROOT=~/zed-panshul
cd ~/zed-panshul/zed-ps2-tracker
df -h "$ZED_ROOT"
```
`df` should show well over 60 GB available (setup refuses to start below 25 GB).
The scripts set `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so `GPU=3` is the same GPU `nvidia-smi` calls 3.

> These two `export`s live only in this shell. If you open a new tmux window or log in again
> without re-attaching, run the three lines above again **before any other command**; without
> them the scripts fall back to GPU 0 and `~/zed-ps2-stagea`.

## Step 4 · Setup (≈10–20 min)

🖥️ **DGX:**
```bash
bash scripts/dgx/setup.sh
```
✅ The self-test prints **`SELFTEST OK`** and the script ends with **`SETUP OK`**. If it fails,
re-run it (finished steps are skipped).

If it says **`torch cannot see a GPU`** (old driver), rebuild the env with the CUDA 11.8 wheel
(torch 2.7.1 has wheels for cu118, cu126 and cu128 only; there is **no cu121** build):
```bash
rm -rf "${ZED_ROOT:?}/env" && TORCH_CUDA=cu118 bash scripts/dgx/setup.sh
```
If only `sparsetrack_mot20_ablation` fails the self-test, carry on; the other MOT20 variants still
run. Tell Build Lab.

## Step 5 · GPU jobs, most important first

🖥️ **DGX**, one at a time (each can be re-run after an interruption; finished sequences are
skipped):
```bash
bash scripts/dgx/run.sh mot17          # ~10–20 min + 5.9 GB download  (headline results)
bash scripts/dgx/run.sh mot20          # ~1–2 h + 5.0 GB download      (crowded benchmark)
bash scripts/dgx/run.sh dancetrack     # ~30–60 min + 4.2 GB download
bash scripts/dgx/run.sh pexels         # ~10 min                        (demo clips)
```
Gate clip, if you copied it:
```bash
bash scripts/dgx/run.sh video ~/zed-panshul/gate.mp4 gate
```
`CUDA out of memory` → prefix with `BATCH=2`, e.g. `BATCH=2 bash scripts/dgx/run.sh mot20`.
Someone else grabbed your GPU → `Ctrl+C`, pick another one with `export GPU=<n>`, re-run.

## Step 6 · SOMPT22 (only if you did step 0)

💻 **LAPTOP** (a second PowerShell window; the DGX keeps running):
```powershell
ssh srmist1@172.16.0.32 "mkdir -p zed-panshul/downloads"
scp -r C:\Users\Panshul\sompt22 srmist1@172.16.0.32:zed-panshul/downloads/
```
(Copying the folder itself avoids relying on `*` wildcards, which Windows `scp` handles
inconsistently.) Then 🖥️ **DGX:**
```bash
bash scripts/dgx/run.sh sompt22        # ~30–60 min
```
It also writes SOMPT22 test-format results, and **submits nothing**.

## Step 7 · Verify, bundle and pack

🖥️ **DGX:**
```bash
bash scripts/dgx/finalize.sh
bash scripts/dgx/pack.sh
```
✅ `finalize.sh` must end with **`INTEGRITY OK: N caches verified`**. On `INTEGRITY ERRORS`, the
message names the cache file: delete that `.npz`, re-run that dataset's `run.sh` line, then run
`finalize.sh` again.

`pack.sh` turns `bring_back/` into a few **uncompressed tar files** in
`~/zed-panshul/bring_back_tars/` (one per dataset/detector, each carrying the manifest), and
prints their sizes. A few big files copy much faster with `scp` than thousands of small ones,
and the CPU box can take them one at a time.

## Step 8 · Bring the results to the laptop

💻 **LAPTOP** (new PowerShell window):
```powershell
cd C:\Users\Panshul
scp -r srmist1@172.16.0.32:zed-panshul/bring_back_tars .\zed-bring-back
```
Short on time or Wi-Fi? Copy the essentials first, **in this order** (each is one file):
```powershell
mkdir zed-bring-back
scp srmist1@172.16.0.32:zed-panshul/bring_back_tars/core.tar .\zed-bring-back\
scp srmist1@172.16.0.32:zed-panshul/bring_back_tars/MOT17__ablation.tar .\zed-bring-back\
scp srmist1@172.16.0.32:zed-panshul/bring_back_tars/MOT20__sparsetrack_ablation.tar .\zed-bring-back\
scp srmist1@172.16.0.32:zed-panshul/bring_back_tars/MOT20__mot17full_xdomain.tar .\zed-bring-back\
```
then the rest (`ls` on the DGX shows the names). `core.tar` holds the manifest, checksums, gt
and logs: always bring it. Never copy images, zips or weights.

## Step 9 · Hand over to Build Lab

Tell Build Lab in your 1:1 chat that `C:\Users\Panshul\zed-bring-back` is ready. It copies the
tars to the CPU box one at a time (your laptop shows an approval prompt for each) and runs
`make ingest BRING_BACK=<tar>` on each, which checks every file against the DGX checksums. The
CPU box has little disk, so MOT17 and MOT20 go first and the rest follows.

## Step 10 · Clean up on the shared DGX

🖥️ **DGX:** `tmux kill-session -t zed-panshul` (only **your** session; don't kill others).
Keep `~/zed-panshul` until Build Lab confirms that everything arrived. After that,
`rm -rf ~/zed-panshul/data ~/zed-panshul/downloads` frees ~50 GB; keep `cache/` and `weights/`
if a re-run might be needed.

---

### Troubleshooting

| Problem | Fix |
|---|---|
| `syntax error near unexpected token` | you typed `<` or `>`; use a plain number |
| `$'\r': command not found` | the clone has Windows line endings: re-clone (step 1) and `scp` again |
| `No such file or directory` for the repo | step 1's `scp` didn't run, or you are not in `~/zed-panshul/zed-ps2-tracker` |
| caches land in `~/zed-ps2-stagea` | `ZED_ROOT` was not exported in this shell: step 3 again |
| download stopped | re-run the same command (`wget -c` resumes) |
| `CUDA out of memory` | `BATCH=2 bash scripts/dgx/run.sh <dataset>` |
| GPU suddenly busy | `Ctrl+C`, `export GPU=<other>`, re-run the same command |
| `torch cannot see a GPU` | `rm -rf "${ZED_ROOT:?}/env" && TORCH_CUDA=cu118 bash scripts/dgx/setup.sh` |

#!/usr/bin/env bash
# One-time Stage A setup on the DGX:  bash scripts/dgx/setup.sh
#   1. checks GPU / disk / tools      2. creates a Python 3.11 env with CUDA PyTorch
#   3. clones ByteTrack + torchreid at pinned commits   4. downloads + checksums all weights
#   5. GPU self-test (every checkpoint loads and runs on a dummy frame)
# Safe to re-run: every step is skipped when already done.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log setup

# ---------------------------------------------------------------- 1. checks
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: is this a GPU node?"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
for t in wget unzip git awk sha256sum; do command -v "$t" >/dev/null || die "missing tool: $t"; done
free_gb=$(df -BG --output=avail "$ZED_ROOT" | tail -1 | tr -dc 0-9)
log "free disk under $ZED_ROOT: ${free_gb} GB (need ~60 GB for everything, ~25 GB for MOT17+MOT20)"
(( free_gb >= ${MIN_FREE_GB:-25} )) || die "not enough disk under ZED_ROOT=$ZED_ROOT (${free_gb} GB < ${MIN_FREE_GB:-25} GB)"

# ---------------------------------------------------------------- 2. environment
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU" | head -1 | cut -d. -f1)
: "${TORCH_CUDA:=$([[ "$cap" -ge 10 ]] && echo cu128 || echo cu126)}"   # Blackwell needs cu128
if [[ ! -e "$ENV_DIR/.ready" ]]; then
  if command -v uv >/dev/null; then
    log "creating env with uv"; uv venv -p "$PYVER" "$ENV_DIR"
    PIP=(uv pip install -p "$ENV_DIR/bin/python")
  elif command -v conda >/dev/null; then
    log "creating env with conda"; conda create -y -p "$ENV_DIR" "python=$PYVER" pip
    PIP=("$ENV_DIR/bin/python" -m pip install)
  else
    py=$(command -v "python$PYVER" || command -v python3)
    log "creating env with $py -m venv"; "$py" -m venv "$ENV_DIR"
    PIP=("$ENV_DIR/bin/python" -m pip install)
  fi
  "${PIP[@]}" --upgrade pip wheel setuptools
  log "installing torch 2.7.1 ($TORCH_CUDA)"
  "${PIP[@]}" torch==2.7.1 torchvision==0.22.1 --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
  "${PIP[@]}" -r "$HERE/requirements-stage-a.txt"
  "${PIP[@]}" -e "$REPO"
  touch "$ENV_DIR/.ready"
fi
activate
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see a GPU"
print("torch", torch.__version__, "cuda", torch.version.cuda, "->", torch.cuda.get_device_name(0))
PY

# ---------------------------------------------------------------- 3. pinned sources
clone_pinned() {
  local url="$1" dir="$2" commit="$3"
  if [[ ! -d "$dir/.git" ]]; then git clone -q "$url" "$dir"; fi
  git -C "$dir" fetch -q origin "$commit" 2>/dev/null || true
  git -C "$dir" checkout -q "$commit"
  log "$(basename "$dir") @ $(git -C "$dir" rev-parse --short HEAD)"
}
clone_pinned "$BYTETRACK_REPO" "$SRC/ByteTrack" "$BYTETRACK_COMMIT"
clone_pinned "$TORCHREID_REPO" "$SRC/deep-person-reid" "$TORCHREID_COMMIT"

# ---------------------------------------------------------------- 4. weights
for entry in "${WEIGHT_LIST[@]}"; do
  IFS='|' read -r name url bytes <<<"$entry"
  fetch "$url" "$WEIGHTS/$name" "$bytes"
done
( cd "$WEIGHTS" && sha256sum ./*.pth* > SHA256SUMS && cat SHA256SUMS )
got=$(sha256sum "$WEIGHTS/osnet_ain_x1_0_msmt17.pth" | cut -d' ' -f1)
[[ "$got" == "$OSNET_SHA256" ]] || die "OSNet checksum mismatch ($got)"
a=$(grep bytetrack_x_mot17 "$WEIGHTS/SHA256SUMS" | cut -d' ' -f1)
b=$(grep dancetrack_bytetrack_x "$WEIGHTS/SHA256SUMS" | cut -d' ' -f1)
[[ "$a" != "$b" ]] || warn "DanceTrack checkpoint is byte-identical to bytetrack_x_mot17: tell the CPU side"

# ---------------------------------------------------------------- 5. self-test
# SELFTEST_ARGS=--allow-oom (set by `run.sh all --profile laptop`): an out-of-memory on one
# checkpoint is reported but does not fail setup; Stage A then falls back to YOLOX-S if needed
# shellcheck disable=SC2086
python "$HERE/selftest.py" --weights "$WEIGHTS" ${SELFTEST_ARGS:-}
log "SETUP OK. Next: bash scripts/dgx/run.sh mot17"

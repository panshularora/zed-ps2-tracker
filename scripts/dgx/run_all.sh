#!/usr/bin/env bash
# The WHOLE GPU pipeline in one resumable command (start it inside tmux and leave it):
#
#   bash run.sh all [--profile dgx|laptop] [options]          (= bash scripts/dgx/run.sh all ...)
#
#   preflight   GPU visible (nvidia-smi, WSL2 aware), tools, disk, datasets present/reachable,
#               frozen pre-registration files unmodified
#   setup       scripts/dgx/setup.sh: env, pinned sources, weights, GPU self-test
#   gpucheck    torch + CUDA versions; torch must see the GPU (else the documented cu118 rebuild)
#   stagea_*    detector + re-ID caches per dataset (MOT17, MOT20, DanceTrack; SOMPT22 if its
#               archives are there; --gate clip; --with-pexels)
#   ingest_*    integrity check (models.bringback) + `make ingest` into $ZED_ROOT/stagea
#   phase3_*    `make phase3`: tune, every tracker + TrackEval, stress plots, reproduction check,
#               the frozen go/no-go v2 verdict; gate events; every file stamped with run info
#   finalize_*  finalize.sh --results: ONE small tar + manifest + sha256 in $ZED_ROOT/final/
#
# Profiles:
#   dgx     (default unless WSL)  all Stage A first, then ingest/phase3/finalize once. BATCH=8.
#   laptop  (default under WSL)   FP16, BATCH=1, YOLOX-X bytetrack_ablation at 800x1440 (the
#           reproduction check stays valid). MOT17 first and each dataset's results are finalized
#           as soon as it finishes. Only a real CUDA out-of-memory error switches to YOLOX-S (that
#           and every later dataset; rows tagged "laptop detector, not comparable to published
#           numbers"; reproduction + go/no-go v2 pending). MOT20 caches only the two clean
#           variants (MOT20_VARIANTS to change).
#
# Options:
#   --profile dgx|laptop   --detector auto|x|s (auto: x; laptop falls back to s on CUDA OOM)
#   --dry-run              print the stages and what would run; execute nothing
#   --force | --force=a,b  redo all stages / the named stages (later stages are redone too)
#   --until STAGE          stop after STAGE        --skip mot20,dancetrack,sompt22
#   --gate PATH            also cache your gate clip    --with-pexels   also the Pexels clips
#   --gpu N  --batch N  --jobs N      (env GPU, BATCH, JOBS work too)
#   --fp32                 FP32 detector + re-ID instead of FP16 (env FP16=0)
#   (env) PHASE3_TIMEOUT_MIN_MOT20=MIN   go/no-go MOT20 replication timeout (laptop 240, plan 60)
#
# State: $ZED_ROOT/STATUS (cat it), $ZED_ROOT/run_all/ (done-markers, notes, gpu.json),
#        $ZED_ROOT/run_all/logs/run_all_<time>.log (+ $ZED_ROOT/run_all.log -> the latest).
BATCH_FROM_ENV="${BATCH:+1}" MOT20_VARIANTS_FROM_ENV="${MOT20_VARIANTS:+1}"   # before config.env defaults
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=pipeline_lib.sh
source "$HERE/pipeline_lib.sh"

: "${RUN_SH:=$HERE/run.sh}"           # tests point this at a stub
: "${ETA_WATCH:=1}"
usage() { sed -n 2,39p "$HERE/run_all.sh"; }

is_wsl() { [[ -n "${WSL_DISTRO_NAME:-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; }

# ------------------------------------------------------------------ arguments
PROFILE="${PROFILE:-auto}" DETECTOR_ARG="auto" UNTIL="" SKIP="" GATE="" WITH_PEXELS=0
PIPE_FORCE="" PIPE_FORCE_ALL=0 PIPE_DRY_RUN=0
parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --profile)    PROFILE="${2:?--profile needs dgx|laptop}"; shift 2 ;;
      --profile=*)  PROFILE="${1#*=}"; shift ;;
      --detector)   DETECTOR_ARG="${2:?--detector needs auto|x|s}"; shift 2 ;;
      --detector=*) DETECTOR_ARG="${1#*=}"; shift ;;
      --dry-run)    PIPE_DRY_RUN=1; shift ;;
      --force)      PIPE_FORCE_ALL=1; shift ;;
      --force=*)    PIPE_FORCE="${1#*=}"; PIPE_FORCE="${PIPE_FORCE//,/ }"; shift ;;
      --until)      UNTIL="${2:?--until needs a stage}"; shift 2 ;;
      --until=*)    UNTIL="${1#*=}"; shift ;;
      --skip)       SKIP="${2:?}"; shift 2 ;;
      --skip=*)     SKIP="${1#*=}"; shift ;;
      --gate)       GATE="${2:?--gate needs a video path}"; shift 2 ;;
      --gate=*)     GATE="${1#*=}"; shift ;;
      --with-pexels) WITH_PEXELS=1; shift ;;
      --fp32)       FP16=0; shift ;;
      --gpu)        GPU="${2:?}"; shift 2 ;;
      --batch)      BATCH="${2:?}"; BATCH_SET=1; shift 2 ;;
      --jobs)       JOBS="${2:?}"; shift 2 ;;
      -h|--help)    usage; exit 0 ;;
      -*)           die "unknown option '$1' (bash run.sh all --help)" ;;
      *)            [[ -z "$GATE" && -f "$1" ]] || die "unexpected argument '$1'"
                    GATE="$1"; shift ;;   # old form: run.sh all /path/gate.mp4
    esac
  done
  if [[ "$PROFILE" == auto ]]; then if is_wsl; then PROFILE=laptop; else PROFILE=dgx; fi; fi
  [[ "$PROFILE" == dgx || "$PROFILE" == laptop ]] || die "--profile must be dgx or laptop, got '$PROFILE'"
  [[ "$DETECTOR_ARG" =~ ^(auto|x|s)$ ]] || die "--detector must be auto, x or s, got '$DETECTOR_ARG'"
  SKIP=" ${SKIP//,/ } "
  [[ -z "$GATE" || -f "$GATE" ]] || die "gate clip not found: $GATE"
  [[ -z "$GATE" ]] || GATE="$(cd "$(dirname "$GATE")" && pwd)/$(basename "$GATE")"
  local ncpu; ncpu=$(nproc 2>/dev/null || echo 4)
  if [[ "$PROFILE" == laptop ]]; then
    [[ -n "${BATCH_SET:-}" || -n "$BATCH_FROM_ENV" ]] || BATCH=1
    : "${JOBS:=$(( ncpu > 3 ? (ncpu - 2 < 6 ? ncpu - 2 : 6) : 1 ))}"
    [[ -n "$MOT20_VARIANTS_FROM_ENV" ]] || MOT20_VARIANTS="sparsetrack_ablation mot17full_xdomain"
    MIN_GB=${MIN_FREE_GB:-25} WARN_GB=40
  else
    : "${JOBS:=$(( ncpu < 16 ? ncpu : 16 ))}"
    MIN_GB=${MIN_FREE_GB:-25} WARN_GB=70
  fi
  : "${FP16:=1}"
  export GPU BATCH JOBS MOT20_VARIANTS FP16
}

# The detector to use from now on: the --detector choice, or s after a recorded OOM fallback
current_detector() {
  if [[ -f "$PIPE_STATE/detector" ]]; then cat "$PIPE_STATE/detector"
  elif [[ "$DETECTOR_ARG" == s ]]; then echo s
  else echo x; fi
}
fallback_allowed() { [[ "$PROFILE" == laptop && "$DETECTOR_ARG" == auto ]]; }

# ------------------------------------------------------------------ stage list
CORE=() EXTRA=()
build_stages() {
  local d i last
  for d in mot17 mot20 dancetrack; do [[ "$SKIP" == *" $d "* ]] || CORE+=("$d"); done
  [[ ${#CORE[@]} -gt 0 ]] || die "--skip removed every dataset"
  [[ "$SKIP" == *" sompt22 "* ]] || EXTRA+=(sompt22)
  [[ -z "$GATE" ]] || EXTRA+=(video)
  [[ "$WITH_PEXELS" == 0 ]] || EXTRA+=(pexels)
  pipe_add preflight always "GPU via nvidia-smi$(is_wsl && echo ' (WSL2)'), tools, disk, datasets, frozen files"
  pipe_add setup once "env + pinned sources + weights + GPU self-test (scripts/dgx/setup.sh)"
  pipe_add gpucheck always "torch/CUDA versions; torch sees the GPU (cu118 rebuild if not)"
  last="${CORE[-1]}"
  for i in "${!CORE[@]}"; do
    d="${CORE[$i]}"
    pipe_add "stagea_$d" once "Stage A $d: detector + re-ID caches"
    if [[ "$d" == "$last" ]]; then
      for e in "${EXTRA[@]}"; do pipe_add "stagea_$e" once "Stage A $e (optional)"; done
    fi
    if [[ "$PROFILE" == laptop ]]; then add_results_stages "$d"; fi
  done
  [[ "$PROFILE" == laptop ]] || add_results_stages all
  if [[ -n "$UNTIL" ]]; then
    pipe_has "$UNTIL" || die "--until: unknown stage '$UNTIL' (stages: ${PIPE_STAGES[*]})"
    local keep=() n
    for n in "${PIPE_STAGES[@]}"; do keep+=("$n"); [[ "$n" == "$UNTIL" ]] && break; done
    PIPE_STAGES=("${keep[@]}")
  fi
}
add_results_stages() {
  pipe_add "ingest_$1" once "integrity check + make ingest -> \$ZED_ROOT/stagea (all caches so far)"
  pipe_add "phase3_$1" once "make phase3 (trackers, TrackEval, reproduction, go/no-go v2) + gate events + run-info stamp"
  pipe_add "finalize_$1" once "finalize.sh --results: one tar + manifest + sha256 -> \$ZED_ROOT/final/"
}

pipe_exec() {
  case "$1" in
    preflight) stage_preflight ;;
    setup)     stage_setup ;;
    gpucheck)  stage_gpucheck ;;
    stagea_*)  stage_stagea "${1#stagea_}" ;;
    ingest_*)  stage_ingest ;;
    phase3_*)  stage_phase3 "${1#phase3_}" ;;
    finalize_*) stage_finalize "${1#finalize_}" ;;
    *) die "no such stage: $1" ;;
  esac
}

# ------------------------------------------------------------------ preflight
dataset_ready() {   # 0 if extracted
  case "$1" in
    mot17) [[ -f "$DATA/.extracted_MOT17_train" ]] ;;
    mot20) [[ -f "$DATA/.extracted_MOT20_train" ]] ;;
    dancetrack) [[ -f "$DATA/DanceTrack/.extracted_dancetrack_val.zip" ]] ;;
  esac
}
dataset_zip() {
  case "$1" in
    mot17) echo "$DL/MOT17.zip|5860214001|$MOT17_URL" ;;
    mot20) echo "$DL/MOT20.zip|5028926248|$MOT20_URL" ;;
    dancetrack) echo "$DL/dancetrack_val.zip|4209785614|$DANCETRACK_VAL_URL" ;;
  esac
}

write_gpu_json() {
  local csv="$PIPE_STATE/gpu.csv"
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits \
    -i "$GPU" > "$csv" 2>/dev/null \
    || nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits -i "$GPU" > "$csv" \
    || die "nvidia-smi cannot query GPU $GPU (nvidia-smi -L lists the indices; set --gpu N)"
  if command -v python3 >/dev/null && PYTHONPATH="$REPO" python3 -m eval.run_info gpu < "$csv" > "$PIPE_STATE/gpu.json" 2>/dev/null; then
    :
  else   # no usable system python: name + memory via awk
    awk -F', *' 'NR==1{gsub(/"/,"",$1); m=$2+0; printf "{\"index\": 0, \"name\": \"%s\", \"memory_mib\": %s, \"driver\": \"%s\"}\n", $1, (m>0?m:"null"), $3}' \
      "$csv" > "$PIPE_STATE/gpu.json"
  fi
  GPU_NAME="$(head -1 "$csv" | cut -d, -f1 | sed 's/ *$//')"
  GPU_MEM="$(head -1 "$csv" | cut -d, -f2 | tr -dc 0-9)"
}

stage_preflight() {
  local t miss=() free_gb d zip bytes url
  (( BASH_VERSINFO[0] >= 4 )) || die "bash >= 4 needed (have $BASH_VERSION)"
  for t in wget unzip git tar sha256sum make awk df flock; do command -v "$t" >/dev/null || miss+=("$t"); done
  [[ ${#miss[@]} -eq 0 ]] || die "missing tools: ${miss[*]}  (Ubuntu: sudo apt install -y ${miss[*]/flock/util-linux})"

  # -- GPU visible
  if is_wsl; then
    log "WSL2 detected ($(uname -r)); the GPU comes from the WINDOWS NVIDIA driver"
    [[ -x /usr/lib/wsl/lib/nvidia-smi ]] && export PATH="$PATH:/usr/lib/wsl/lib"
    if dpkg-query -W -f='${Package}\n' 'nvidia-driver-*' 'nvidia-dkms-*' 2>/dev/null | grep -q .; then
      warn "a Linux NVIDIA driver package is installed inside WSL; it can shadow the Windows driver. Remove it: sudo apt remove --purge 'nvidia-driver-*' 'nvidia-dkms-*'"
    fi
    case "$REPO $ZED_ROOT" in
      */mnt/[a-z]/*) warn "repo or ZED_ROOT is on the Windows drive (/mnt/...): I/O is several times slower; use ~/ inside WSL" ;;
    esac
  fi
  if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
    if is_wsl; then
      die "nvidia-smi does not work inside WSL. Fix on WINDOWS (not in Ubuntu): install/update the NVIDIA Game Ready or Studio driver, run 'wsl --update' in PowerShell, then 'wsl --shutdown' and reopen Ubuntu. Do NOT install an NVIDIA driver inside WSL Ubuntu."
    fi
    die "nvidia-smi not found or failing: is this a GPU node? (nvidia-smi -L)"
  fi
  nvidia-smi -L
  write_gpu_json
  log "GPU $GPU: $GPU_NAME, ${GPU_MEM:-?} MiB; profile $PROFILE; detector policy: $DETECTOR_ARG (now: $(current_detector)); batch $BATCH; phase-3 jobs $JOBS"
  if [[ "$PROFILE" == laptop && -n "$GPU_MEM" ]] && (( GPU_MEM < 8000 )); then
    log "under 8 GB: still YOLOX-X fp16, batch 1 at 800x1440; only a real CUDA out-of-memory error switches to YOLOX-S$(fallback_allowed || echo ' (disabled: --detector given)')"
  fi
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -dc 0-9)
  if [[ -n "$used" && -n "$GPU_MEM" ]] && (( used * 2 > GPU_MEM )); then
    warn "GPU $GPU already has ${used} MiB in use by other processes: expect OOM (pick another with --gpu N, or close them)"
  fi

  # -- disk
  free_gb=$(df -BG --output=avail "$ZED_ROOT" | tail -1 | tr -dc 0-9)
  log "free disk under $ZED_ROOT: ${free_gb} GB"
  (( free_gb >= MIN_GB )) || die "only ${free_gb} GB free under ZED_ROOT=$ZED_ROOT; need >= ${MIN_GB} GB (about ${WARN_GB} GB for everything)"
  (( free_gb >= WARN_GB )) || warn "only ${free_gb} GB free (< ${WARN_GB} GB): downloads + extraction + caches may not fit. Delete $DL/*.zip after extraction if short."

  # -- datasets present (or downloadable)
  for d in "${CORE[@]}"; do
    if dataset_ready "$d"; then log "dataset $d: extracted"; continue; fi
    IFS='|' read -r zip bytes url <<<"$(dataset_zip "$d")"
    if [[ -f "$zip" && "$(stat -c %s "$zip")" == "$bytes" ]]; then log "dataset $d: archive complete, will extract"; continue; fi
    if wget -q --spider --timeout=20 --tries=2 "$url"; then
      log "dataset $d: not here yet; will download $(( bytes / 1000000 )) MB from $url (or pre-fetch: bash run.sh download $d)"
    else
      die "dataset $d is not present and $url is unreachable. Put the archive at $zip (from any connected machine), or run: bash run.sh download $d"
    fi
  done
  if [[ " ${EXTRA[*]} " == *" sompt22 "* ]]; then
    if compgen -G "$DL/sompt22/*" >/dev/null; then log "SOMPT22: archives present"; else log "SOMPT22: no archives in $DL/sompt22 -> that stage will be skipped"; fi
  fi
  [[ -z "$GATE" ]] || log "gate clip: $GATE"

  # -- frozen pre-registration files (the go/no-go v2 verdict is only valid on them)
  if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
    if ! git -C "$REPO" diff --quiet --ignore-cr-at-eol HEAD -- configs/phase3/plan.yaml \
         docs/preregistration configs/ours/delayed_v2_reid.yaml; then
      die "the frozen files (configs/phase3/plan.yaml, docs/preregistration/, configs/ours/delayed_v2_reid.yaml) differ from git HEAD: the verdict would not be the pre-registered one. Restore them: git -C $REPO checkout -- configs/phase3/plan.yaml docs/preregistration configs/ours/delayed_v2_reid.yaml"
    fi
    log "repo $(git -C "$REPO" rev-parse --short HEAD) on $(git -C "$REPO" rev-parse --abbrev-ref HEAD); frozen files unmodified"
  else
    warn "$REPO is not a git checkout: cannot verify the frozen files (their sha256 goes into the manifest)"
  fi

  # -- environment dump (goes into the results tar)
  {
    echo "# run.sh all preflight, $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "profile=$PROFILE detector_policy=$DETECTOR_ARG batch=$BATCH jobs=$JOBS gpu=$GPU zed_root=$ZED_ROOT"
    echo "mot20_variants=$MOT20_VARIANTS core=${CORE[*]} extra=${EXTRA[*]}"
    uname -a; cat /proc/version 2>/dev/null || true
    is_wsl && echo "WSL: ${WSL_DISTRO_NAME:-yes}"
    (. /etc/os-release 2>/dev/null && echo "os: $PRETTY_NAME") || true
    nvidia-smi || true
    df -h "$ZED_ROOT" || true
    git -C "$REPO" log -1 --format='git %H %cd' 2>/dev/null || true
    git -C "$REPO" status --short 2>/dev/null || true
  } > "$PIPE_STATE/env.txt" 2>&1
  printf 'profile=%s\n' "$PROFILE" > "$PIPE_STATE/profile.env"
}

stage_setup() {
  local args=""
  [[ "$PROFILE" == laptop ]] && args="--allow-oom"
  SELFTEST_ARGS="$args" bash "$HERE/setup.sh"
}

torch_probe() {
  python - "$PIPE_STATE/torch.json" <<'PY'
import json, sys
import torch
info = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}
if info["cuda_available"]:
    p = torch.cuda.get_device_properties(0)
    info.update(device=p.name, memory_mib=p.total_memory // 2**20, capability=f"{p.major}.{p.minor}")
    x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    info["fp16_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
json.dump(info, open(sys.argv[1], "w"), indent=1)
print(json.dumps(info))
sys.exit(0 if info["cuda_available"] and info.get("fp16_matmul_ok") else 3)
PY
}

stage_gpucheck() {
  activate
  if torch_probe; then return 0; fi
  local build; build=$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo none)
  if [[ "$build" == 11.8 || "${NO_CU118_FALLBACK:-0}" == 1 || -f "$PIPE_STATE/cu118_tried" ]]; then
    die "torch ($build) cannot use the GPU even after the cu118 rebuild. Check: nvidia-smi driver version (needs >= 452.39 for cu118), another process holding the GPU, or $(is_wsl && echo 'the Windows NVIDIA driver / wsl --update' || echo 'the node driver')."
  fi
  warn "torch (CUDA $build) cannot use the GPU: the documented fallback, rebuilding the env with the CUDA 11.8 wheel"
  pipe_note "gpucheck: torch CUDA $build could not use the GPU -> rebuilt env with TORCH_CUDA=cu118"
  touch "$PIPE_STATE/cu118_tried"
  deactivate 2>/dev/null || true
  rm -rf "${ENV_DIR:?}"
  TORCH_CUDA=cu118 SELFTEST_ARGS="$([[ "$PROFILE" == laptop ]] && echo --allow-oom)" bash "$HERE/setup.sh"
  activate
  torch_probe || die "still no usable GPU with the cu118 env; see the torch line above"
}

# ------------------------------------------------------------------ Stage A
# x-detector cache labels per dataset (moved aside on an OOM fallback)
x_labels() {
  case "$1" in
    mot17) echo "MOT17/ablation MOT17/mot17full_SAW_EVAL_HALF" ;;
    mot20) echo "MOT20/sparsetrack_ablation MOT20/mot17full_xdomain MOT20/mot20full_SAW_EVAL_HALF MOT20/mot20half" ;;
    dancetrack) echo "DanceTrack/dancetrack" ;;
  esac
}
is_oom() { grep -qiE 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation|CUDNN_STATUS_ALLOC_FAILED' "$1"; }

eta_watch() {   # mot17 only: after the first new sequence, log fps + time left
  local logf="$1" stage_pid="$2" n0 n later
  n0=$(wc -l < "$logf" 2>/dev/null || echo 0)
  while kill -0 "$stage_pid" 2>/dev/null; do
    sleep 20
    n=$(wc -l < "$logf" 2>/dev/null || echo 0)
    if (( n > n0 )); then
      if [[ "$(current_detector)" == x ]]; then
        later="MOT20=8931x$(wc -w <<<"$MOT20_VARIANTS")"
      else
        later="MOT20=8931"
      fi
      [[ " ${CORE[*]} " == *" mot20 "* ]] || later=""
      [[ " ${CORE[*]} " == *" dancetrack "* ]] && later="$later DanceTrack=?"
      # shellcheck disable=SC2086
      pipe_note "$(python -m eval.eta --log "$logf" --seqs "$DATA/MOT17/train" --glob '*-FRCNN' --later $later \
                   --json "$PIPE_STATE/eta_mot17.json" 2>&1 | tail -1)"
      pipe_write_status
      return 0
    fi
  done
}

run_stagea_once() {   # DATASET DETECTOR -> runs run.sh, returns its exit code; output in $2.out
  local d="$1" det="$2" out="$PIPE_STATE/stagea_$1.out" rc watcher=""
  if [[ "$d" == mot17 && "$ETA_WATCH" == 1 ]]; then
    local label=ablation; [[ "$det" == s ]] && label="$SMALL_LABEL_MOT17"
    mkdir -p "$CACHE/MOT17/$label/train"
    eta_watch "$CACHE/MOT17/$label/train/stage_a_log.jsonl" "$BASHPID" & watcher=$!
  fi
  local args=("$d")
  [[ "$d" == video ]] && args=(video "$GATE" gate)
  set +e
  DETECTOR="$det" BATCH="$BATCH" bash "$RUN_SH" "${args[@]}" 2>&1 | tee "$out"
  rc=${PIPESTATUS[0]}
  set -e
  [[ -z "$watcher" ]] || { kill "$watcher" 2>/dev/null || true; wait "$watcher" 2>/dev/null || true; }
  return "$rc"
}

stage_stagea() {
  local d="$1" det rc out="$PIPE_STATE/stagea_$1.out" dst lbl
  if [[ "$d" == sompt22 ]] && ! compgen -G "$DL/sompt22/*" >/dev/null; then
    pipe_skip "no SOMPT22 archives in $DL/sompt22 (manual download, see the runbook); re-run later"
    return 0
  fi
  det="$(current_detector)"
  case "$d" in sompt22|video|pexels) det=x ;; esac   # not in phase 3; YOLOX-X mot17 weights as before
  log "Stage A $d with detector $det (batch $BATCH)"
  rc=0; run_stagea_once "$d" "$det" || rc=$?
  if [[ $rc -ne 0 ]]; then
    if is_oom "$out" && [[ "$det" == x ]] && fallback_allowed && [[ "$d" =~ ^(mot17|mot20|dancetrack)$ ]]; then
      dst="$ZED_ROOT/cache_oom_partial/$(date +%Y%m%d_%H%M%S)"
      for lbl in $(x_labels "$d"); do
        if [[ -d "$CACHE/$lbl" ]]; then mkdir -p "$dst/$(dirname "$lbl")"; mv "$CACHE/$lbl" "$dst/$lbl"; fi
      done
      echo s > "$PIPE_STATE/detector"
      pipe_note "CUDA out of memory with YOLOX-X on $d ($(cut -d, -f1-2 "$PIPE_STATE/gpu.csv" 2>/dev/null | head -1) MiB). Switched to YOLOX-S bytetrack_s_mot17 for $d and every later dataset; its rows are tagged 'laptop detector, not comparable to published numbers'. Partial YOLOX-X caches moved to $dst"
      pipe_write_status
      rc=0; run_stagea_once "$d" s || rc=$?
      [[ $rc -eq 0 ]] || die "Stage A $d failed again with YOLOX-S (exit $rc); see $out"
      det=s
    elif is_oom "$out"; then
      die "CUDA out of memory in Stage A $d (detector $det, batch $BATCH). Free the GPU (nvidia-smi), pick another (--gpu N), lower --batch, or use --detector s; then re-run."
    else
      die "Stage A $d failed (exit $rc); the log above names the cause ($out). Re-run to resume: finished sequences are kept."
    fi
  fi
  echo "$det" > "$PIPE_STATE/detector_$d"
}

# ------------------------------------------------------------------ results
small_datasets() { local d; for d in mot17 mot20 dancetrack; do [[ "$(cat "$PIPE_STATE/detector_$d" 2>/dev/null)" == s ]] && echo "$d"; done; return 0; }

stage_ingest() {
  activate
  python -m models.bringback --root "$ZED_ROOT"   # INTEGRITY ERRORS -> non-zero -> stop
  rm -rf "$ZED_ROOT/stagea"                         # a fresh, consistent snapshot every time
  make -C "$REPO" --no-print-directory ingest PY=python BRING_BACK="$ZED_ROOT/bring_back" \
    STAGEA="$ZED_ROOT/stagea" DATA="$ZED_ROOT/data"
  [[ -f "$ZED_ROOT/stagea/index.json" ]] || die "ingest wrote no $ZED_ROOT/stagea/index.json"
}

phase3_args() {   # STAGE -> the PHASE3_ARGS string for `make phase3`
  local p3="" d
  for d in $(small_datasets); do
    case "$d" in
      mot17) p3+=" --remap-cache MOT17/ablation/train=MOT17/$SMALL_LABEL_MOT17/train" ;;
      dancetrack) p3+=" --remap-cache DanceTrack/dancetrack/val=DanceTrack/$SMALL_LABEL_XDOMAIN/val" ;;
    esac
  done
  [[ -z "$(small_datasets)" ]] || p3+=" --tag-cache '*/laptop_yolox_s_*/*'"
  # per-run timeout of the go/no-go MOT20 replication (plan default 60 min was too short for the
  # laptop: the Oct 2026 sparsetrack K=5 arm timed out). Finished arms are kept, so re-running resumes.
  local t20="${PHASE3_TIMEOUT_MIN_MOT20:-}"
  [[ -n "$t20" || "$PROFILE" != laptop ]] || t20=240
  [[ -z "$t20" ]] || p3+=" --timeout-min MOT20=$t20"
  if pipe_forced "$1"; then p3+=" --force"; fi
  echo "$p3"
}

stage_phase3() {
  local label="$1" report="$ZED_ROOT/phase3/report" out="$ZED_ROOT/phase3/out" args gt runs=() r tagged
  activate
  args="$(phase3_args "phase3_$label")"
  log "make phase3 (jobs $JOBS)${args:+ with:$args}"
  make -C "$REPO" --no-print-directory phase3 PY=python STAGEA="$ZED_ROOT/stagea" \
    PHASE3_OUT="$out" PHASE3_REPORT="$report" JOBS="$JOBS" PHASE3_ARGS="$args"
  # gate events on the MOT17 val-half tracker outputs (frozen lines, configs/gates/)
  gt="$out/mot17_val_half/bytetrack/gt"
  if [[ -d "$gt/MOT17-02-FRCNN" && -d "$gt/MOT17-04-FRCNN" && -d "$gt/MOT17-09-FRCNN" ]]; then
    for r in "$out"/mot17_val_half/*/trackers/*/; do runs+=("${r%/}"); done
    (cd "$REPO" && python -m eval.gate_events --gates configs/gates/mot17_val_half.yaml --gt-dir "$gt" \
      --include-gt --runs "${runs[@]}" --out "$report/gate_events")
  else
    log "gate events: MOT17-02/04/09 val-half outputs not there (yet); skipped"
  fi
  tagged="$(for d in $(small_datasets); do case $d in mot17) echo "MOT17/$SMALL_LABEL_MOT17/train" ;; mot20) echo "MOT20/$SMALL_LABEL_XDOMAIN/train" ;; dancetrack) echo "DanceTrack/$SMALL_LABEL_XDOMAIN/val" ;; esac; done)"
  # shellcheck disable=SC2086
  python -m eval.run_info stamp --report "$report" --index "$ZED_ROOT/stagea/index.json" \
    --profile "$PROFILE" --gpu-json "$PIPE_STATE/gpu.json" --host "$(hostname)" \
    --tagged $tagged --out "$PIPE_STATE/run_info.json"
  python - "$report/summary.json" "$PIPE_STATE/verdict.txt" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
rep = (s.get("reproduction") or {}).get("verdict", "not computed")
gng = s.get("go_no_go_v2", "not computed")
line = f"ByteTrack reproduction: {rep}; go/no-go v2: {gng}; ran: {[r['id'] for r in s['ran']]}"
open(sys.argv[2], "w").write(line + "\n")
print(line)
for w in s.get("waiting", []):
    print("waiting:", w)
PY
  pipe_note "phase3 ($label): $(cat "$PIPE_STATE/verdict.txt")"
  if [[ " ${CORE[*]} " == *" mot17 "* ]] && ! grep -q '"go_no_go_v2"' "$report/summary.json"; then
    die "phase 3 produced no go/no-go v2 entry although MOT17 is in the plan; see 'waiting:' above"
  fi
}

stage_finalize() {
  RUN_ALL_LOGGING=1 bash "$HERE/finalize.sh" --results --profile "$PROFILE" --label "$1"
  pipe_note "results ($1): $ZED_ROOT/final/$(cat "$ZED_ROOT/final/LATEST")"
}

# ------------------------------------------------------------------ main
[[ "${RUN_ALL_SOURCE_ONLY:-0}" == 1 ]] && return 0
parse_args "$@"
build_stages
# shellcheck disable=SC2034   # read by pipeline_lib.sh
PIPE_TITLE="run.sh all --profile $PROFILE (ZED_ROOT=$ZED_ROOT)"
if [[ "$PIPE_DRY_RUN" == 1 ]]; then
  printf 'profile %s | detector %s (now %s) | batch %s | phase-3 jobs %s | GPU index %s\n' \
    "$PROFILE" "$DETECTOR_ARG" "$(current_detector)" "$BATCH" "$JOBS" "$GPU"
  printf 'datasets: %s | optional: %s | MOT20 variants: %s\n' "${CORE[*]}" "${EXTRA[*]:-none}" "$MOT20_VARIANTS"
  printf 'ZED_ROOT %s (GPU, disk and datasets are checked by preflight on a real run)\n\n' "$ZED_ROOT"
  pipe_run
  exit 0
fi
mkdir -p "$PIPE_STATE"
if [[ -f "$PIPE_STATE/profile.env" ]] && ! grep -qx "profile=$PROFILE" "$PIPE_STATE/profile.env" && [[ "$PIPE_FORCE_ALL" != 1 ]]; then
  die "$ZED_ROOT was run with $(cat "$PIPE_STATE/profile.env"); use the same --profile, --force, or another ZED_ROOT"
fi
exec 9>"$PIPE_STATE/lock"
flock -n 9 || die "another 'run.sh all' is already running on $ZED_ROOT (see: cat $PIPE_STATUS)"
# outside $LOGS on purpose: bring_back/ hard-links $LOGS, and this file grows while ingest runs
mkdir -p "$PIPE_STATE/logs"
PIPE_LOG="$PIPE_STATE/logs/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$PIPE_LOG") 2>&1
ln -sfn "$PIPE_LOG" "$ZED_ROOT/run_all.log"
log "logging to $PIPE_LOG ; status: cat $PIPE_STATUS"
if [[ "$PIPE_FORCE_ALL" == 1 ]]; then rm -f "$PIPE_STATE/detector" "$PIPE_STATE"/detector_*; fi
pipe_run
log "ALL DONE. Results: $ZED_ROOT/final/$(cat "$ZED_ROOT/final/LATEST" 2>/dev/null)"
cat "$PIPE_STATUS"
if [[ "$PROFILE" == laptop ]]; then
  cat <<MSG

View locally (repo venv: make setup once):
  cd $REPO && make ingest BRING_BACK=$ZED_ROOT/final/zed_results_latest.tar.gz && make demo
MSG
else
  cat <<MSG

Copy ONE file home (Windows PowerShell on your laptop), then ingest it there:
  scp $(whoami)@<DGX address>:${ZED_ROOT#"$HOME"/}/final/$(cat "$ZED_ROOT/final/LATEST" 2>/dev/null) .
  make ingest BRING_BACK=<that file> DELETE=1
MSG
fi

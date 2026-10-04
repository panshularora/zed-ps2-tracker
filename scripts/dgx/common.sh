# Shared helpers for scripts/dgx/*.sh. Source it; don't run it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck source=config.env
source "$HERE/config.env"
mkdir -p "$DL" "$DATA" "$WEIGHTS" "$CACHE" "$CLIPS" "$RESULTS" "$LOGS" "$SRC"

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[%s] WARNING:\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '\033[1;31m[%s] ERROR:\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

# Tee everything from here on into logs/<name>_<timestamp>.log
start_log() {
  local f="$LOGS/$1_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$f") 2>&1
  log "logging to $f"
}

activate() {
  if [[ -f "$ENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
  elif [[ -d "$ENV_DIR/conda-meta" ]]; then
    eval "$(conda shell.bash hook)"; conda activate "$ENV_DIR"
  else
    die "environment not found at $ENV_DIR, run: bash scripts/dgx/setup.sh"
  fi
  export PYTHONPATH="$REPO:$SRC/ByteTrack:$SRC/deep-person-reid${PYTHONPATH:+:$PYTHONPATH}"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID   # so GPU=<n> is the same index nvidia-smi shows
  export CUDA_VISIBLE_DEVICES="$GPU"
  export PYTHONWARNINGS="ignore"
}

# fetch URL DEST [EXPECTED_BYTES] : resumable download, size-checked
fetch() {
  local url="$1" dest="$2" want="${3:-}"
  if [[ -n "$want" && -f "$dest" && "$(stat -c %s "$dest")" == "$want" ]]; then
    log "have $(basename "$dest") ($want bytes)"; return 0
  fi
  log "downloading $(basename "$dest") ..."
  wget -c --tries=20 --retry-connrefused --waitretry=5 --progress=dot:giga -O "$dest" "$url" \
    || die "download failed: $url (re-run the same command to resume)"
  if [[ -n "$want" && "$(stat -c %s "$dest")" != "$want" ]]; then
    die "$(basename "$dest"): got $(stat -c %s "$dest") bytes, expected $want. Re-run to resume."
  fi
}

# unzip_once ZIP DESTDIR : extract once; a marker file makes re-runs instant
unzip_once() {
  local zip="$1" dest="$2" marker="$2/.extracted_$(basename "$1")"
  [[ -f "$marker" ]] && { log "already extracted $(basename "$zip")"; return 0; }
  mkdir -p "$dest"; log "extracting $(basename "$zip") -> $dest"
  unzip -q -o "$zip" -d "$dest" && touch "$marker"
}

# stage_a ARGS... : run models.stage_a with the shared defaults
stage_a() {
  python -m models.stage_a "$@" --conf-floor "$CONF_FLOOR" --nms "$NMS" --batch-size "$BATCH" \
    --device "$DEVICE" --variant "$VARIANT" \
    --reid-weights "$WEIGHTS/osnet_ain_x1_0_msmt17.pth"
}

ffmpeg_bin() { command -v ffmpeg || python -c "import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())"; }

# make_clip SEQ_DIR OUT.mp4 : the agreed demo encoding from an img1/ folder
make_clip() {
  local seq="$1" out="$2" fps
  [[ -s "$out" ]] && { log "clip exists: $out"; return 0; }
  fps="$(awk -F= '/^frameRate/{print $2}' "$seq/seqinfo.ini" | tr -d '\r ')"
  log "encoding $out at ${fps} fps"
  "$(ffmpeg_bin)" -hide_banner -loglevel error -y -framerate "$fps" -i "$seq/img1/%06d.jpg" \
    "${FFMPEG_ARGS[@]}" "$out.tmp.mp4" && mv "$out.tmp.mp4" "$out"
}

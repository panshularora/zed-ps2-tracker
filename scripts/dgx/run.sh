#!/usr/bin/env bash
# Stage A run, one command per dataset. Resumable: re-run the same command after any failure.
#
#   bash scripts/dgx/run.sh all [--profile dgx|laptop] [--dry-run] [--force[=STAGES]] ...
#                                                # THE WHOLE PIPELINE (scripts/dgx/run_all.sh)
#   bash scripts/dgx/run.sh mot17                # MOT17 train (half-split eval), ablation weights
#   bash scripts/dgx/run.sh mot20                # MOT20 train: ablation (clean) + mot20 full (SAW EVAL HALF)
#   bash scripts/dgx/run.sh dancetrack           # DanceTrack val
#   bash scripts/dgx/run.sh sompt22              # SOMPT22 train + test (+ baseline test-format results)
#   bash scripts/dgx/run.sh video /path/gate.mp4 [name]   # your own gate clip
#   bash scripts/dgx/run.sh pexels               # optional Pexels demo clips
#   bash scripts/dgx/run.sh download [mot17 mot20 dancetrack]   # only download + extract
#   bash scripts/dgx/run.sh stagea-all [/path/gate.mp4]  # every Stage A target (the old `all`)
#
# DETECTOR=s uses YOLOX-S bytetrack_s_mot17 for every dataset (laptop fallback, see config.env).
# Then: bash scripts/dgx/finalize.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
target="${1:-}"; [[ -n "$target" ]] || { sed -n 2,17p "$0"; exit 1; }
if [[ "$target" == all ]]; then shift; exec bash "$HERE/run_all.sh" "$@"; fi
[[ "$DETECTOR" == x || "$DETECTOR" == s ]] || die "DETECTOR must be x or s, got '$DETECTOR'"
W="$WEIGHTS"
begin() { start_log "run_${target}"; activate; log "detector: $DETECTOR, batch $BATCH, GPU $GPU"; }

# seq_root DIR : the directory whose children are sequence folders (with seqinfo.ini)
seq_root() {
  local f; f="$(find "$1" -mindepth 2 -maxdepth 4 -name seqinfo.ini -print -quit)"
  [[ -n "$f" ]] || die "no sequence (seqinfo.ini) under $1"
  dirname "$(dirname "$f")"
}

# small_weights : download + sha256-check bytetrack_s_mot17 (only for DETECTOR=s); sets SMALL_PATH
small_weights() {
  local name url bytes got
  IFS='|' read -r name url bytes <<<"$SMALL_WEIGHT"
  SMALL_PATH="$W/$name"
  fetch "$url" "$SMALL_PATH" "$bytes"
  got=$(sha256sum "$SMALL_PATH" | cut -d' ' -f1)
  [[ "$got" == "$SMALL_SHA256" ]] || die "$name: sha256 $got != $SMALL_SHA256 (delete it and re-run)"
}

# stage_a_small LABEL ARGS... : YOLOX-S on one dataset (same input sizes as YOLOX-X)
stage_a_small() {
  small_weights
  VARIANT=yolox_s stage_a "$@" --det-weights "$SMALL_PATH" --trained-on "$SMALL_TRAINED_ON"
}

download_mot17() {
  fetch "$MOT17_URL" "$DL/MOT17.zip" 5860214001
  # only the FRCNN copy of each train sequence: the DPM/SDP folders hold identical images
  if [[ ! -f "$DATA/.extracted_MOT17_train" ]]; then
    log "extracting MOT17 train (FRCNN copies only)"
    unzip -q -o "$DL/MOT17.zip" 'MOT17/train/*-FRCNN/*' -d "$DATA" && touch "$DATA/.extracted_MOT17_train"
  fi
}

download_mot20() {
  fetch "$MOT20_URL" "$DL/MOT20.zip" 5028926248
  if [[ ! -f "$DATA/.extracted_MOT20_train" ]]; then
    log "extracting MOT20 train"
    unzip -q -o "$DL/MOT20.zip" 'MOT20/train/*' -d "$DATA" && touch "$DATA/.extracted_MOT20_train"
  fi
}

download_dancetrack() {
  fetch "$DANCETRACK_VAL_URL" "$DL/dancetrack_val.zip" 4209785614
  unzip_once "$DL/dancetrack_val.zip" "$DATA/DanceTrack"
}

want_mot20() { [[ " $MOT20_VARIANTS " == *" $1 "* ]]; }

run_mot17() {
  download_mot17
  if [[ "$DETECTOR" == s ]]; then
    stage_a_small dataset --root "$DATA/MOT17/train" --glob '*-FRCNN' --dataset MOT17 --split train \
      --out "$CACHE/MOT17/$SMALL_LABEL_MOT17/train" --test-size 800 1440 --saw-eval-half
    make_clip "$DATA/MOT17/train/MOT17-04-FRCNN" "$CLIPS/MOT17-04.mp4"
    return 0
  fi
  stage_a dataset --root "$DATA/MOT17/train" --glob '*-FRCNN' --dataset MOT17 --split train \
    --out "$CACHE/MOT17/ablation/train" --det-weights "$W/bytetrack_ablation.pth.tar" \
    --test-size 800 1440 --trained-on "CrowdHuman + MOT17 train first half (ByteTrack ablation)"
  if [[ "${WITH_MOT17_FULL:-0}" == 1 ]]; then   # optional; never for half-val numbers
    stage_a dataset --root "$DATA/MOT17/train" --glob '*-FRCNN' --dataset MOT17 --split train \
      --out "$CACHE/MOT17/mot17full_SAW_EVAL_HALF/train" --det-weights "$W/bytetrack_x_mot17.pth.tar" \
      --test-size 800 1440 --saw-eval-half \
      --trained-on "CrowdHuman + MOT17 train (full) + CityPersons + ETHZ"
  fi
  make_clip "$DATA/MOT17/train/MOT17-04-FRCNN" "$CLIPS/MOT17-04.mp4"
}

run_mot20() {
  download_mot20
  if [[ "$DETECTOR" == s ]]; then   # bytetrack_s_mot17 has no MOT20 data: cross-domain, clean
    stage_a_small dataset --root "$DATA/MOT20/train" --dataset MOT20 --split train \
      --out "$CACHE/MOT20/$SMALL_LABEL_XDOMAIN/train" --test-size 896 1600
    make_clip "$DATA/MOT20/train/MOT20-02" "$CLIPS/MOT20-02.mp4"
    return 0
  fi
  log "MOT20 variants: $MOT20_VARIANTS"
  # (a) clean, in-domain: SparseTrack's yolox_x_mot20_ablation (Drive file name
  #     bytetrack_20_ablation.pth.tar), trained on MOT20 train FIRST half + CrowdHuman (per the
  #     SparseTrack README; the weights themselves are unverified, selftest checks they load)
  want_mot20 sparsetrack_ablation && \
  stage_a dataset --root "$DATA/MOT20/train" --dataset MOT20 --split train \
    --out "$CACHE/MOT20/sparsetrack_ablation/train" --det-weights "$W/sparsetrack_mot20_ablation.pth.tar" \
    --test-size 896 1600 --trained-on "CrowdHuman + MOT20 train first half (SparseTrack ablation)"
  # (b) clean, cross-domain: ByteTrack MOT17 full model has no MOT20 data (Deep OC-SORT's choice)
  want_mot20 mot17full_xdomain && \
  stage_a dataset --root "$DATA/MOT20/train" --dataset MOT20 --split train \
    --out "$CACHE/MOT20/mot17full_xdomain/train" --det-weights "$W/bytetrack_x_mot17.pth.tar" \
    --test-size 896 1600 --trained-on "CrowdHuman + MOT17 + CityPersons + ETHZ (no MOT20)"
  # (d) strongest detector, but trained on ALL of MOT20 train: never for val-half headline numbers
  want_mot20 mot20full_SAW_EVAL_HALF && \
  stage_a dataset --root "$DATA/MOT20/train" --dataset MOT20 --split train \
    --out "$CACHE/MOT20/mot20full_SAW_EVAL_HALF/train" --det-weights "$W/bytetrack_x_mot20.pth.tar" \
    --test-size 896 1600 --saw-eval-half --trained-on "CrowdHuman + MOT20 train (full)"
  # (c) optional: our own first-half fine-tune (scripts/dgx/finetune_mot20_half.sh)
  if [[ -f "$W/yolox_x_mot20_half.pth.tar" ]]; then
    stage_a dataset --root "$DATA/MOT20/train" --dataset MOT20 --split train \
      --out "$CACHE/MOT20/mot20half/train" --det-weights "$W/yolox_x_mot20_half.pth.tar" \
      --test-size 896 1600 --trained-on "bytetrack_ablation fine-tuned on MOT20 train first half"
  fi
  make_clip "$DATA/MOT20/train/MOT20-02" "$CLIPS/MOT20-02.mp4"
}

run_dancetrack() {
  download_dancetrack
  local root; root="$(seq_root "$DATA/DanceTrack")"
  if [[ "$DETECTOR" == s ]]; then   # no small DanceTrack checkpoint: the MOT17 one, cross-domain
    stage_a_small dataset --root "$root" --dataset DanceTrack --split val \
      --out "$CACHE/DanceTrack/$SMALL_LABEL_XDOMAIN/val" --test-size 800 1440
    return 0
  fi
  stage_a dataset --root "$root" --dataset DanceTrack --split val \
    --out "$CACHE/DanceTrack/dancetrack/val" --det-weights "$W/dancetrack_bytetrack_x.pth.tar" \
    --test-size 800 1440 --trained-on "DanceTrack train (DanceTrack authors' ByteTrack YOLOX-X)"
}

run_sompt22() {
  mkdir -p "$DL/sompt22"
  if ! compgen -G "$DL/sompt22/*" >/dev/null; then
    log "trying gdown on the SOMPT22 Google Drive folder"
    gdown --folder "$SOMPT22_FOLDER" -O "$DL/sompt22" --remaining-ok || {
      warn "automatic download failed (the folder refused anonymous listing when tested)."
      cat <<MSG
Manual step (2 minutes):
  1. open $SOMPT22_FOLDER in a browser (sign in to Google if asked)
  2. download every file (train and test archives) into: $DL/sompt22/
  3. re-run: bash scripts/dgx/run.sh sompt22
MSG
      return 2; }
  fi
  for z in "$DL"/sompt22/*.zip "$DL"/sompt22/**/*.zip; do [[ -f "$z" ]] && unzip_once "$z" "$DATA/SOMPT22"; done
  for t in "$DL"/sompt22/*.tar* ; do
    [[ -f "$t" && ! -f "$DATA/SOMPT22/.extracted_$(basename "$t")" ]] && \
      { mkdir -p "$DATA/SOMPT22"; tar -xf "$t" -C "$DATA/SOMPT22"; touch "$DATA/SOMPT22/.extracted_$(basename "$t")"; }
  done
  python -m models.prepare sompt22 --root "$DATA/SOMPT22" --fps "${SOMPT22_FPS:-30}"
  for split in train test; do
    [[ -d "$DATA/SOMPT22/_seqs/$split" ]] || continue
    stage_a dataset --root "$DATA/SOMPT22/_seqs/$split" --dataset SOMPT22 --split "$split" \
      --out "$CACHE/SOMPT22/mot17full/$split" --det-weights "$W/bytetrack_x_mot17.pth.tar" \
      --test-size 800 1440 --trained-on "CrowdHuman + MOT17 + CityPersons + ETHZ (no SOMPT22)"
  done
  # test-format outputs of the ByteTrackLike baseline (NOT submitted; submission is a manual,
  # once-a-week decision via the SOMPT22 GitHub issue template)
  if [[ -d "$CACHE/SOMPT22/mot17full/test" ]]; then
    python -m tracker.cli --caches "$CACHE/SOMPT22/mot17full/test" \
      --config "$REPO/configs/baselines/bytetrack.yaml" \
      --out "$RESULTS/SOMPT22/bytetrack_like/test" --zip "$RESULTS/SOMPT22/bytetrack_like_test_results.zip"
  fi
  local first; first="$(ls -d "$DATA"/SOMPT22/_seqs/train/*/ 2>/dev/null | head -1 || true)"
  [[ -n "$first" ]] && make_clip "${first%/}" "$CLIPS/$(basename "$first").mp4"
}

run_video() {
  local src="$1" name="${2:-$(basename "${1%.*}")}" labels="${3:-mot17full ablation}"
  [[ -f "$src" ]] || die "video not found: $src"
  mkdir -p "$DATA/videos"
  for label in $labels; do
    local wfile=bytetrack_x_mot17.pth.tar trained="CrowdHuman + MOT17 + CityPersons + ETHZ"
    [[ "$label" == ablation ]] && { wfile=bytetrack_ablation.pth.tar; trained="CrowdHuman + MOT17 train first half"; }
    stage_a video --video "$src" --name "$name" --dataset video --split custom \
      --out "$CACHE/video/$label" --det-weights "$W/$wfile" --test-size 800 1440 --trained-on "$trained"
  done
  if [[ ! -s "$CLIPS/$name.mp4" ]]; then
    "$(ffmpeg_bin)" -hide_banner -loglevel error -y -i "$src" "${FFMPEG_ARGS[@]}" -an "$CLIPS/$name.tmp.mp4" \
      && mv "$CLIPS/$name.tmp.mp4" "$CLIPS/$name.mp4"
  fi
}

run_pexels() {
  mkdir -p "$DL/pexels" "$DATA/videos/pexels"
  for entry in "${PEXELS[@]}"; do
    IFS='|' read -r id slug <<<"$entry"
    local raw="$DL/pexels/$slug.mp4" norm="$DATA/videos/pexels/$slug.mp4"
    [[ -s "$raw" ]] || wget -c -q --user-agent="Mozilla/5.0" -O "$raw" "https://www.pexels.com/download/video/$id/" \
      || { warn "pexels $id failed; download it by hand from https://www.pexels.com/video/$slug-$id/ to $raw"; continue; }
    # normalise to <=1080p (short side) and <=30 fps so all clips share detector settings
    if [[ ! -s "$norm" ]]; then
      local fps vf="scale='if(gt(iw,ih),-2,min(1080,iw))':'if(gt(iw,ih),min(1080,ih),-2)'"
      fps=$(python -c "import cv2,sys; print(cv2.VideoCapture(sys.argv[1]).get(cv2.CAP_PROP_FPS))" "$raw")
      python -c "import sys; sys.exit(0 if float(sys.argv[1]) > 31 else 1)" "$fps" && vf="$vf,fps=30"
      "$(ffmpeg_bin)" -hide_banner -loglevel error -y -i "$raw" -vf "$vf" \
        -c:v libx264 -crf 18 -pix_fmt yuv420p -an "$norm.tmp.mp4" && mv "$norm.tmp.mp4" "$norm"
    fi
    run_video "$norm" "pexels_$slug" mot17full
  done
}

case "$target" in
  mot17)      begin; run_mot17 ;;
  mot20)      begin; run_mot20 ;;
  dancetrack) begin; run_dancetrack ;;
  sompt22)    begin; run_sompt22 ;;
  video)      [[ $# -ge 2 ]] || die "usage: run.sh video /path/clip.mp4 [name]"; begin; run_video "$2" "${3:-}" ;;
  pexels)     begin; run_pexels ;;
  download)   start_log download
              sets=("${@:2}"); [[ ${#sets[@]} -gt 0 ]] || sets=(mot17 mot20 dancetrack)
              for d in "${sets[@]}"; do
                case "$d" in mot17|mot20|dancetrack) "download_$d" ;; *) die "download: unknown dataset '$d'" ;; esac
              done ;;
  stagea-all) # each target in its own process: own log, own `set -e`, and one failure
              # (e.g. SOMPT22's manual download step) does not stop the others
              start_log run_stagea-all
              failed=()
              for t in mot17 mot20 sompt22 dancetrack; do bash "$0" "$t" || failed+=("$t"); done
              if [[ -n "${2:-}" ]]; then bash "$0" video "$2" gate || failed+=(video); fi
              bash "$0" pexels || failed+=(pexels)
              [[ ${#failed[@]} -eq 0 ]] || die "targets that need attention: ${failed[*]} (re-run each)" ;;
  *)          die "unknown target '$target' (all | mot17 | mot20 | dancetrack | sompt22 | video | pexels | download | stagea-all)" ;;
esac
log "run.sh $target finished. When all targets are done: bash scripts/dgx/finalize.sh"

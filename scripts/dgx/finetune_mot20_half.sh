#!/usr/bin/env bash
# OPTIONAL, UNTESTED ON GPU. Fine-tune YOLOX-X on MOT20 train first half only, starting from
# bytetrack_ablation, so MOT20 val-half has a strong detector that never saw it.
#   bash scripts/dgx/finetune_mot20_half.sh        (after run.sh mot20 extracted the data)
#   EPOCHS=10 NGPU=1 TBATCH=8 bash scripts/dgx/finetune_mot20_half.sh
# Output: $WEIGHTS/yolox_x_mot20_half.pth.tar ; then re-run `run.sh mot20` to cache with it.
# Rough cost: ~1.5-3 h per 10 epochs on one 48 GB GPU at batch 8 (estimate, not measured).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log finetune_mot20_half
activate
: "${EPOCHS:=10}" "${NGPU:=1}" "${TBATCH:=8}"
export MOT20_ROOT="$DATA/MOT20" EPOCHS
[[ -d "$MOT20_ROOT/train" ]] || die "run 'bash scripts/dgx/run.sh mot20' first (extracts MOT20)"
# training needs ByteTrack's compiled ops (COCO eval); build them once
if [[ ! -f "$SRC/ByteTrack/.built" ]]; then
  ( cd "$SRC/ByteTrack" && python setup.py develop ) && touch "$SRC/ByteTrack/.built"
fi
python -m models.mot_to_coco --root "$MOT20_ROOT" --split train_half
python -m models.mot_to_coco --root "$MOT20_ROOT" --split val_half
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NGPU - 1)))"
( cd "$SRC/ByteTrack" && python tools/train.py -f "$REPO/models/exps/yolox_x_mot20_half.py" \
    -d "$NGPU" -b "$TBATCH" --fp16 -o -c "$WEIGHTS/bytetrack_ablation.pth.tar" )
ckpt="$SRC/ByteTrack/YOLOX_outputs/yolox_x_mot20_half/best_ckpt.pth.tar"
[[ -f "$ckpt" ]] || ckpt="$SRC/ByteTrack/YOLOX_outputs/yolox_x_mot20_half/latest_ckpt.pth.tar"
cp "$ckpt" "$WEIGHTS/yolox_x_mot20_half.pth.tar"
log "fine-tuned weights: $WEIGHTS/yolox_x_mot20_half.pth.tar -> now: bash scripts/dgx/run.sh mot20"

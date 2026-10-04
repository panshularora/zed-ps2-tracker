#!/usr/bin/env bash
# After finalize.sh: pack bring_back/ into a few uncompressed tars for scp from Windows.
#   bash scripts/dgx/pack.sh      ->  $ZED_ROOT/bring_back_tars/{core,MOT17__ablation,...}.tar
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log pack
activate
python -m models.pack --root "$ZED_ROOT"
cat <<MSG

Copy home from your LAPTOP (Windows PowerShell), core.tar first:
  scp -r $(whoami)@<DGX address>:${ZED_ROOT#"$HOME"/}/bring_back_tars .\\zed-bring-back
MSG

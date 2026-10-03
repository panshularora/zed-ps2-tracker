#!/usr/bin/env bash
# Final integrity check + bring-back bundle:  bash scripts/dgx/finalize.sh
# Verifies every cache, then assembles $ZED_ROOT/bring_back (hard links, no extra disk) with
# MANIFEST.json + SHA256SUMS, and prints the size of each category.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log finalize
activate
python -m models.bringback --root "$ZED_ROOT"
cat <<MSG

Copy home (resumable; re-run if the connection drops):
  rsync -avhP --partial $(hostname):$BRING/ ./zed-stagea-bring-back/
then on the CPU box:
  cd zed-stagea-bring-back && sha256sum -c SHA256SUMS
MSG

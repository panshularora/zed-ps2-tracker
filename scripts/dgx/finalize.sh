#!/usr/bin/env bash
# Final integrity check + bring-back bundle:  bash scripts/dgx/finalize.sh
# Verifies every cache, then assembles $ZED_ROOT/bring_back (hard links, no extra disk) with
# MANIFEST.json + SHA256SUMS, and prints the size of each category.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log finalize
activate
python -m models.bringback --root "$ZED_ROOT"
cat <<MSG

Next, still on the DGX:  bash scripts/dgx/pack.sh   (a few tars to scp; docs/dgx_runbook.md step 7-8)
On the CPU box, each tar is checked against SHA256SUMS by:  make ingest BRING_BACK=<tar>
MSG

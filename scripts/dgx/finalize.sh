#!/usr/bin/env bash
# Two modes:
#
#   bash scripts/dgx/finalize.sh
#       Stage A integrity check + bring-back bundle. Verifies every cache, then assembles
#       $ZED_ROOT/bring_back (hard links, no extra disk) with MANIFEST.json + SHA256SUMS, and
#       prints the size of each category. (Per-dataset flow: then pack.sh, scp the tars home.)
#
#   bash scripts/dgx/finalize.sh --results [--profile dgx|laptop] [--label NAME]
#       The final results bundle of `run.sh all` (its last stage): ONE small
#       $ZED_ROOT/final/zed_results_<time>_<profile>_<label>.tar.gz with the phase-3 tables,
#       plots, go/no-go v2 verdict, reproduction check, configs, logs, run status and a
#       RESULTS_MANIFEST.json + SHA256SUMS (+ a .sha256 file for the tar). No caches, images,
#       clips or weights. Locally: make ingest BRING_BACK=<that tar>  (eval/results_bundle.py)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ "${1:-}" == --results ]]; then
  shift
  profile="${PROFILE:-dgx}" label="final"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --profile) profile="$2"; shift 2 ;;
      --label)   label="$2"; shift 2 ;;
      *) die "finalize.sh --results: unknown argument '$1'" ;;
    esac
  done
  [[ -n "${RUN_ALL_LOGGING:-}" ]] || start_log finalize_results
  activate
  report="${PHASE3_REPORT:-$ZED_ROOT/phase3/report}"
  [[ -f "$report/summary.json" ]] || die "no $report/summary.json: run the phase-3 stage first (bash run.sh all)"
  python -m eval.results_bundle build --zed-root "$ZED_ROOT" --repo "$REPO" --report "$report" \
    --index "$ZED_ROOT/stagea/index.json" --state "$ZED_ROOT/run_all" \
    --out-dir "$ZED_ROOT/final" --profile "$profile" --label "$label"
  exit 0
fi

start_log finalize
activate
python -m models.bringback --root "$ZED_ROOT"
cat <<MSG

Next, still on the DGX:  bash scripts/dgx/pack.sh   (a few tars to scp; docs/dgx_runbook.md step 7-8)
On the CPU box, each tar is checked against SHA256SUMS by:  make ingest BRING_BACK=<tar>
(Or run everything, phase 3 included, on the GPU host: bash run.sh all, then copy ONE tar home.)
MSG

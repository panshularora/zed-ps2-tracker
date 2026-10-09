#!/usr/bin/env bash
# Repo-root shortcut for the GPU scripts:  bash run.sh all --profile laptop|dgx  (see
# scripts/dgx/run_all.sh), or any per-dataset target: bash run.sh mot17 | mot20 | dancetrack ...
exec bash "$(dirname "${BASH_SOURCE[0]}")/scripts/dgx/run.sh" "$@"

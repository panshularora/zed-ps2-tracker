#!/usr/bin/env bash
# Download the Pexels demo clips listed in configs/demo/clips.yaml into data/demo/raw (gitignored),
# fetch the ONNX models, and write the face-blurred copies to data/demo/blurred.
# Same as `make models clips blur`. Needs ffmpeg and `pip install -e .[serve]`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-.venv/bin/python}"
"$PY" -m serve.detector fetch --out data/models
"$PY" -m serve.clips fetch --out data/demo/raw "$@"
"$PY" -m serve.blur --src data/demo/raw --out data/demo/blurred --models data/models

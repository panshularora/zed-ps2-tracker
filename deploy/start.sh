#!/bin/sh
# PaaS (Render, Fly, Hugging Face Spaces) sets PORT. Local Docker defaults to 8000.
set -eu
PORT="${PORT:-8000}"
exec uvicorn serve.app:app --host 0.0.0.0 --port "$PORT"

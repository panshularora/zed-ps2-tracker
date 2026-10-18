# Deploy

One Docker image serves the React app and the FastAPI job API.

## Live (Render)

The repo root has `render.yaml` and a `Dockerfile` that honours `$PORT`.

1. Push `main` to GitHub.
2. On [Render](https://render.com) → New → Blueprint → this repo.
3. Health check: `GET /api/health`.
4. YOLOX-tiny downloads on first use (`ZED_AUTO_DOWNLOAD=1`).

Local equivalent:

```bash
docker build -t zed-ps2-demo .
docker run --rm -p 8000:8000 -e PORT=8000 zed-ps2-demo
```

`docker compose up --build` is the two-service path (API `:8000`, nginx `:8080`) when `data/models` and blurred clips are on disk.

## What the image contains

The built web app (`web/dist`), MOT17 viewer, tracker code, and CPU ONNX runtime. Model weights and raw videos are not baked in; the detector fetches YOLOX-tiny on first job if the cache is empty.

## PaaS notes

Render, Fly, and Hugging Face Spaces set `PORT`. `deploy/start.sh` reads it. Do not pin uvicorn to 8000 on those hosts.

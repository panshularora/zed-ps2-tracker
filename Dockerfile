# Backend image: FastAPI job service + CPU detector + the built web app (and the MOT17 viewer).
# No GPU, no datasets, no videos and no model weights in the image: mount them (docker-compose.yml).
#   docker build -t zed-ps2-demo .
#   docker run --rm -p 8000:8000 -v $PWD/data/models:/data/models:ro \
#       -v $PWD/data/demo/blurred:/data/demo/blurred:ro zed-ps2-demo
#   -> http://localhost:8000/ (web app), /viewer (MOT17 viewer), /docs (API)

# ---- stage 1: static MOT17 viewer (single self-contained index.html, clips embedded)
FROM node:20-slim AS viewer
WORKDIR /src/viewer
COPY viewer/package.json viewer/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY viewer/ ./
RUN npm run build

# ---- stage 2: web app (imports viewer/src/{theme,media}.ts through the @viewer alias)
FROM node:20-slim AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY viewer/src/ /src/viewer/src/
COPY web/ ./
RUN npm run build

# ---- stage 3: runtime
FROM python:3.11-slim
# ffmpeg: re-encoding face-blurred uploads (serve/blur.py) and probing
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    ZED_CONFIG_ROOT=/app/configs ZED_VIEWER_DIST=/app/viewer ZED_WEB_DIST=/app/web \
    ZED_RESULTS_DIR=/app/docs/results ZED_CLIP_MANIFEST=/app/configs/demo/clips.yaml \
    ZED_DATA_DIR=/data/serve ZED_CLIP_DIR=/data/demo/blurred ZED_MODEL_DIR=/data/models \
    ZED_CLIP_ROOT=/app/clips ZED_RESULTS_ROOT=/app/docs/results
WORKDIR /app
COPY serve/requirements.txt serve/requirements.txt
RUN pip install --no-cache-dir -r serve/requirements.txt
COPY tracker/ tracker/
COPY eval/ eval/
COPY models/ models/
COPY serve/ serve/
COPY configs/ configs/
COPY docs/results/ docs/results/
# MOT17 viewer clips for the workstation screens (/dashboard/studio, /api/viewer-clips)
COPY viewer/data/ clips/
COPY --from=viewer /src/viewer/dist/ viewer/
COPY --from=web /src/web/dist/ web/
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data/serve /data/demo/blurred /data/models && chown -R app /data
USER app
VOLUME ["/data/serve"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]

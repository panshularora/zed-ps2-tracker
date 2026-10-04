# CPU demo service: FastAPI + the built viewer. No GPU, no detector, no datasets in the image.
#   docker build -t zed-ps2-demo .
#   docker run --rm -p 8000:8000 zed-ps2-demo      ->  http://localhost:8000/  (viewer)
#                                                       http://localhost:8000/docs (API)

# ---- stage 1: static viewer (single self-contained index.html, clips embedded)
FROM node:20-slim AS viewer
WORKDIR /viewer
COPY viewer/package.json viewer/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY viewer/ ./
RUN npm run build

# ---- stage 2: runtime
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    ZED_CONFIG_ROOT=/app/configs ZED_VIEWER_DIST=/app/viewer
WORKDIR /app
COPY serve/requirements.txt serve/requirements.txt
RUN pip install --no-cache-dir -r serve/requirements.txt
COPY tracker/ tracker/
COPY eval/ eval/
COPY models/ models/
COPY serve/ serve/
COPY configs/ configs/
COPY --from=viewer /viewer/dist/ viewer/
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]

# serve/: CPU demo service

FastAPI app (`serve/app.py`) that replays a detection cache through any tracker config and returns
**gate events** in the ZIG contract (`serve/events.py`, `docs/integration.md` section 2), together
with a viewer clip and the tracker's own latency. The MOT research dashboard (`dashboard/`,
`make dashboard`) is served at `/` when built; the older static viewer is at `/viewer`.

It is an **offline replay demo**. It does not decode video and does not run a detector (that needs a
GPU). Its input is the detector output an edge box would produce: a Stage A `.npz` cache (format in
`tracker/io.py`). Uploads are processed in a temp dir and deleted. Nothing is stored or sent anywhere.

```bash
uv pip install -e ".[serve]"       # or: pip install -r serve/requirements.txt
make viewer                        # optional: builds viewer/dist for the UI at /
make serve                         # http://127.0.0.1:8000  (API docs at /docs)

curl -s localhost:8000/api/configs
curl -s -F cache=@data/stagea/<...>/MOT17-04-FRCNN.npz -F config=ours/visibility \
     -F gate=0,648,1920,648 -F start_time=2026-10-05T09:00:00+05:30 localhost:8000/api/track
```

Docker (CPU only, Python 3.11 slim, viewer built in a node stage):

```bash
docker build -t zed-ps2-demo .
docker run --rm -p 8000:8000 zed-ps2-demo
```

| Endpoint | |
|---|---|
| `GET /api/health` | status, version, number of configs, whether the viewer/dashboard is bundled |
| `GET /api/dashboard` | clip count and which published result files are present |
| `GET /api/configs` | names accepted by `config` (`baselines/*`, `ours/*`); paths are never accepted |
| `GET /api/config/{name}` | parsed YAML for a name from `/api/configs` |
| `GET /api/clips` | viewer clips in `viewer/data`; `GET /api/clips/{file}` the gzip |
| `GET /api/results/ablation` | `docs/results/ablation/ablation.json` |
| `GET /api/results/gate-events` | `docs/results/gate_events/gate_events.json` |
| `GET /api/results/go-no-go` | `docs/results/go_no_go/rows_v2.json` |
| `GET /api/outputs` | COMBINED metrics from local `outputs/*/metrics/*/metrics.json` |
| `POST /api/track` | multipart `cache` (.npz, ≤ `ZED_MAX_UPLOAD_MB`, default 200), `config`, optional `gate` `x1,y1,x2,y2` (default: horizontal line at 60 % height), optional `start_time` (ISO 8601 with offset; default 1970-01-01T00:00:00+00:00 = replay time) |

Response: `{"sequence", "config", "events": [...], "latency": {"frames", "p50_ms", "p95_ms",
"machine"}, "clip": {viewer clip}}`. In the clip, crossings have `correct: null` because there is
no ground truth for an uploaded cache.

Tests: `tests/test_serve.py` covers the event rules, the endpoints, bad input and the upload limit.

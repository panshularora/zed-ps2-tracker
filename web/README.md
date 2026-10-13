# web/ — product frontend

React 18 + Vite 6 + TypeScript. Four pages, all reading the `serve/` API:

| Page | What it shows | Data |
|---|---|---|
| **Overview** `/` | One sentence over a still of the blurred turnstile clip; IDF1, ID switches and tracker p95 (or `—` when not measured); a button to the demo | `GET /api/benchmarks`, `GET /api/clips/{id}/poster.jpg` |
| **Demo** `/demo` | Pick a blurred clip (or upload one), run a CPU tracking job with progress, then play it back with boxes, persistent IDs, 30-frame trails, the gate line, live IN/OUT counters, a 300 ms ring when a track comes back with its ID, an event log and a scrubber with event ticks | `/api/clips`, `/api/uploads`, `/api/jobs*` |
| **Results** `/results` | MOT17 tables first (ours with a blue bar, baselines grey), gate events, visibility estimate, the delayed-decision sanity check, the pending-DGX list, then stress curves | `GET /api/benchmarks` (built from `docs/results/`) |
| **Architecture** `/architecture` | camera → detector → tracker → gate logic → event API; click a box for its payload (from the latest job, or the contract examples in `docs/integration.md`) | `GET /api/jobs`, `/api/jobs/{id}/result` |

Stock footage has no consent, so for `event_styling: neutral` clips the UI shows only IDs,
crossings, counts and recoveries: no tailgate/ambiguous-hold/"swap" alerts and no red.
Alert styling is used only for uploads where the uploader ticked "people in my video consented".

## Run

```bash
npm ci
npm run dev            # http://localhost:5173, proxies /api to VITE_API_TARGET (default http://127.0.0.1:8000)
npm run build          # web/dist; `uvicorn serve.app:app` serves it at / when present
npm run preview        # serve web/dist on :4173 with the same /api proxy
```

From the repo root, `make web` builds this and `make demo` starts backend + frontend.

## Checks

```bash
npm run lint && npm run typecheck && npm test
```

Vitest (jsdom) covers frame/time mapping with strides, trails, counters, recovery rings,
neutral-event filtering, the API client's structured errors, and render tests for Overview,
Results and the Architecture diagram.

## Screenshots

With a stack running (`make demo`):

```bash
BASE_URL=http://127.0.0.1:5173 OUT=../docs/media/screens node scripts/screenshots.mjs
```

It uses `playwright-core` with the system Chrome (`CHROME=/usr/bin/google-chrome`), opens every
page, and by default presses **Run tracking** on the turnstile clip and waits for the job
(`RUN_JOB=0` to reuse the latest finished job). It fails if the page logs any error.

## Code map

- `src/api.ts`: typed client; errors become `ApiError {status, code, message}`.
- `src/lib/playback.ts`: pure maths (video frame ↔ tracked frame with stride, trails, counts, rings, ticks). Reuses `viewer/src/media.ts` via the `@viewer` alias.
- `src/lib/draw.ts`: canvas overlay: gate, trails under boxes, boxes, ID tabs, rings.
- `src/pages/*`: one file per page; each exports a pure view for tests.

## Landing page and Scenarios tab

- `/` is a six-step landing page for non-specialists; the dashboard lives under `/dashboard`
  (`/dashboard/scenarios`, `/dashboard/demo`, `/dashboard/results`, `/dashboard/architecture`).
- Every number on the landing page comes from `src/lib/summary.ts`, which reads
  `docs/results/summary.json` at build time (`import.meta.glob`, so a missing file is fine).
  Expected shape: `{ "metrics": { "mot17.idsw_reduction_pct": 0, "mot20.idsw_reduction_pct": 0,
  "mot17.idf1_gain_pts": 0, "mot17.p95_ms": 0, "mot20.deta_delta_pts": 0 }, "source": "..." }`.
  A missing or non-numeric key renders "pending". No number is hard-coded.
- The Scenarios tab plays the six face-blurred Pexels clips (served by `/api/clips/<id>/video`
  from `data/demo/blurred`, gitignored) with overlays from `public/scenarios/*.json`, made by
  `.venv/bin/python scripts/demo/precompute_scenarios.py` (CPU YOLOX-S detections once per clip,
  then ByteTrack and ours on the identical detections). These clips have no ground truth: the
  ID-switch counts are the `serve.analysis.suspected_switches` heuristic, illustrative only.
  Two benchmark slots (MOT17/MOT20 val, human labels) are reserved and show as pending.

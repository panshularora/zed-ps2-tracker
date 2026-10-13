# dashboard/ — MOT research workstation (phase 3 UI)

React + Vite + TypeScript UI. Three screens, one flow:

1. **Watch** (`#/studio`) — pick a clip, drop a video, follow one person. The box colour stays
   put when the tracker gives them a new number. Amber fill marks overlapping people.
2. **Try my video** (`#/experiments`) — drop an MP4 (and optionally a Stage A `.npz`) then play
   or `POST /api/track`. The CPU demo does not detect people from video alone.
3. **Scores** (`#/analytics`) — MOT17 val-half ablation from `docs/results/`. Delayed assignment
   is a no-go.

```bash
cd dashboard && npm ci && npm run dev   # http://127.0.0.1:5173  (proxies /api → :8000)
make serve                              # API on :8000; after `make dashboard`, UI at /
make dashboard                          # production build → dashboard/dist
```

The CPU demo still does not decode video or run a detector on your MP4. Open an MP4 or numbered
JPEGs on Watch (or Try my video) for real frames. Boxes come from saved clips or from
`POST /api/track` with a detection cache.

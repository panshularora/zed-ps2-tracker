# Static tracker viewer

Vite + React + TypeScript. `npm run build` produces one self-contained `dist/index.html`: the JS,
the CSS and every clip in `data/*.json.gz` are inlined as base64 and decoded in the browser with
`DecompressionStream`. It opens from `file://` with no server. More clips can be opened at runtime
with "open clips".

```bash
make viewer            # export demo clips (MOT17-04, -09 val_half: gt, ByteTrack, BoT-SORT, delayed K=5) + build
cd viewer && npm test  # vitest: clip parsing/validation, gzip round trip, seeking, gate counts, held windows, palette
npm run screenshots    # headless Chrome -> ../docs/media/viewer_*.png (CHROME=/path overrides /usr/bin/google-chrome)
```

What it shows:
- Two synced panels side by side, stacked on narrow screens, drawn on a blank canvas (no video).
- Boxes in the ID palette with label tabs.
- Held (delayed) decisions as dashed white boxes with a "held n" countdown in frames.
- Boxes that just took part in an ID switch, drawn in the error colour.
- The gate line, which pulses after a crossing, and in/out counters.
- A timeline with event marks: ID switches and wrong crossings in red, correct crossings in amber,
  tailgates in white. Clicking a mark jumps to 2 s before the event. `#f=<frame>` deep-links.

Theme tokens are in `src/theme.ts` and the clip format is in `src/data.ts` (it matches
`eval/export_viewer.py`). The decision fade from dashed to solid (150 ms) is **not implemented**:
held boxes switch to solid when the window closes.

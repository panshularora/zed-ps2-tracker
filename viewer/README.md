# Static tracker viewer

Vite + React + TypeScript. `npm run build` produces one self-contained `dist/index.html`: the JS,
the CSS and every clip in `data/*.json.gz` are inlined as base64 and decoded in the browser with
`DecompressionStream`. It opens from `file://` with no server. More clips can be opened at runtime
with "open clips".

```bash
make viewer            # export demo clips (MOT17-04, -09 val_half: gt, ByteTrack, BoT-SORT, delayed K=5) + build
cd viewer && npm test  # vitest: clip parsing/validation, gzip round trip, seeking, gate counts, held windows, palette
npm run screenshots    # headless Chrome -> ../docs/media/viewer_*.png (CHROME=/path overrides /usr/bin/google-chrome)
npm run e2e:media      # headless Chrome + ffmpeg: real-frame sync check (synthetic MP4 and JPEGs)
```

What it shows:
- Two synced panels side by side, stacked on narrow screens. Under the boxes: the real frames when
  available, else a blank canvas (see "Real frames" below).
- Boxes in the ID palette with label tabs.
- Held (delayed) decisions as dashed white boxes with a "held n" countdown in frames.
- Boxes that just took part in an ID switch, drawn in the error colour.
- The gate line, which pulses after a crossing, and in/out counters.
- A timeline with event marks: ID switches and wrong crossings in red, correct crossings in amber,
  tailgates in white. Clicking a mark jumps to 2 s before the event. `#f=<frame>` deep-links.

- When a held window closes, the box fades from dashed white to its solid ID colour over 150 ms
  of **video time** (`theme.decisionFadeMs`; at 25 fps the next 3 frames are part-faded), so the fade is
  the same when stepping, scrubbing or exporting frames.

Theme tokens are in `src/theme.ts` and the clip format is in `src/data.ts` (it matches
`eval/export_viewer.py`).

## Real frames

A clip may carry `media: {type: "video" | "frames", src, frame_offset}` (written by
`eval/export_viewer.py --media SEQ=SRC`; `frame_offset` maps val-half clip frames to the frames
of the full-sequence MP4 or `img1/` JPEGs). `src` is resolved relative to `index.html`, so put
`clips/MOT17-04.mp4` next to `dist/index.html`. You can also pick a video or a set of JPEGs named
by frame number (`000527.jpg`) at runtime with **open video / frames**. If the video fails to load,
or a JPEG is missing, that frame falls back to the blank canvas.

Sync (`src/media.ts`, unit tests in `tests/media.test.ts`):
- one hidden `<video>` is the clock. While playing, frames come from
  `video.requestVideoFrameCallback((now, meta) => ...)` as
  `frame = Math.round(meta.mediaTime * fps) + 1`, and both panels are drawn **inside** that
  callback (`flushSync` + `useLayoutEffect`), so the boxes match the presented frame;
- when paused, the frame comes from `currentTime` (`floor(currentTime * fps) + 1`); pausing snaps
  to the centre of the frame that was on screen;
- seeking to frame `f` sets `currentTime = (f − 1 + 0.5) / fps`;
- without `requestVideoFrameCallback` (Firefox) the video stays paused and the app steps frames
  itself, seeking the video to each one. No `timeupdate`, no rAF loop reading `currentTime`.

`npm run e2e:media` checks this in headless Chrome with a synthetic H.264 MP4 whose frame *k* has
a known grey level: seeks to frames 5, 37, 64 and 100 draw exactly those frames; during playback
the drawn frame equals the frame counter in every sample; after pause they agree; JPEG frames draw
and a missing JPEG falls back to the blank canvas. The Firefox stepping path has **not** been run in a
browser (no Firefox on the CPU box); only the clock's fallback switch is unit-tested.

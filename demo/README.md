# demo/ (placeholder): static offline viewer

**Planned, not built.** It will be built in phase 3 from the outputs the pipeline already writes.

## What it is

A **static, offline** web viewer. It is a Vite + React build that opens from `index.html`
straight off disk (`file://`), with **no server** and no network calls. It can be zipped and
emailed to a judge, or opened from a USB stick at a venue.

## What it shows

* **Ours vs ByteTrack side by side:** the same clip in two synchronised panes, with boxes and
  persistent ID colours.
* **Timeline** under the video, with markers for
  * **ID switches** (from the stress/CLEAR matching, where gt exists),
  * **gate crossings** (from `eval/gate_events.py`), with direction.
* **Pending decisions:** while the delayed assignment holds a decision open, the competing
  hypotheses are drawn as **dashed links** between the track and its candidate boxes. They
  resolve to solid when committed (from the `pending` field of the per-frame JSON).
* **Gate counters:** an entry line overlay with in/out counts and tailgating flags per tracker,
  next to gt counts where available.
* Playback controls: play/pause, frame step, speed, jump to next ID switch or crossing.

## Inputs (already produced by the pipeline)

| Input | Producer |
|---|---|
| `clips/*.mp4` (720p H.264, `+faststart`) | `scripts/dgx/run.sh` (ffmpeg settings in `config.env`) |
| `<seq>.frames.jsonl` (`zed-ps2-frames/v1`: tracks + pending) per tracker | `python -m tracker.cli ... --jsonl` |
| gate line YAML + crossing events | `eval/gate_events.py` (phase 2b) |
| ID-switch events | `eval/stress.py` per-box outcomes (export to be added) |

A small build script will bundle the selected clips and JSONL next to `index.html`, with data
inlined as JS modules so that `file://` works without fetch/CORS issues.

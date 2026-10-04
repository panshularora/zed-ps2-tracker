# Integration proposal: identity-preserving tracking for ZIG gates

**Status: proposal.** Every number below names its source in this repo. GPU-detector numbers
(Stage A on the DGX, 5 Oct 2026) are not in yet and appear as **⟨GPU: …⟩** placeholders. Phase-2
numbers are **public-detection baselines** (MOT17 val_half, public FRCNN boxes): they compare
trackers on the same boxes, not the detector we would ship. `make phase3` does not compute gate
events yet, so the GPU gate-metric cells need `eval.gate_events` run on its MOT17 outputs.

## 1. Placement: an edge box between the gate camera and ZIG's decision logic

```
gate camera ──frames──▶ [ edge box: detector → tracker → gate events ] ──events (JSON)──▶ ZIG decision logic
                         (on site; video never leaves it)                                  (validation, Fast Lane,
                                                                                            fare-evasion checks)
```

- **Frames in, gate events out.** The box takes the camera's stream and emits a few small JSON
  events per passenger. ZIG keeps all of its own decisions: the box only says who crossed, which
  way, and how sure it is.
- **Video stays on site.** Frames are processed in memory and dropped. There is no face
  recognition and no identity database, and no video is stored or uploaded. Body-appearance
  embeddings, if used, exist only for the life of a track (`docs/integration/README.md`,
  privacy note).
- **One box per gate group.** Tracker cost is small next to the detector: tracker-only p95 is
  1.5–2.7 ms per frame on one CPU thread (section 4). Detector throughput on the edge hardware
  is **⟨GPU: detector fps / p95 ms per frame, from the Stage A log⟩**.
- What exists today is `serve/` (`serve/README.md`). It is a CPU replay of the same
  pipeline from the tracker onward: you upload a detection cache and get the events below back.
  It runs in Docker (`Dockerfile`).

## 2. Event contract

Small HTTP/JSON API. In the demo it is `POST /api/track` (batch replay). On a live box the same
objects would be pushed per event. Implementation and rules: `serve/events.py`; tests:
`tests/test_serve.py`.

Every event has:

| field | type | meaning |
|---|---|---|
| `type` | `crossing` \| `tailgate` \| `ambiguous_hold` | |
| `track_id` | int | tracker identity, stable for the life of the track (not a person ID) |
| `direction` | `in` \| `out` | side change across the gate line (hysteresis 10 px) |
| `timestamp` | ISO 8601 with offset | wall time of the frame: camera start time + (frame − 1) / fps |
| `confidence` | 0–1 | heuristic, **not calibrated**: min(mean box score, min visibility) over the 0.5 s before the crossing |
| `frame`, `gate` | int, str | for audit and replay |

- **`crossing`**: a confirmed track crossed the line, and the tracker was sure of it.
- **`tailgate`**: a second track crossed the same line in the same direction within 1.0 s of a
  first. It adds `lead_track_id` and `gap_s`.
- **`ambiguous_hold`**: means *tracker unsure, don't guess*. A crossing happened, but in the
  0.5 s before it the track's estimated visibility fell below 0.5 (occlusion) or its mean score
  was below 0.5 (low confidence). It adds `reasons`. ZIG should apply its own fallback (for
  example, ask for a tap or route to staff) instead of trusting the identity. A held crossing is
  never also sent as a `crossing` and takes no part in tailgate pairs. This uses only the online
  tracker's own per-frame state. **It does not use the delayed (look-ahead) decision,** which
  failed its go/no-go (`docs/results/go_no_go.md`).

Example payloads. These are real outputs of `ours/visibility` (ByteTrack-style tracker plus the
visibility estimate) on MOT17 val_half public FRCNN boxes, with `start_time` set to
2026-10-05 09:00 IST:

```json
{"type": "crossing", "track_id": 5, "direction": "in", "timestamp": "2026-10-05T09:00:07.233+05:30",
 "confidence": 1.0, "frame": 218, "gate": "h60"}
```
(MOT17-04)

```json
{"type": "tailgate", "track_id": 41, "direction": "out", "timestamp": "2026-10-05T09:00:13.066+05:30",
 "confidence": 1.0, "frame": 393, "gate": "h60", "lead_track_id": 39, "gap_s": 0.233}
```
(MOT17-04)

```json
{"type": "ambiguous_hold", "track_id": 7, "direction": "out", "timestamp": "2026-10-05T09:00:06.233+05:30",
 "confidence": 0.093, "frame": 188, "gate": "h60", "reasons": ["occlusion"]}
```
(MOT17-09)

How often the hold fires (exploratory, on 19 candidate crossings): **4 of 19** tentative
crossings were held on the MOT17 gates (MOT17-02/-04/-09). Of those 4, 1 was a correct
crossing, 1 had the wrong identity and 2 had no matching ground-truth crossing. Holding them
lowers attribution error on the remaining matched crossings from 4/15 to 3/13 (26.7 % → 23.1 %).
The 0.5 thresholds were set before looking. **This sample is far too small to claim anything.**
It shows the mechanism, not a benefit. Baseline trackers have no visibility estimate and only
hold on low scores, which never fired on these boxes. The pilot (section 5) is where this gets
measured.

## 3. What each ZIG flow gains

Only three flows, each tied to one gate metric from `eval/gate_events.py`. Results are in
`docs/results/gate_events/` (MOT17 val_half, public FRCNN, frozen gates
`configs/gates/mot17_val_half.yaml`). **Only 25 ground-truth crossings and 9 tailgates**, so
one event is 4 points; these numbers show the metric works, they do not rank trackers.

| ZIG flow | What goes wrong today | Gate metric | Baselines (public dets) | Ours |
|---|---|---|---|---|
| **Hands-free entry** (charge the right account as the person walks through) | the validation is attached to the wrong person when two people overlap | **attribution error**: matched crossings given to the wrong identity | 14.3–33.3 % | ⟨GPU: attribution error, final tracker⟩ |
| **Fast Lane** (a second person following through on one validation) | tailgates go unflagged | **tailgate recall** | 44.4 % (precision 80.0 %) | ⟨GPU: tailgate P / R⟩ |
| **Fare-evasion checks** (people through vs validations) | the passenger count drifts | **count error**: \|tracker crossings − gt crossings\| / gt | 28.0–40.0 % | ⟨GPU: count error⟩ |

In addition, `ambiguous_hold` gives each flow an explicit "don't know" instead of a silent wrong
answer.

Supporting tracker metrics, also public-detection baselines (`docs/results/ablation/ablation.md`):
BoT-SORT (CMC) + visibility reaches HOTA 52.5 / IDF1 59.8 / IDSW 180, against BoT-SORT's
51.6 / 58.8 / 197. The visibility estimate gives the same result as a 90-frame track buffer
control (52.4 / 59.5 / 180), so it has **not** been shown to beat simply keeping lost tracks
longer. GPU-detector rows: ⟨GPU: `docs/results/phase3/<id>/ablation.md`⟩.

## 4. Honest limits

- **Latency (measured, CPU).** These are tracker-only `update()` times, one thread, on this CPU
  box (`docs/results/ablation/ablation.md`, MOT17 val_half):
  - baselines: p95 1.55–2.66 ms
  - + visibility: 2.19 ms
  - BoT-SORT + visibility: 2.27 ms (max 3.0 ms)

  In the Docker demo, MOT17-04 val-half gave p95 3.6 ms per frame for `ours/visibility` (one
  request, CPU). The detector is extra: ⟨GPU: detector ms/frame⟩. The delayed decision would add
  K/fps of algorithmic delay (199 ms at K = 5) plus p95 18–20 ms of compute. That is one more
  reason it is not in the product path.
- **Non-commercial benchmarks.** MOT17 and MOT20 (CC BY-NC-SA 3.0), the DanceTrack videos and
  SOMPT22 are all non-commercial, and so is everything the YOLOX weights were trained on
  (`docs/licenses.md`). The weights and numbers here are for research. A product model has to be
  retrained on licensed or consented footage.
- **Fine-tuning on their own cameras is required.** Benchmark scenes are not gate scenes: the
  camera height, angle, lighting and crowd flow all differ. Expect to fine-tune the detector
  (and re-ID, if used) on labelled ZIG footage before any number here carries over. The gate
  lines are virtual and placed by image geometry, not real gate positions.
- **Small gate sample.** 25 crossings (section 3). The pilot exists to fix this.
- The `confidence` field is a heuristic and is not calibrated against error rates yet.

## 5. Pilot: one gate, two weeks

- **Scope:** one gate (one camera) with the edge box in shadow mode. It emits events, and ZIG
  logs them but **acts on none of them**.
- **Ground truth:** hand-labelled crossings from the same video, labelled on site, covering
  direction, identity continuity through the gate, tailgates, and the time of each validation.
  A labelling protocol is agreed before data collection.
- **Metrics:** attribution error, tailgate precision/recall and count error, each with a 95 %
  CI; the `ambiguous_hold` rate; and edge-box p95 latency.
- **Go/no-go agreed up front:** before the pilot starts, ZIG and we sign off on one number: the
  maximum attribution error ⟨to agree, e.g. ≤ X % with the CI upper bound below Y %⟩ at a hold
  rate ≤ ⟨Z %⟩. If it is met, move to a live trial with ZIG acting on `crossing`. If not, stop,
  or fine-tune and repeat once. The threshold is not moved after the data is seen.
- **Needed from ZIG:** a camera feed at one gate, footage-handling consent and on-site storage,
  labelling time (or footage access for us to label on site), and the validation log to align
  timestamps.

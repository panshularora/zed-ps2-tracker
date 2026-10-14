# Demo video: script and shot list (2–3 min)

The structure follows `docs/integration.md`. Everything is recorded from the real viewer
(`viewer/`) playing real frames (`clip.media`, `viewer/README.md` "Real frames"). There are no
mock-ups.

**Recording spec:**
- 1920×1080 at 60 fps, browser in kiosk/fullscreen, **no cursor** (`--hide-cursor` or move it
  off-screen; check every take).
- All on-screen text ≥ 32 px. Viewer theme: #0E1013 background, Inter, tabular numbers. Baseline
  #8A929C, ours #56B4E9, error #E5484D.
- Clips are 30 fps (MOT17-04 / -09) and are shown at 60 fps by frame doubling. The 0.25x shots
  play each source frame for 8 output frames, so the counter must step cleanly; check
  `npm run e2e:media` first.
- **Frames:** use the MP4s in `clips.tar` from the DGX bundle (`make ingest`). Export with
  `python -m eval.export_viewer --media SEQ=clips/<seq>.mp4 ...`.
- **"Ours"** is the configuration that wins on GPU data in `docs/results/phase3/<id>/ablation.md`:
  **⟨GPU: final tracker name⟩**. Until that is known, it is labelled ⟨OURS⟩ below.

| # | Time | Shot | On screen | Voice-over (draft) |
|---|---|---|---|---|
| 1 | 0:00–0:12 | **Raw clip** | MOT17-04 frames only, no boxes, no UI. Gate camera framing. | "A busy gate. ZIG has to know who walked through, which way, and whether someone followed them." |
| 2 | 0:12–0:22 | **Boxes on** | Boxes switch on; the gate line pulses once (opacity 0.4 → 1 → 0.6 over 600 ms, gate colour); in/out counters appear. | "An edge box next to the camera turns frames into people with stable IDs, and into gate events." |
| 3 | 0:22–0:50 | **Problem: the ID swap** | Two lookalikes crossing paths, **0.25x**, frame counter top-right. Baseline only. At the swap, both boxes flash #E5484D (2 × 150 ms), with a label such as "ID 12 → 31" (use the real IDs). The moment is a frame where the baseline has an `id_switch` event and ⟨OURS⟩ has none (find it with "next ID switch" on both panels): **⟨GPU: seq + frame⟩**. | "When two similar people overlap, a standard tracker swaps their identities. For hands-free entry, that charges the wrong person." |
| 4 | 0:50–1:10 | **Ours, same moment** | Same frames at 0.25x, ⟨OURS⟩ alone: IDs hold through the overlap. | "Our tracker keeps each identity through the occlusion." |
| 5 | 1:10–1:40 | **Side by side** | Synced panels (baseline left, #8A929C label; ours right, #56B4E9) at 1x, with live ID-switch counters under each. | "Across the whole clip, ⟨GPU: n⟩ swaps against ⟨GPU: m⟩." |
| 6 | 1:40–2:10 | **Events** | On each crossing, a JSON card slides in from the right (200 ms ease-out) showing the real `serve/events.py` payload: `crossing`, then a `tailgate` (two people within 1 s), then an `ambiguous_hold` where visibility drops. | "Each crossing becomes a small event for ZIG: a crossing, a tailgate, or 'unsure, don't guess', so ZIG never acts on a guess." |
| 7 | 2:10–2:35 | **Close** | Three lines with metrics: attribution error, tailgate recall, count error, with ⟨GPU⟩ values and a "public benchmark, non-commercial data" footnote. Then: "Pilot: one gate, two weeks." | "Next step: one gate, two weeks, against hand-labelled crossings, with a pass mark we agree before we start." |

**Conditional shot (only if go/no-go v2 passes).** The "dashed box while holding" shot shows a
held decision as dashed candidate boxes that fade to solid over 150 ms
(`docs/media/viewer_mot17-04_decision_fade.png`). It depends on the **delayed decision**. Include
it (as 4b, about 15 s, before shot 5) **only if** `docs/results/phase3/<id>/go_no_go_v2.md`
says GO. Otherwise, leave it out and give shot 4 to the method that actually helps on GPU data
(per the phase-3 ablation, e.g. re-ID or visibility gating), showing what that method does at
the swap. Phase-2 v1 was a no-go (`docs/results/go_no_go.md`).

The `ambiguous_hold` card in shot 6 does **not** depend on the delayed decision (it comes from
visibility/score state, `docs/integration.md` §2). It can always be shown.

**Partly built.** The web app's live run panel on the Scenarios tab (`make demo`, `web/`) now has the gate-line pulse
(shot 2), live in/out counters, a 300 ms ring when a track comes back with its ID, an event log
and a scrubber with event ticks, on real (face-blurred) footage. Still missing: the swap flash
(shot 3), per-tracker live ID-switch counters (shot 5) and the sliding JSON cards (shot 6).
Composite those in editing over the real recording, using event frames from the clip JSON,
`/api/track` or a job's `events.json`. Pexels footage must stay neutral (no swap / tailgate /
fare-evasion labels, no red): use it only for shots 1–2 and the counters; alert shots need the
consented gate clip.

**Before recording, check:**
- [ ] clips.tar ingested; the MP4s play in the viewer and the frame counter matches the clip
      (`npm run e2e:media` passes)
- [ ] ⟨OURS⟩ chosen from the phase-3 tables; the swap frame for shot 3 found
- [ ] go/no-go v2 verdict read; shot 4b included or dropped
- [ ] no cursor, text ≥ 32 px, 1080p60, real frames only
- [ ] numbers on screen copied from the repo files named in `docs/deck_outline.md`

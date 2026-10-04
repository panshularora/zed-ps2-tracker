# Deck outline (about 10 slides, 16:9)

**Theme** (same as the viewer, `viewer/src/theme.ts`):
- background #0E1013, Inter, tabular numbers
- baseline #8A929C, ours #56B4E9, error #E5484D

**Rules:**
- Each headline is a **finding stated as a sentence**, with **one chart per slide**.
- Images are **real viewer frames only**, screenshots from `viewer/` (`npm run screenshots`). The
  current `docs/media/*.png` are on a blank canvas; re-shoot them over real frames once
  clips.tar is ingested.
- Each slide names the repo file its number comes from. **⟨GPU: …⟩** marks a number still
  missing.
- Phase-2 numbers are labelled **"public-detection baseline"**.
- **Go/no-go results** carry a grey **"sanity check"** tag (#8A929C outline pill) and never
  appear inside a results table.

The sections follow `docs/integration.md`: placement, contract, gains, limits, pilot.

| # | Section | Headline (finding as a sentence) | One chart | Source |
|---|---|---|---|---|
| 1 | Title | "Gate tracking that knows when it isn't sure" | real viewer frame, MOT17-04, boxes on | `docs/media/viewer_mot17-04_bytetrack_vs_botsort.png` (re-shoot on real frames) |
| 2 | Problem | "A standard tracker gives 14–33 % of gate crossings to the wrong person" | bar: attribution error per baseline (#8A929C), with a "public-detection baseline · 25 crossings" footnote | `docs/results/gate_events/gate_events.md` |
| 3 | Placement (§1) | "An edge box turns gate video into events, and the video never leaves the site" | block diagram: camera → edge box (detector → tracker → events) → ZIG | `docs/integration.md` §1 |
| 4 | Contract (§2) | "ZIG gets three events and an explicit 'don't guess'" | three JSON cards: crossing / tailgate / ambiguous_hold (real payloads) | `docs/integration.md` §2, `serve/events.py` |
| 5 | Gains: hands-free entry (§3) | "⟨GPU: Ours cuts attribution error from A % to B %⟩" | bar: attribution error, baseline vs ours | ⟨GPU: gate events on the phase-3 MOT17 runs (not yet automated, see `docs/next_gpu_session.md`)⟩ (public-det today: 14.3–33.3 % baselines, 26.7 % BoT-SORT + vis) |
| 6 | Gains: Fast Lane (§3) | "⟨GPU: Ours flags C % of tailgates against 44 %⟩" | bar: tailgate recall (with precision as a label) | ⟨GPU⟩; public-det: 44.4 % R / 80.0 % P, `gate_events.md` |
| 7 | Gains: fare-evasion checks (§3) | "⟨GPU: Count error falls from D % to E %⟩" | bar: count error | ⟨GPU⟩; public-det: 28–40 %, `gate_events.md` |
| 8 | Evidence under occlusion | "⟨GPU: the gain grows as visibility drops⟩", or, if it does not, "the gain is flat across visibility" | line: IDSW rate by visibility bin, baseline vs ours | `docs/results/phase3/<id>/stress_visibility.png`; public-det version `docs/results/ablation/stress_visibility.png` |
| 9 | Honest limits (§4) | "It runs in 2–3 ms per frame on one CPU thread, but it is trained on non-commercial data" | bar: p95 latency per tracker (tracker only) + ⟨GPU: detector ms⟩ stacked | `docs/results/ablation/ablation.md` (p95 1.55–2.72 ms); `docs/licenses.md` |
| 10 | Pilot (§5) | "One gate, two weeks, pass mark agreed up front" | timeline: week 0 (agree threshold), weeks 1–2 (shadow mode + labelling), decision | `docs/integration.md` §5 |
| A1 | Appendix | "Delayed decision: no-go on MOT17" **[sanity check]** | IDSW vs K curve | `docs/results/go_no_go/curve.png`, `docs/results/go_no_go.md`; v2: ⟨GPU: `go_no_go_v2.md`⟩ |
| A2 | Appendix | "Our ByteTrack reproduction matches the paper within tolerance" | MOTA/IDF1/IDs ours vs paper | ⟨GPU: `docs/results/phase3/<id>/reproduction.md`⟩ |

**Notes:**
- Slide 5–7 headlines are filled in only from GPU-detector gate events. `make phase3` does **not** compute gate events yet: run `eval.gate_events` on its MOT17 val_half outputs. If ours does not beat the
  baseline on a metric, say so in the headline ("Attribution error is unchanged at …").
- On slides 5–7, use the 95 % CI or the event counts (25 crossings / 9 tailgates on MOT17) as an
  annotation. With so few events, a 1-event difference is 4 points.
- The visibility estimate equals the 90-frame buffer control on public detections
  (`ablation.md`: 52.5 vs 52.4 HOTA). Do not claim it as a gain unless the GPU tables separate
  them.
- The `ambiguous_hold` example (slide 4) is real. The "4 of 19 held" figure
  (`docs/integration.md` §2) can be a footnote; it is too small for a chart.
- If go/no-go v2 is GO, the dashed-hold frame (`docs/media/viewer_mot17-04_decision_fade.png`)
  may go on slide 8. It still carries the "sanity check" tag, and its numbers stay out of the
  tables.

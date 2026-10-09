# docs/integration (placeholder)

The integration proposal is [../integration.md](../integration.md); the demo video script is [../demo_video_script.md](../demo_video_script.md); the deck outline is [../deck_outline.md](../deck_outline.md). The rest of this table is still planned; see [../roadmap.md](../roadmap.md).

| Deliverable | What it will contain |
|---|---|
| **Zed integration write-up** | where the tracker sits in the ZIG pipeline: camera → detector → tracker → gate logic (validation link, Fast Lane entry, tailgating flag, passenger count). Includes the input/output contract (the `zed-ps2-frames/v1` JSONL schema, MOT txt), the latency budget (tracker p95 measured in `eval/latency.py`, detector fps from the Stage A log), and failure modes and fallbacks |
| **Privacy note** (in `docs/integration.md` and `docs/licenses.md`) | body-appearance embeddings kept in memory for the life of a track only, no face recognition, no identity database, nothing persisted. This matches Zed's own "preserving passenger privacy" wording |
| **Detailed docs** | an extended version of `docs/` for Zed engineers (API reference generated from the docstrings) |
| **PPT** | the presentation deck (problem → method → results with stress bins → gate demo → integration) |
| **Explainer video** | a 2–3 min narrated overview |
| **Code-walkthrough video** | a guided tour of `tracker/`, `eval/` and the Stage A/B split |
| **AI demo** | built: the web app's Demo page (`web/`, `make demo`) on face-blurred Pexels clips, CPU detector + tracker; a recording on the consented gate clip still waits for that clip |

Licensing constraint for anything product-facing: see [../licenses.md](../licenses.md). The
benchmark weights are trained on non-commercial data and must be replaced before production.

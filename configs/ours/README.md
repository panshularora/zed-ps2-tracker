# configs/ours/

Configs for our method. Each novel component is a flag on top of a baseline config.

| Config | What | Status |
|---|---|---|
| `visibility.yaml` | depth-order × score visibility, longer buffer for occluded lost tracks | measured: `docs/results/visibility/README.md` |

| `delayed.yaml` | K-frame look-ahead on ambiguous associations (near-online, ByteTrackLike only) | measured: `docs/results/go_no_go.md` |

`delayed.enabled: true` on any tracker other than ByteTrackLike raises, so a config can never
silently fall back to the baseline.

| `visibility_botsort.yaml` | BoT-SORT + the visibility estimate (the `ours_vis_botsort` ablation row written out as a file; default of the demo service) | measured: `docs/results/ablation/ablation.md` (HOTA 52.5 / IDF1 59.8 / IDSW 180) |

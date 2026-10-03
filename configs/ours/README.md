# configs/ours/

Configs for our method. Each novel component is a flag on top of a baseline config.

| Config | What | Status |
|---|---|---|
| `visibility.yaml` | depth-order × score visibility, longer buffer for occluded lost tracks | measured: `docs/results/visibility/README.md` |

| `delayed.yaml` | K-frame look-ahead on ambiguous associations (near-online, ByteTrackLike only) | measured: `docs/results/go_no_go.md` |

`delayed.enabled: true` on any tracker other than ByteTrackLike raises, so a config can never
silently fall back to the baseline.

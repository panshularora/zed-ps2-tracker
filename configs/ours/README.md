# configs/ours/

Configs for our method. Each novel component is a flag on top of a baseline config.

| Config | What | Status |
|---|---|---|
| `visibility.yaml` | depth-order × score visibility, longer buffer for occluded lost tracks | measured: `docs/results/visibility/README.md` |

`delayed.enabled: true` still raises `NotImplementedError` until the delayed-decision PR lands, so
a config can never silently fall back to the baseline.

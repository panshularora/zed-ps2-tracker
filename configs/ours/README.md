# configs/ours/

Configs for our method go here (later phases). Each novel component is a flag on top of
`configs/baselines/bytetrack.yaml`, e.g.

```yaml
inherit: ../baselines/bytetrack.yaml
visibility:   { estimator: depth_order }        # phase 2: overlap + depth-order visibility
appearance:   { enabled: true, gate_by_visibility: true }
association:  { delayed: { enabled: true, max_delay_frames: 3, margin: 0.05 } }
```

Nothing here is measured yet.

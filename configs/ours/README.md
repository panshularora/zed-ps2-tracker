# configs/ours/ (phase 2)

Configs for our method. Each novel component is a flag on top of a baseline config. The keys
below already exist in `tracker/config.py` (strict schema); the behaviour behind them lands in
phase 2. Today `depth_order` and `delayed.enabled: true` raise `NotImplementedError`, so a config
can never silently fall back to the baseline.

```yaml
# configs/ours/full.yaml (planned)
inherit: ../baselines/bytetrack_reid.yaml
visibility: { estimator: depth_order }        # overlap + bottom-edge depth order, from detections
appearance: { enabled: true, min_visibility: 0.5 }   # freeze embedding EMA when mostly hidden
delayed:    { enabled: true, max_delay_frames: 4, margin: 0.05 }   # K-frame bounded deferral
```

Nothing here has been measured yet. See [docs/algorithm.md](../../docs/algorithm.md) §6–7.

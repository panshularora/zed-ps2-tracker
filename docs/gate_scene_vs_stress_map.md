# Gate scene world fix vs the Stress Map

The Gate scene world fix (PR #45, `serve/gate_sim.py`) **does not touch the Stress Map generator**.

* **Two separate simulators.** The Gate scene runs `serve/gate_sim.py` (`World.seed_play`,
  `World.step`, `detect`). The Stress Map runs `eval/stress_sim.py` (`build_walkers`, `render`),
  driven by `eval/stress_run.py`; `serve/stress_live.py` and `eval/stress_metrics.py` import the
  same module. No file imports `serve.gate_sim` except `serve/app.py` (its router) and
  `scripts/web/export_gate_sim.py` (the offline recording), plus its own tests. `git diff 3fb44c4..feat/gpu-results`
  shows no change under `eval/`, `tracker/` or `docs/results/stress/`.
* **Same trackers.** Both use the real `tracker/` code with `configs/baselines/bytetrack.yaml` and
  `configs/ours/visibility_botsort.yaml`; no relabelling anywhere.
* **Re-run check (CPU, 18 Oct 2026, on this branch).** `python -m eval.stress_run grid`, reporting
  seeds 1000–1019, same stats rules, re-run into a scratch folder:
  * default map: 24 better / 14 worse / 6 unclear / 4 tie of 48, the same as the committed
    `stress_map.json`, with every per-seed value identical;
  * `--ours configs/ours/visibility_botsort_mem6.yaml`: 36 better / 1 worse / 10 unclear / 1 tie,
    the same as `stress_map_mem6.json`, with every per-seed value identical.

  The committed JSON files were not rewritten.
* **They agree.** In the Gate scene, #1 is hidden for 45 frames = 1.5 s (measured from the
  frames, shown on screen). On the Stress Map, ours is better in every cell with a hide of 1.0–2.5 s
  at every crowd size (5–40 people), and worse in most cells of 3 s or more (longer memory, mem6,
  is opt-in).

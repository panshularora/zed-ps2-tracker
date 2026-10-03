# configs/ablations/

* `manifest.yaml`: the rows of `make ablate` (`eval/ablate.py`). Each entry names a config and,
  optionally, dotted overrides. Every entry runs in its own output folder, in parallel, and then
  gets a single-threaded `eval.latency` pass. Result: `docs/results/ablation/`.
* `bytetrack_buffer90.yaml`: the control for `configs/ours/visibility.yaml`. It uses the same
  longer lost-track buffer, but for every track and without visibility.

A single row reproduces with
`make eval CONFIG=<config> NAME=<name>` plus the entry's overrides as `-o key=value`.

# configs/ablations/

One file per ablation row: each `inherit`s from a baseline or from `ours/` and flips exactly one
flag, so every row of the ablation table is reproducible with
`make eval CONFIG=configs/ablations/<name>.yaml NAME=<name> DETS=<cache dir>`.

Empty in phase 1.

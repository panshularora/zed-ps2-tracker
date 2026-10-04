# scripts/dgx: Stage A on the GPU

Day-of runbook (Windows laptop): [`docs/dgx_runbook.md`](../../docs/dgx_runbook.md). Reference: [`docs/dgx_guide.md`](../../docs/dgx_guide.md).

```bash
bash scripts/dgx/setup.sh                 # once: env, pinned sources, weights, GPU self-test
bash scripts/dgx/run.sh mot17             # then one per dataset (resumable):
bash scripts/dgx/run.sh mot20             #   mot17 | mot20 | sompt22 | dancetrack | pexels
bash scripts/dgx/run.sh video gate.mp4 gate
bash scripts/dgx/finalize.sh              # integrity check + bring_back/ bundle + sizes
bash scripts/dgx/pack.sh                  # a few tars to scp home (see docs/dgx_runbook.md)
```

| File | Role | Tested |
|---|---|---|
| `config.env` | paths, pinned commits, weight URLs + sizes, dataset URLs, clip list | URLs/sizes checked by HTTP HEAD on 4 Oct 2026 |
| `common.sh` | logging, resumable `fetch`, `unzip_once`, `stage_a`, `make_clip` | CPU dry run |
| `setup.sh` | env + sources + weights + self-test | **not run** (needs a GPU) |
| `selftest.py` | loads every checkpoint on the GPU | **not run** |
| `run.sh` | per-dataset Stage A | `sompt22` (synthetic archive) and `video` targets on CPU; others not run |
| `finalize.sh` | verification + bundle | CPU dry run |
| `pack.sh` | `bring_back/` → per-dataset uncompressed tars for `scp` from Windows (`models/pack.py`) | unit test + CPU end-to-end (`tests/test_ingest.py`) |
| `finetune_mot20_half.sh` | optional MOT20 first-half fine-tune | **not run** |
| `requirements-stage-a.txt` | Stage A pip deps | CPU venv |

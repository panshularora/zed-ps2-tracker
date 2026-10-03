# zed-ps2-tracker

**Identity-preserving, real-time, strictly online multi-object tracking through dense occlusion.**
Entry for the Zed Digital hackathon, problem statement 2.

> Status: **phase 1** (infrastructure + a ByteTrack-equivalent baseline). The novel components
> are not built yet, and **no headline number in this README has been measured yet.**

## Goal

Track every person through a crowded gate, online and in real time, so that:

* each person keeps **one stable ID** through partial and full occlusion;
* a person who **reappears** gets their **old ID back**;
* IDs **don't jump onto a neighbour** in tight groups;
* per-frame latency is **bounded** (we report p50 / p95 / max), and accuracy **degrades gracefully**
  as density and occlusion rise (we report a stress analysis, not just a single number).

Metrics: HOTA, IDF1, MOTA and ID switches from the official
[TrackEval](https://github.com/JonathonLuiten/TrackEval) code, plus IDF1 / IDsw binned by gt
visibility and by people per frame.

## Design: two stages

```
 Stage A (GPU, once per sequence)                Stage B (CPU, every experiment)
 ┌──────────────────────────────────┐            ┌──────────────────────────────────────┐
 │ frames → YOLOX (ByteTrack MOT    │   cache    │ tracker/  (NumPy, causal)            │
 │ weights) → boxes, scores         │ ─────────► │  Kalman → cost → Hungarian →         │
 │ crops  → OSNet → 512-d embedding │  .npz per  │  lifecycle → tracks → MOT txt        │
 │ (scripts/dgx/, models/)          │  sequence  │ eval/  TrackEval, stress, latency    │
 └──────────────────────────────────┘ (fp16 emb) └──────────────────────────────────────┘
```

Detection and re-ID run once per sequence on the GPU and are written to a compact cache. The
tracker then runs on CPU against that cache, so (1) every tracker, baselines included, sees
**identical detections and embeddings**, and (2) an ablation is seconds, not GPU-hours.

The tracker is assembled from small pieces behind interfaces (`tracker/interfaces.py`) so each
novel component is a config flag:

| Piece | Phase 1 (this repo) | Later phases (planned, not built) |
|---|---|---|
| Motion | Kalman, `xywh` state (BoT-SORT style) | occlusion-aware process noise |
| Cost | IoU, cosine appearance, BoT-SORT-style fusion | visibility-weighted appearance cost |
| Visibility | constant 1.0 | estimated from box overlap + depth order (lower bottom edge = in front), scored against gt visibility |
| Appearance memory | EMA embedding | EMA update gated by estimated visibility |
| Assignment | Hungarian (scipy / lapjv), two-stage high/low score (ByteTrack) | bounded-latency ambiguity-aware delayed assignment (online MHT-lite, ≤ K frames) |
| Lifecycle | tentative → confirmed → lost → removed | — |

**Strictly online.** Headline numbers use no future frames, no interpolation and no global linking.

## Repo layout

```
tracker/    the algorithm, framework-free (NumPy/SciPy), mypy --strict
eval/       TrackEval runner, stress binning, latency harness
models/     Stage A detector + re-ID wrappers (GPU)
scripts/dgx Stage A environment, dataset download, cache generation (run on the DGX)
configs/    baselines/, ours/, ablations/ (plain YAML with `inherit`)
third_party TrackEval (vendored, MIT)
tests/      pytest
ci/         GitHub Actions workflow (add it as .github/workflows/ci.yml through the GitHub UI)
serve/ dashboard/ deploy/   later phases
```

## Quick start

```bash
make setup        # .venv with Python 3.11 + dev tools (needs uv)
make check        # ruff + mypy --strict tracker/ + pytest
```

## Results

**Not yet measured.** Every cell below is a placeholder until Stage A caches exist.

| Method | Dataset / split | HOTA | IDF1 | MOTA | IDsw | AssA | p95 ms/frame |
|---|---|---|---|---|---|---|---|
| ByteTrackLike (IoU) | MOT17 val-half | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| ByteTrackLike + re-ID | MOT17 val-half | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| Ours | MOT17 val-half | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| Ours | MOT20 val | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| Ours | DanceTrack val | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |

## Related work (to compare against and cite)

ByteTrack (Zhang et al., 2021, arXiv:2110.06864), OC-SORT (Cao et al., 2022, arXiv:2203.14360),
BoT-SORT (Aharon et al., 2022, arXiv:2206.14651), Deep OC-SORT (Maggiolino et al., 2023,
arXiv:2302.11813), Hybrid-SORT (Yang et al., 2023, arXiv:2308.00783), MHT (Reid, 1979; Kim et al.,
ICCV 2015). Metrics: HOTA (Luiten et al., 2020), IDF1 (Ristani et al., 2016), CLEAR MOT (Bernardin
& Stiefelhagen, 2008).

# Licences: data, weights, code

**Bottom line.** Every dataset we evaluate on is **non-commercial**. Everything in this repo is a
**research benchmark**: nothing trained on, or tuned against, these datasets should ship in a Zed
product without a separate licensing review. The project itself has no LICENSE file yet; that is
Panshul's decision.

Checked October 2026. "Unverified" means we could not find an authoritative current statement.

## Datasets

| Dataset | Licence | Commercial use | Notes |
|---|---|---|---|
| MOT17 | CC BY-NC-SA 3.0 | ❌ | Confirmed only from a **2022 archived copy** of motchallenge.net. The live site went into archive mode (notice dated 16 Apr 2026) and no longer shows a licence page. |
| MOT20 | CC BY-NC-SA 3.0 | ❌ | Same source and same caveat as MOT17. |
| DanceTrack | annotations CC BY 4.0; **videos non-commercial** | ❌ (videos) | The annotations are permissive, but they are useless without the videos. |
| SOMPT22 | conflicting statements: CC BY 4.0 vs CC BY-NC-SA 4.0 | treat as ❌ | When sources conflict we take the stricter one. |
| CrowdHuman (inside every YOLOX weight we use) | **unverified** | treat as ❌ | Its terms are commonly described as non-commercial research use; we did not find the text. |
| MSMT17 (inside the OSNet weights) | research-only (dataset agreement) | ❌ | |
| CityPersons, ETHZ (inside `bytetrack_x_mot17`) | research / non-commercial terms | ❌ | |
| Pexels demo clips | Pexels licence (free use, including commercial; no resale of unmodified clips) | ✅ for demos | Only used for demos, never for metrics. |
| Our own gate clip | owned by us; consent of people filmed is our responsibility | – | |

## Model weights

Weights trained on non-commercial data are a **legal grey area**. They are used here for
benchmarking only.

| Weights | Code licence | Training data | Our use |
|---|---|---|---|
| ByteTrack YOLOX-X (`ablation`, `x_mot17`, `x_mot20`) | MIT (ByteTrack) | CrowdHuman + MOT (+ CityPersons/ETHZ) | benchmark only |
| SparseTrack `yolox_x_mot20_ablation` | MIT (SparseTrack) | CrowdHuman + MOT20 train first half (per README) | benchmark only |
| DanceTrack YOLOX-X | MIT (DanceTrack repo) | DanceTrack train | benchmark only |
| OSNet-AIN x1.0 (torchreid) | MIT (torchreid) | MSMT17 | benchmark only |
| BoT-SORT FastReID SBS-S50 | – | MOT17/MOT20 patches (may include val halves, unverified) | **not used** |

## Code

| Project | Licence | How we use it |
|---|---|---|
| YOLOX (via ByteTrack's copy) | Apache-2.0 | **default detector**; cloned on the DGX at a pinned commit, not vendored |
| ByteTrack | MIT | algorithm re-implemented in `tracker/` (no files copied); model code imported on the DGX |
| OC-SORT, Deep OC-SORT, SparseTrack, Hybrid-SORT, DanceTrack | MIT | ideas and reference only |
| StrongSORT | not checked (no local clone) | ideas from the paper only; no code |
| BoT-SORT | MIT for the tracker, **but the repo vendors a GPL-3.0 `yolov7/` folder** | port only its tracker code (camera-motion compensation, IoU+ReID fusion), never anything from `yolov7/` |
| TrackEval | MIT | evaluation; vendored with its licence file in phase 1 (`third_party/trackeval/`) |
| torchreid (deep-person-reid) | MIT | re-ID model on the DGX |
| lap, scipy, numpy, PyYAML | BSD/MIT | runtime deps |
| **boxmot** | **AGPL-3.0** | **deliberately avoided**: using it would force the whole service open |
| **Ultralytics (YOLOv8/11)** | **AGPL-3.0** | **deliberately avoided** for the same reason; YOLOX (Apache-2.0) is the default |

## Implications for a Zed product

1. The tracker code in this repo (pure NumPy/SciPy) has no copyleft dependencies.
2. A product detector must be retrained on data Zed has rights to, such as its own gate footage
   with consent. Commercially licensed datasets are a possible substitute, but none has been
   verified here.
3. The re-ID model needs the same treatment, or must be replaced by a model trained on licensed
   data.
4. Any code from BoT-SORT has to come only from its tracker files.

## Attribution file

The top-level `NOTICE` lists every ported and vendored third-party component (ByteTrack, OC-SORT,
BoT-SORT, Deep OC-SORT, Hybrid-SORT, TrackEval, the pre-computed CMC affines) with its MIT
copyright line, and repeats the non-commercial-research restriction on datasets and weights.
The project's own licence is still to be chosen (Apache-2.0 proposed); until then there is no
top-level `LICENSE` file.

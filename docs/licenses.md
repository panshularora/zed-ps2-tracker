# Licences: data, weights, code

**Bottom line.** Every dataset we evaluate on is **non-commercial**. Everything in this repo is a
**research benchmark**: nothing trained on, or tuned against, these datasets should ship in a Zed
product without a separate licensing review. The project's own code is licensed
under **Apache-2.0** (top-level `LICENSE`), and the ported and vendored MIT code keeps its notices
(`NOTICE`). That licence does not cover the datasets or weights.

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
| Pexels demo clips | Pexels licence (free use, including commercial; no resale of unmodified clips) | ✅ for demos, with the limits below | Only used for demos, never for metrics. See "Demo clips" below. |
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
| YOLOX-Nano/Tiny/S ONNX (Megvii release `0.1.1rc0`) | Apache-2.0 (YOLOX) | COCO 2017 (annotations CC BY 4.0; images under their Flickr terms) | CPU demo detector of `serve/` only (demo clips, uploads); never in a results table |
| YuNet `face_detection_yunet_2023mar` (OpenCV Zoo) | MIT | WIDER FACE (research-use terms) | only to **blur** faces in demo clips (`serve/blur.py`) |

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

## Demo clips (Pexels)

Five short clips from [Pexels](https://www.pexels.com) are used for the web demo and the demo
video. Videos from Pexels, used under the [Pexels License](https://www.pexels.com/license/):
free to use, including commercially, attribution not required (we credit Pexels anyway). The
licence does **not** allow showing identifiable people in a bad light or in an offensive way,
implying that they endorse a product, or selling or redistributing unmodified copies.

| Clip | Pexels page |
|---|---|
| Busy subway turnstile in urban station | https://www.pexels.com/video/busy-subway-turnstile-in-urban-station-32511497/ |
| Bustling Mexico City metro entrance | https://www.pexels.com/video/bustling-mexico-city-metro-entrance-30138564/ |
| Modern Milan metro station turnstiles | https://www.pexels.com/video/modern-milan-metro-station-turnstiles-31540854/ |
| Fans arriving at stadium in Rabat | https://www.pexels.com/video/fans-arriving-at-stadium-in-rabat-34452568/ |
| Man entering a metro station | https://www.pexels.com/video/man-entering-a-metro-station-7251031/ |

How the repo complies:
* **Not redistributed.** The clips are not in git. `make clips` downloads them from Pexels into
  `data/demo/raw/` (gitignored); the manifest is `configs/demo/clips.yaml`.
* **Faces blurred.** `make blur` (`serve/blur.py`) writes face- and head-blurred copies to
  `data/demo/blurred/` (YuNet faces + the head region of every detected person, held ±2 frames,
  keyframe every 10 frames). The service lists and serves **only** these copies. Blurring is best
  effort: check a recording frame by frame before publishing it.
* **No accusations.** The people in these clips did not agree to be shown as tailgaters or fare
  evaders. Every Pexels clip is `consent: false`, `event_styling: neutral`: the UI shows track
  IDs, crossings and in/out counts only. Tailgate and `ambiguous_hold` alerts and the error colour
  are shown only for clips marked `consent: true` (our own gate recording; none yet).
  `serve/clips.py` refuses `event_styling: alert` without consent.
* **No endorsement.** Zed branding is kept outside the video frame (page chrome only), never
  composited onto the footage.

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
The project's own code is Apache-2.0 (top-level `LICENSE`, Copyright 2026 Panshul Arora). See
ADR-011.

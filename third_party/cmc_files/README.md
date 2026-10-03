# Pre-computed camera-motion (GMC/CMC) affines (vendored, unmodified)

One text file per sequence, one line per frame: `idx<TAB>a00<TAB>a01<TAB>a02<TAB>a10<TAB>a11<TAB>a12`,
the 2×3 affine that maps frame `idx-1` coordinates to frame `idx` (line 0 is the identity).
They were computed by the BoT-SORT authors with OpenCV sparse optical flow on the original frames,
which lets camera-motion compensation run on CPU without any video.

| Folder | Lines cover | Source (MIT) |
|---|---|---|
| `MOTChallenge/` | full MOT17 / MOT20 train+test sequences | BoT-SORT `tracker/GMC_files/MOTChallenge` @ `2519854` (identical copy in Deep OC-SORT) |
| `MOT17_ablation/` | MOT17 train **val half** (frame 1 of the half = line 0) | BoT-SORT `tracker/GMC_files/MOT17_ablation` @ `2519854` |
| `MOT20_ablation/` | MOT20 train val half | Deep OC-SORT `cache/cmc_files/MOT20_ablation` @ `6bb51d0` |
| `DanceTrack/` | DanceTrack sequences | Deep OC-SORT `cache/cmc_files/DanceTrack` @ `6bb51d0` |

Licences: `LICENSE.BoT-SORT` (Copyright (c) 2022 Nir Aharon) and `LICENSE.Deep-OC-SORT`
(Copyright (c) 2023 Gerard Maggiolino), both MIT, copied verbatim.

`tracker/cmc.py::FileCMC` reads a file **by frame index** (not sequentially, so a skipped or
repeated frame can't silently desynchronise it), and checks the line count against the split
length before using it.

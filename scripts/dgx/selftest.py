"""GPU self-test: every checkpoint loads into the right architecture and runs one frame.

Run by setup.sh. Uses a synthetic frame, so it checks loading / shapes / CUDA / fp16 only, not
accuracy (accuracy was spot-checked on CPU with bytetrack_s_mot17 on a real crowd photo).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _free_cuda() -> None:
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # best effort only
        pass


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--allow-oom", action="store_true",
                   help="a CUDA out-of-memory on one checkpoint is a warning, not a failure "
                   "(laptop profile: run.sh all falls back to YOLOX-S on a real OOM)")  # fmt: skip
    a = p.parse_args()
    from models.osnet_embedder import OsnetConfig, OsnetEmbedder
    from models.yolox_detector import YoloxConfig, YoloxDetector

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    ok = True
    for name, size in [
        ("bytetrack_ablation.pth.tar", (800, 1440)),
        ("bytetrack_x_mot17.pth.tar", (800, 1440)),
        ("bytetrack_x_mot20.pth.tar", (896, 1600)),
        ("dancetrack_bytetrack_x.pth.tar", (800, 1440)),
        ("sparsetrack_mot20_ablation.pth.tar", (896, 1600)),
    ]:
        try:
            det = YoloxDetector(YoloxConfig(weights=str(a.weights / name), test_size=size))
            det([img])  # warm-up
            t = time.perf_counter()
            ((_, score),) = det([img])
            ms = 1000 * (time.perf_counter() - t)
            print(f"[ok] {name}: {len(score)} dets on noise, {ms:.1f} ms")
            del det
        except Exception as e:  # report all failures, then exit non-zero
            if a.allow_oom and "out of memory" in repr(e).lower():
                print(f"[OOM ] {name}: {e!r} (allowed: --allow-oom)")
                _free_cuda()
                continue
            ok = False
            print(f"[FAIL] {name}: {e!r}")
    try:
        emb = OsnetEmbedder(OsnetConfig(weights=str(a.weights / "osnet_ain_x1_0_msmt17.pth")))
        boxes = np.array([[100, 100, 50, 150], [400, 300, 60, 160]], np.float32)
        e = emb(img, boxes)
        if e.shape != (2, 512) or e.dtype != np.float16:
            raise RuntimeError(f"bad embedding {e.shape} {e.dtype}")
        print(f"[ok] osnet_ain_x1_0: {e.shape}, |e|={np.linalg.norm(e.astype(np.float32), axis=1)}")
    except Exception as e:
        ok = False
        print(f"[FAIL] osnet: {e!r}")
    print("SELFTEST OK" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

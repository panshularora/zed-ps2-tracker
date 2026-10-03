"""Run upstream BoT-SORT (tracker part) on dumped detections; see parity_upstream.py.

Separate process because upstream's top-level package is also called ``tracker``.
Usage: python parity_botsort_upstream.py <upstream_root> <in.npz> <out.npz>
"""

import sys
import types

import numpy as np


def stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def bbox_overlaps(a, b):
    """cython_bbox.bbox_overlaps (Faster R-CNN convention: areas use +1)."""
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    out = np.zeros((len(a), len(b)))
    for k in range(len(b)):
        box_area = (b[k, 2] - b[k, 0] + 1) * (b[k, 3] - b[k, 1] + 1)
        for n in range(len(a)):
            iw = min(a[n, 2], b[k, 2]) - max(a[n, 0], b[k, 0]) + 1
            if iw > 0:
                ih = min(a[n, 3], b[k, 3]) - max(a[n, 1], b[k, 1]) + 1
                if ih > 0:
                    ua = (a[n, 2] - a[n, 0] + 1) * (a[n, 3] - a[n, 1] + 1) + box_area - iw * ih
                    out[n, k] = iw * ih / ua
    return out


def main():
    root, inp, outp = sys.argv[1:4]
    stub("cv2")
    stub("matplotlib")
    stub("matplotlib.pyplot")
    stub("cython_bbox", bbox_overlaps=bbox_overlaps)
    stub("fast_reid")
    stub("fast_reid.fast_reid_interfece", FastReIDInterface=object)
    import os
    os.chdir(root)
    sys.path[:] = [root] + [p for p in sys.path if "zed-ps2-tracker" not in p]
    from tracker.bot_sort import BoTSORT

    d = np.load(inp, allow_pickle=True)
    seqs = list(d["seqs"])
    res = {}
    for s in seqs:
        args = types.SimpleNamespace(track_high_thresh=0.6, track_low_thresh=0.1,
                                     new_track_thresh=0.7, track_buffer=30, match_thresh=0.8,
                                     proximity_thresh=0.5, appearance_thresh=0.25,
                                     with_reid=False, cmc_method="file", name=s, ablation=True,
                                     mot20=False)
        trk = BoTSORT(args, frame_rate=30)
        frames = d[f"{s}_frames"]
        dets = d[f"{s}_dets"]
        rows = []
        for f in range(1, int(d[f"{s}_len"]) + 1):
            x = dets[frames == f]
            out = trk.update(x.copy(), None)
            for t in out:
                rows.append([f, t.track_id, *t.tlwh])
        res[s] = np.asarray(rows, np.float64).reshape(-1, 6)
    np.savez(outp, **res)


if __name__ == "__main__":
    main()

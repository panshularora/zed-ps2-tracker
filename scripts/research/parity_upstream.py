"""Frame-by-frame parity of our baseline ports against the upstream reference code.

Not part of the package or CI: needs local clones of the upstream repos and a scratch venv with
``filterpy`` (the upstream Kalman code imports it). Run once per implementation, e.g.::

    /workspace/parity-venv/bin/python scripts/research/parity_upstream.py \
        --impl ocsort --upstream /workspace/zed-ps2/src/noahcao_OC_SORT

It feeds the *same* MOT17 val-half public detections to the upstream tracker and to ours
(output filter disabled on both sides), and compares every frame's reported (id, box) set.
Upstream ``linear_assignment`` is patched to scipy (our backend) so that tie-breaking is
identical; lapjv and scipy return the same optimum but can differ on exact ties.
Heavy upstream imports (cv2, torch, re-ID) are stubbed: they are only used for images/re-ID.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.splits import crop_cache, find_sequences  # noqa: E402
from tracker.config import load_config  # noqa: E402
from tracker.io import load_mot_det_txt  # noqa: E402
from tracker.runner import build_tracker  # noqa: E402


def lsa(cost, *a, **k):
    x, y = linear_sum_assignment(cost)
    return np.array(list(zip(x, y)))


def stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def load_upstream(impl: str, root: Path):
    sys.path.insert(0, str(root))
    if impl == "ocsort":
        from trackers.ocsort_tracker import association, ocsort

        association.linear_assignment = lsa
        ocsort.linear_assignment = lsa
        return ocsort
    if impl == "hybrid":
        stub("lap")
        from trackers.hybrid_sort_tracker import association, hybrid_sort

        association.linear_assignment = lsa
        hybrid_sort.linear_assignment = lsa
        return hybrid_sort
    if impl == "deep":
        stub("cv2", cvtColor=lambda img, code: img, COLOR_BGR2GRAY=0)
        stub("torch", Tensor=type("Tensor", (), {}))
        stub("torchvision")
        stub("torchreid")
        class EmbeddingComputer:  # noqa: N801
            def __init__(self, *a, **k):
                pass
        stub("trackers.integrated_ocsort_embedding.embedding", EmbeddingComputer=EmbeddingComputer)
        os.chdir(root)  # CMCComputer reads ./cache/cmc_files
        from trackers.integrated_ocsort_embedding import association, ocsort

        association.linear_assignment = lsa
        ocsort.linear_assignment = lsa
        return ocsort
    raise ValueError(impl)


def botsort_parity(a, cfg) -> int:
    import subprocess
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    blob, ours_rows = {}, {}
    seqs = []
    for seq in find_sequences(a.data, "MOT17"):
        cache = crop_cache(load_mot_det_txt(seq, "MOT17", "train"), "val_half")
        seqs.append(seq.name)
        frames, dets = [], []
        ours, _ = build_tracker(cfg, cache.meta.frame_rate, cache.meta)
        rows = []
        for f, d in cache.frames():
            frames += [f] * len(d)
            dets.append(np.concatenate([d.xyxy, d.scores[:, None]], 1))
            for t in ours.update(d).tracks:
                rows.append([f, t.track_id, *t.tlwh])
        blob[f"{seq.name}_frames"] = np.asarray(frames)
        blob[f"{seq.name}_dets"] = np.concatenate(dets) if dets else np.zeros((0, 5))
        blob[f"{seq.name}_len"] = cache.meta.seq_length
        ours_rows[seq.name] = np.asarray(rows, np.float64).reshape(-1, 6)
    blob["seqs"] = np.asarray(seqs)
    np.savez(tmp / "in.npz", **blob)
    subprocess.run([sys.executable, str(Path(__file__).with_name("parity_botsort_upstream.py")),
                    str(a.upstream.resolve()), str(tmp / "in.npz"), str(tmp / "out.npz")],
                   check=True)
    up = np.load(tmp / "out.npz")
    report = {"impl": "botsort", "sequences": {}}
    tf = tb = 0
    for s in seqs:
        u, o = up[s], ours_rows[s]
        n = int(blob[f"{s}_len"])
        bad, first = 0, None
        for f in range(1, n + 1):
            uf = sorted(map(tuple, np.round(u[u[:, 0] == f][:, 1:], 4)))
            of = sorted(map(tuple, np.round(o[o[:, 0] == f][:, 1:], 4)))
            same = len(uf) == len(of) and all(
                x[0] == y[0] and np.allclose(x[1:], y[1:], atol=1e-3) for x, y in zip(uf, of))
            if not same:
                bad += 1
                if first is None:
                    su, so = set(uf), set(of)
                    first = {"frame": f, "only_upstream": sorted(su - so)[:6],
                             "only_ours": sorted(so - su)[:6]}
        report["sequences"][s] = {"frames": n, "mismatched_frames": bad, "first_mismatch": first}
        tf += n
        tb += bad
        print(f"botsort {s}: {n} frames, {bad} mismatched", flush=True)
    report["frames"], report["mismatched_frames"] = tf, tb
    print(f"botsort: {tb}/{tf} frames differ")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1, default=str))
    return 0


class Shape:
    def __init__(self, shape):
        self.shape = shape


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["ocsort", "hybrid", "deep", "botsort"], required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "MOT17")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    cfg_name = {"ocsort": "ocsort", "hybrid": "hybrid_sort", "deep": "deep_ocsort",
                "botsort": "botsort"}[a.impl]
    cfg = load_config(ROOT / "configs" / "baselines" / f"{cfg_name}.yaml",
                      ["output.min_box_area=-1.0", "output.max_aspect_ratio=0.0"])
    if a.impl == "botsort":
        return botsort_parity(a, cfg)
    mod = load_upstream(a.impl, a.upstream.resolve())
    report = {"impl": a.impl, "sequences": {}}
    total_frames = total_bad = 0
    for seq in find_sequences(a.data, "MOT17"):
        cache = crop_cache(load_mot_det_txt(seq, "MOT17", "train"), "val_half")
        h, w = cache.meta.im_height, cache.meta.im_width
        ours, _ = build_tracker(cfg, cache.meta.frame_rate, cache.meta)
        if a.impl == "ocsort":
            up = mod.OCSort(det_thresh=0.6, iou_threshold=0.3, delta_t=3, asso_func="iou",
                            inertia=0.2, use_byte=False)
        elif a.impl == "hybrid":
            args = types.SimpleNamespace(track_thresh=0.6, TCM_first_step=True, TCM_byte_step=True,
                                         TCM_first_step_weight=1.0, TCM_byte_step_weight=1.0)
            up = mod.Hybrid_Sort(args, det_thresh=0.6, iou_threshold=0.25,
                                 asso_func="Height_Modulated_IoU", delta_t=3, inertia=0.05,
                                 use_byte=True)
        else:
            up = mod.OCSort(det_thresh=0.6, iou_threshold=0.3, delta_t=3, asso_func="iou",
                            inertia=0.2, w_association_emb=0.75, alpha_fixed_emb=0.95,
                            aw_param=0.5, embedding_off=True, cmc_off=False, aw_off=True,
                            new_kf_off=True, grid_off=True, args=types.SimpleNamespace(
                                dataset="mot17", test_dataset=False))
            img = np.zeros((h, w), np.uint8)
            ten = Shape((1, 3, h, w))
        bad = 0
        first_bad = None
        n = 0
        for f, dets in cache.frames():
            n += 1
            inp = np.concatenate([dets.xyxy, dets.scores[:, None]], 1).astype(np.float64)
            if a.impl == "deep":
                res_up = up.update(inp.copy(), ten, img, f"{seq.name}:{f}")
            else:
                res_up = up.update(inp.copy(), (h, w), (h, w))
            res_ours = ours.update(dets)
            u = sorted((int(r[4]), tuple(np.round(r[:4], 6))) for r in res_up)
            o = sorted((t.track_id, tuple(np.round([t.tlwh[0], t.tlwh[1], t.tlwh[0] + t.tlwh[2],
                                                     t.tlwh[1] + t.tlwh[3]], 6)))
                       for t in res_ours.tracks)
            same = len(u) == len(o) and all(
                iu == io and np.allclose(bu, bo, atol=1e-4) for (iu, bu), (io, bo) in zip(u, o))
            if not same:
                bad += 1
                if first_bad is None:
                    first_bad = {"frame": f, "upstream": u[:8], "ours": o[:8]}
        report["sequences"][seq.name] = {"frames": n, "mismatched_frames": bad,
                                         "first_mismatch": first_bad}
        total_frames += n
        total_bad += bad
        print(f"{a.impl} {seq.name}: {n} frames, {bad} mismatched", flush=True)
    report["frames"] = total_frames
    report["mismatched_frames"] = total_bad
    print(f"{a.impl}: {total_bad}/{total_frames} frames differ")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Scoring one stress run: TrackEval (HOTA/AssA, IDF1, CLEAR IDSW/Frag), the strict per-person
tally, gate-count error, and the first frame where two trackers disagree about someone's ID.

Evaluation set: a person's box counts as ground truth in a frame when its true visible fraction is
at least ``GT_MIN_VIS`` (the detector's own floor). Fully hidden frames are not scored, so no
tracker is rewarded or punished for not drawing a box on someone nobody can see; what matters is
the number they get when they come back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from eval.stress_sim import GATE_Y, VIS_DROP, Frame
from eval.trackeval_runner import import_trackeval

GT_MIN_VIS = VIS_DROP
MATCH_IOU = 0.5
KEPT_MIN_COVER = 0.8
MIN_PERSON_FRAMES = 15
GATE_HYST = 8.0

TrackRows = list[list[tuple[int, tuple[float, float, float, float]]]]
"""Per frame: ``(track_id, tlwh)`` of every reported box."""


def iou_tlwh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(
        np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0:1], b[None, :, 0]), 0, None
    )
    ih = np.clip(
        np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1:2], b[None, :, 1]), 0, None
    )
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.asarray(np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0))


def _gt(fr: Frame) -> tuple[list[int], np.ndarray]:
    rows = [t for t in fr.truth if t[5] >= GT_MIN_VIS]
    return [r[0] for r in rows], np.asarray([r[1:5] for r in rows], np.float64).reshape(-1, 4)


def match_frames(frames: list[Frame], tracks: TrackRows) -> list[dict[int, int]]:
    """Per frame ``pid -> track id`` (Hungarian on IoU, kept when IoU >= MATCH_IOU)."""
    out: list[dict[int, int]] = []
    for fr, tr in zip(frames, tracks, strict=True):
        pids, g = _gt(fr)
        k = np.asarray([b for _, b in tr], np.float64).reshape(-1, 4)
        m: dict[int, int] = {}
        if len(pids) and len(k):
            iou = iou_tlwh(g, k)
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c, strict=True):
                if iou[i, j] >= MATCH_IOU:
                    m[pids[i]] = tr[j][0]
        out.append(m)
    return out


def trackeval_metrics(frames: list[Frame], tracks: TrackRows) -> dict[str, float]:
    te = import_trackeval()
    gt_map: dict[int, int] = {}
    tk_map: dict[int, int] = {}
    gt_ids, tk_ids, sims = [], [], []
    n_gt = n_tk = 0
    for fr, tr in zip(frames, tracks, strict=True):
        pids, g = _gt(fr)
        k = np.asarray([b for _, b in tr], np.float64).reshape(-1, 4)
        gt_ids.append(np.asarray([gt_map.setdefault(p, len(gt_map)) for p in pids], int))
        tk_ids.append(np.asarray([tk_map.setdefault(t, len(tk_map)) for t, _ in tr], int))
        sims.append(iou_tlwh(g, k))
        n_gt += len(pids)
        n_tk += len(tr)
    data = {
        "num_timesteps": len(frames),
        "num_gt_ids": len(gt_map),
        "num_tracker_ids": len(tk_map),
        "num_gt_dets": n_gt,
        "num_tracker_dets": n_tk,
        "gt_ids": gt_ids,
        "tracker_ids": tk_ids,
        "similarity_scores": sims,
    }
    hota = te.metrics.HOTA().eval_sequence(data)
    clear = te.metrics.CLEAR({"PRINT_CONFIG": False}).eval_sequence(data)
    ident = te.metrics.Identity({"PRINT_CONFIG": False}).eval_sequence(data)
    return {
        "HOTA": float(np.mean(hota["HOTA"])),
        "AssA": float(np.mean(hota["AssA"])),
        "DetA": float(np.mean(hota["DetA"])),
        "IDF1": float(ident["IDF1"]),
        "IDSW": int(clear["IDSW"]),
        "Frag": int(clear["Frag"]),
        "MOTA": float(clear["MOTA"]),
    }


def _crossings(ys: list[tuple[int, float]]) -> int:
    """Side changes of a foot-y sequence over the gate line, with hysteresis."""
    side, n = 0, 0
    for _, y in ys:
        s = 1 if y > GATE_Y + GATE_HYST else (-1 if y < GATE_Y - GATE_HYST else 0)
        if s == 0:
            continue
        if side and s != side:
            n += 1
        side = s
    return n


def gate_counts(frames: list[Frame], tracks: TrackRows) -> tuple[int, int]:
    """``(true crossings, tracker crossings)``; truth from every frame (hidden included)."""
    tru: dict[int, list[tuple[int, float]]] = {}
    trk: dict[int, list[tuple[int, float]]] = {}
    for fr, tr in zip(frames, tracks, strict=True):
        for p, _x, y, _w, h, _v in fr.truth:
            tru.setdefault(p, []).append((fr.f, y + h))
        for t, b in tr:
            trk.setdefault(t, []).append((fr.f, b[1] + b[3]))
    return sum(map(_crossings, tru.values())), sum(map(_crossings, trk.values()))


@dataclass
class PersonTally:
    people: int
    kept: int
    changed: int


def tally(frames: list[Frame], matches: list[dict[int, int]]) -> PersonTally:
    """Strict one-person-one-ID: kept = matched in >= 80 % of the person's scored frames, under a
    single ID that never matched anybody else. ``kept <= people`` by construction."""
    elig: dict[int, int] = {}
    for fr in frames:
        for p in _gt(fr)[0]:
            elig[p] = elig.get(p, 0) + 1
    ids_of: dict[int, list[int]] = {}
    owners: dict[int, set[int]] = {}
    for m in matches:
        for p, t in m.items():
            ids_of.setdefault(p, []).append(t)
            owners.setdefault(t, set()).add(p)
    people = [p for p, n in elig.items() if n >= MIN_PERSON_FRAMES]
    kept = changed = 0
    for p in people:
        seq = ids_of.get(p, [])
        uniq = set(seq)
        if len(uniq) > 1:
            changed += 1
        if (
            len(uniq) == 1
            and len(seq) >= KEPT_MIN_COVER * elig[p]
            and owners[next(iter(uniq))] == {p}
        ):
            kept += 1
    assert kept <= len(people)
    return PersonTally(len(people), kept, changed)


def id_change_events(matches: list[dict[int, int]]) -> list[tuple[int, int, int, int]]:
    """``(frame, pid, from_id, to_id)`` every time a person's matched ID differs from the last."""
    last: dict[int, int] = {}
    out = []
    for f, m in enumerate(matches, start=1):
        for p, t in sorted(m.items()):
            if p in last and last[p] != t:
                out.append((f, p, last[p], t))
            last[p] = t
    return out


def first_disagreement(
    ev_a: list[tuple[int, int, int, int]], ev_b: list[tuple[int, int, int, int]]
) -> dict[str, Any] | None:
    """First frame where the two trackers differ on whether someone's ID changed (a person who
    changed in one and not, or not yet, in the other)."""
    by_a: dict[int, list[tuple[int, int]]] = {}
    by_b: dict[int, list[tuple[int, int]]] = {}
    for f, p, *_ in ev_a:
        by_a.setdefault(p, []).append((f, len(by_a.get(p, []))))
    for f, p, *_ in ev_b:
        by_b.setdefault(p, []).append((f, len(by_b.get(p, []))))
    best: dict[str, Any] | None = None
    for p in set(by_a) | set(by_b):
        fa = [f for f, _ in by_a.get(p, [])]
        fb = [f for f, _ in by_b.get(p, [])]
        # k-th change of p: a disagreement if only one tracker has it, at that tracker's frame;
        # if both have it at different frames, the earlier one is where they first differ
        for k in range(max(len(fa), len(fb))):
            x = fa[k] if k < len(fa) else None
            y = fb[k] if k < len(fb) else None
            if x == y:
                continue
            f = min(v for v in (x, y) if v is not None)
            who = "a" if x == f else "b"
            if best is None or f < best["frame"]:
                best = {"frame": f, "pid": p, "changed": who}
            break
    return best

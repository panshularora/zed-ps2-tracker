"""Stress bins: where does a tracker fail? Errors split by gt visibility and by crowd density.

    python -m eval.stress --gt OUT/gt --trackers OUT/trackers --tracker-name NAME --benchmark MOT17

TrackEval only reports whole-sequence numbers. To split errors by condition, this module runs
TrackEval's own loading and preprocessing (same distractor removal, same zero-marked and class
filtering, same IoU) and then repeats its two matchings per gt box:

* **CLEAR** (MOTA, IDSW): per-frame Hungarian with the previous-frame continuation bonus, IoU ≥ 0.5.
  Each gt box is a TP or an FN, and each matched box may be an ID switch.
* **Identity** (IDF1): one global gt-id ↔ tracker-id assignment. A gt box is an IDTP when its
  assigned tracker id is present in that frame with IoU ≥ 0.5.

Summed over bins, TP/FN/FP/IDSW/IDTP/IDFN/IDFP equal TrackEval's totals; ``check_against``
asserts this.

**Visibility bins** (gt column 9: 0–.25, .25–.5, .5–.75, .75–1; vis 0 is also shown on its own).
False positives have no gt box, so per-bin MOTA/IDF1 are undefined. We report recall (TP/GT),
ID recall (IDTP/GT), IDSW and IDSW per 1,000 gt boxes.

**Density bins** (gt people per frame, edges at the quartiles of the eval set's per-frame counts).
Whole frames fall into a bin, so MOTA and IDF1 are exact within each bin.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from eval.trackeval_runner import dataset_config, import_trackeval

VIS_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
EPS = float(np.finfo("float").eps)
THRESH = 0.5


def _vis_dataset_cls() -> type:
    te = import_trackeval()

    class VisMotChallenge2DBox(te.datasets.MotChallenge2DBox):  # type: ignore[misc,name-defined]
        """MotChallenge2DBox that also keeps gt column 9 (visibility) in ``gt_extras``."""

        def _load_raw_file(self, tracker: str, seq: str, is_gt: bool) -> dict[str, Any]:
            if not is_gt:
                return super()._load_raw_file(tracker, seq, is_gt)  # type: ignore[no-any-return]
            captured: dict[str, Any] = {}
            orig = self._load_simple_text_file  # type: ignore[has-type]

            def capture(*a: Any, **k: Any) -> Any:
                read, ignore = orig(*a, **k)
                captured["read"] = read
                return read, ignore

            self._load_simple_text_file = capture
            try:
                raw: dict[str, Any] = super()._load_raw_file(tracker, seq, is_gt)
            finally:
                del self._load_simple_text_file
            for t in range(raw["num_timesteps"]):
                rows = captured["read"].get(str(t + 1))
                vis = np.asarray(rows, float)[:, 8] if rows else np.empty(0)
                raw["gt_extras"][t]["visibility"] = np.atleast_1d(vis)
            return raw

    return VisMotChallenge2DBox


@dataclass
class SeqRecords:
    """Per-frame, per-gt-box outcomes for one sequence (gt arrays aligned with TrackEval's)."""

    name: str
    vis: list[np.ndarray] = field(default_factory=list)
    tp: list[np.ndarray] = field(default_factory=list)
    idsw: list[np.ndarray] = field(default_factory=list)
    idtp: list[np.ndarray] = field(default_factory=list)
    n_trk: list[int] = field(default_factory=list)


def _clear_per_box(data: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """TrackEval CLEAR matching, returning per-gt-box TP and IDSW flags for every frame."""
    n_ids = data["num_gt_ids"]
    prev_tracker_id = np.full(n_ids, np.nan)
    prev_timestep_tracker_id = np.full(n_ids, np.nan)
    tps, sws = [], []
    for t, (gt_ids_t, trk_ids_t) in enumerate(
        zip(data["gt_ids"], data["tracker_ids"], strict=True)
    ):
        tp = np.zeros(len(gt_ids_t), bool)
        sw = np.zeros(len(gt_ids_t), bool)
        if len(gt_ids_t) and len(trk_ids_t):
            sim = data["similarity_scores"][t]
            score = (trk_ids_t[None, :] == prev_timestep_tracker_id[gt_ids_t[:, None]]) * 1000 + sim
            score[sim < THRESH - EPS] = 0
            r, c = linear_sum_assignment(-score)
            ok = score[r, c] > 0 + EPS
            r, c = r[ok], c[ok]
            m_gt, m_trk = gt_ids_t[r], trk_ids_t[c]
            prev = prev_tracker_id[m_gt]
            sw[r] = ~np.isnan(prev) & (m_trk != prev)
            tp[r] = True
            prev_tracker_id[m_gt] = m_trk
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[m_gt] = m_trk
        # TrackEval leaves prev_timestep_tracker_id untouched on frames with no gt or no tracks
        tps.append(tp)
        sws.append(sw)
    return tps, sws


def _identity_per_box(data: dict[str, Any]) -> list[np.ndarray]:
    """TrackEval Identity global assignment, returning per-gt-box IDTP flags for every frame."""
    ng, nt = data["num_gt_ids"], data["num_tracker_ids"]
    out = [np.zeros(len(g), bool) for g in data["gt_ids"]]
    if ng == 0 or nt == 0:
        return out
    pot = np.zeros((ng, nt))
    gt_cnt, trk_cnt = np.zeros(ng), np.zeros(nt)
    for t, (g, k) in enumerate(zip(data["gt_ids"], data["tracker_ids"], strict=True)):
        mi, mj = np.nonzero(data["similarity_scores"][t] >= THRESH)
        pot[g[mi], k[mj]] += 1
        gt_cnt[g] += 1
        trk_cnt[k] += 1
    fp = np.zeros((ng + nt, ng + nt))
    fn = np.zeros((ng + nt, ng + nt))
    fp[ng:, :nt] = 1e10
    fn[:ng, nt:] = 1e10
    for i in range(ng):
        fn[i, :nt] = gt_cnt[i]
        fn[i, nt + i] = gt_cnt[i]
    for j in range(nt):
        fp[:ng, j] = trk_cnt[j]
        fp[j + ng, j] = trk_cnt[j]
    fn[:ng, :nt] -= pot
    fp[:ng, :nt] -= pot
    rows, cols = linear_sum_assignment(fn + fp)
    g2t = np.full(ng, -1)
    real = (rows < ng) & (cols < nt)
    g2t[rows[real]] = cols[real]
    for t, (g, k) in enumerate(zip(data["gt_ids"], data["tracker_ids"], strict=True)):
        if len(g) == 0 or len(k) == 0:
            continue
        want = g2t[g]  # assigned tracker id per gt box (-1: none)
        hit = (k[None, :] == want[:, None]) & (data["similarity_scores"][t] >= THRESH)
        out[t] = hit.any(axis=1)
    return out


def sequence_records(ds: Any, tracker: str, seq: str) -> SeqRecords:
    raw = ds.get_raw_seq_data(tracker, seq)
    data = ds.get_preprocessed_seq_data(raw, "pedestrian")
    tps, sws = _clear_per_box(data)
    idtps = _identity_per_box(data)
    rec = SeqRecords(seq)
    for t in range(raw["num_timesteps"]):
        ex = raw["gt_extras"][t]
        keep = (ex["zero_marked"] != 0) & (raw["gt_classes"][t] == 1)
        vis = ex["visibility"][keep]
        if len(vis) != len(data["gt_ids"][t]):  # pragma: no cover - would mean TrackEval changed
            raise RuntimeError(f"{seq} t={t}: visibility not aligned with TrackEval gt")
        rec.vis.append(vis)
        rec.tp.append(tps[t])
        rec.idsw.append(sws[t])
        rec.idtp.append(idtps[t])
        rec.n_trk.append(len(data["tracker_ids"][t]))
    return rec


def _vis_bin_rows(recs: list[SeqRecords]) -> list[dict[str, Any]]:
    vis = np.concatenate([v for r in recs for v in r.vis])
    tp = np.concatenate([v for r in recs for v in r.tp])
    sw = np.concatenate([v for r in recs for v in r.idsw])
    idtp = np.concatenate([v for r in recs for v in r.idtp])
    rows = []
    bins = [(f"{lo:.2f}-{hi:.2f}", (vis >= lo) & ((vis < hi) if hi < 1.0 else (vis <= hi)))
            for lo, hi in itertools.pairwise(VIS_EDGES)]  # fmt: skip
    bins.append(("=0 (subset)", vis == 0))
    for label, m in bins:
        n = int(m.sum())
        rows.append({
            "bin": label, "gt": n, "share": n / max(len(vis), 1),
            "tp": int(tp[m].sum()), "fn": int(n - tp[m].sum()), "idsw": int(sw[m].sum()),
            "idtp": int(idtp[m].sum()),
            "recall": float(tp[m].mean()) if n else float("nan"),
            "id_recall": float(idtp[m].mean()) if n else float("nan"),
            "idsw_per_1k_gt": 1000.0 * float(sw[m].sum()) / n if n else float("nan"),
        })  # fmt: skip
    return rows


def density_edges(counts: np.ndarray) -> list[int]:
    """Quartile edges of per-frame gt counts (frames with ≥ 1 gt), as integer bin starts."""
    pos = counts[counts > 0]
    if pos.size == 0:
        return [0]
    q = np.unique(np.round(np.percentile(pos, [25, 50, 75])).astype(int))
    return [0, *[int(x) for x in q if x > 0]]


def _density_rows(recs: list[SeqRecords], edges: list[int]) -> list[dict[str, Any]]:
    n_gt = np.array([len(v) for r in recs for v in r.vis])
    n_trk = np.array([n for r in recs for n in r.n_trk])
    tp = np.array([int(v.sum()) for r in recs for v in r.tp])
    sw = np.array([int(v.sum()) for r in recs for v in r.idsw])
    idtp = np.array([int(v.sum()) for r in recs for v in r.idtp])
    bounds = [*edges, 10**9]
    rows = []
    for lo, hi in itertools.pairwise(bounds):
        m = (n_gt >= lo) & (n_gt < hi)
        gt, fp_ = int(n_gt[m].sum()), int((n_trk[m] - tp[m]).sum())
        fn_, ids, itp = int(gt - tp[m].sum()), int(sw[m].sum()), int(idtp[m].sum())
        idfp, idfn = int(n_trk[m].sum() - itp), gt - itp
        rows.append({
            "bin": f"{lo}+" if hi == 10**9 else f"{lo}-{hi - 1}", "frames": int(m.sum()),
            "gt": gt, "fp": fp_, "fn": fn_, "idsw": ids, "idtp": itp, "idfp": idfp, "idfn": idfn,
            "mota": 100.0 * (1 - (fp_ + fn_ + ids) / gt) if gt else float("nan"),
            "idf1": 100.0 * 2 * itp / (2 * itp + idfp + idfn) if itp + idfp + idfn else math.nan,
            "recall": 100.0 * (gt - fn_) / gt if gt else float("nan"),
            "idsw_per_1k_gt": 1000.0 * ids / gt if gt else float("nan"),
        })  # fmt: skip
    return rows


def run_stress(gt: Path, trackers: Path, tracker: str, benchmark: str,
               check_against: dict[str, dict[str, float]] | None = None,
               density: list[int] | None = None) -> dict[str, Any]:  # fmt: skip
    cls = _vis_dataset_cls()
    ds = cls(dataset_config(gt, trackers, benchmark, [tracker]))
    recs = [sequence_records(ds, tracker, s) for s in ds.seq_list]
    counts = np.array([len(v) for r in recs for v in r.vis])
    edges = density if density is not None else density_edges(counts)
    vis_rows, dens_rows = _vis_bin_rows(recs), _density_rows(recs, edges)
    totals = {
        "TP": sum(r["tp"] for r in vis_rows[:4]), "FN": sum(r["fn"] for r in vis_rows[:4]),
        "IDSW": sum(r["idsw"] for r in vis_rows[:4]), "IDTP": sum(r["idtp"] for r in vis_rows[:4]),
        "FP": sum(r["fp"] for r in dens_rows), "IDFP": sum(r["idfp"] for r in dens_rows),
        "IDFN": sum(r["idfn"] for r in dens_rows),
    }  # fmt: skip
    if check_against is not None:
        ref = check_against["COMBINED"]
        bad = {k: (v, ref[k]) for k, v in totals.items() if round(ref[k]) != v}
        if bad:
            raise AssertionError(f"stress totals differ from TrackEval: {bad}")
    return {"tracker": tracker, "benchmark": benchmark, "sequences": [r.name for r in recs],
            "visibility": vis_rows, "density_edges": edges, "density": dens_rows,
            "totals": totals, "verified_against_trackeval": check_against is not None}  # fmt: skip


def stress_markdown(st: dict[str, Any]) -> str:
    def f(x: float, d: int = 1) -> str:
        return "–" if x != x else f"{x:.{d}f}"

    out = ["**By gt visibility** (per gt box; FP cannot be binned, so recall/ID-recall, not MOTA)",
           "", "| Visibility | GT boxes | share | Recall | ID recall | IDSW | IDSW / 1k GT |",
           "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in st["visibility"]:
        out.append(f"| {r['bin']} | {r['gt']} | {100 * r['share']:.1f}% | "
                   f"{f(100 * r['recall'])} | {f(100 * r['id_recall'])} | {r['idsw']} | "
                   f"{f(r['idsw_per_1k_gt'], 2)} |")  # fmt: skip
    out += ["", "**By people per frame** (whole frames; exact MOTA / IDF1 per bin)", "",
            "| People / frame | Frames | GT | MOTA | IDF1 | Recall | FP | IDSW | IDSW / 1k GT |",
            "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in st["density"]:
        out.append(
            f"| {r['bin']} | {r['frames']} | {r['gt']} | {f(r['mota'])} | {f(r['idf1'])} "
            f"| {f(r['recall'])} | {r['fp']} | {r['idsw']} | {f(r['idsw_per_1k_gt'], 2)} |"
        )
    if st.get("verified_against_trackeval"):
        out += ["", "_Bin totals verified equal to TrackEval's TP/FN/FP/IDSW/IDTP/IDFP/IDFN._"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    from eval.trackeval_runner import evaluate

    p = argparse.ArgumentParser(prog="python -m eval.stress", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--trackers", type=Path, required=True)
    p.add_argument("--tracker-name", required=True)
    p.add_argument("--benchmark", default="MOT17", choices=["MOT17", "MOT20", "MOT16"])
    p.add_argument("--density-edges", type=int, nargs="+", default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    ref = evaluate(a.gt, a.trackers, a.benchmark, [a.tracker_name], write_outputs=False)
    st = run_stress(a.gt, a.trackers, a.tracker_name, a.benchmark, ref[a.tracker_name],
                    a.density_edges)  # fmt: skip
    print(stress_markdown(st))
    if a.out is not None:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "stress.json").write_text(json.dumps(st, indent=2))
        (a.out / "stress.md").write_text(stress_markdown(st) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

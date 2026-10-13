"""Bounded-delay, ambiguity-aware assignment ("near-online"): :class:`DelayedTracker`.

The wrapped :class:`~tracker.bytetrack.ByteTrackLike` stays strictly online. The wrapper keeps
``K`` frames of detections in a buffer and commits frame ``f`` only after frame ``f + K`` has
arrived. Output for every frame is emitted exactly once, ``K`` frames late, and is never rewritten.

To commit frame ``f``:

1. Run a copy of the committed tracker on ``f``. Read the first-association cost matrix
   (confirmed + lost tracks ↔ high-score detections).
2. **Ambiguity.** An admissible pair (``cost < match_thresh``) is a *near tie* if its cost is
   within ``margin`` of the best cost in its row or in its column. The connected components of the
   near-tie graph with at least three nodes (two tracks competing for one detection, or one track
   between two detections, or larger) are the ambiguity clusters. A cluster with more than
   ``max_cluster`` tracks or detections is left to the greedy solver (counted as ``skipped``).
3. **Hypotheses.** For each cluster, enumerate its partial matchings over admissible pairs. The
   score is the assignment objective (sum of costs + ``match_thresh / 2`` per unmatched track or
   detection, as in the padded Hungarian solver). Keep the ``max_hypotheses`` best, always
   including the one the immediate solver chose.
4. **Look-ahead.** For each hypothesis: copy the tracker, force the hypothesis in frame ``f``'s
   first association, then run the buffered frames ``f+1 … f+K`` unchanged. The score is the sum,
   over frames ``f … f+K`` and over the cluster's tracks, of each track's association cost that
   frame, or ``miss_penalty`` when it is unmatched or gone. An alternative replaces the immediate
   choice only if its score is lower by more than ``min_gain``. Clusters are decided one at a
   time, with the other clusters left to the solver. ``score_scope = all`` sums over every
   track of the first association instead of the cluster's tracks only, so a hypothesis that
   steals a detection from a track outside the cluster pays for it. ``birth_penalty`` adds a
   cost per frame for every track born during the look-ahead (a fragment). ``confirmed_only``
   keeps lost tracks out of clusters; ``same_tracks`` only considers re-assignments (swaps).
5. Commit frame ``f`` with the chosen hypotheses forced and emit its output.

``K = 0`` (or no ambiguity) reproduces the immediate tracker exactly. The algorithmic delay is
``K`` frames, i.e. ``1000·K / fps`` ms; it is reported separately from compute latency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tracker.bytetrack import ByteTrackLike, Stage1Record
from tracker.lifecycle import Track
from tracker.types import Detections, FloatArray, FrameResult, PendingDecision

Hypothesis = tuple[tuple[int, int], ...]
"""Pairs ``(track_id, det_index)`` of one cluster; cluster tracks not listed stay unmatched."""


@dataclass
class Cluster:
    track_ids: tuple[int, ...]
    cols: tuple[int, ...]
    edges: dict[tuple[int, int], float]
    """``(track_id, col) → cost`` for admissible pairs."""
    greedy: Hypothesis


@dataclass
class DelayedStats:
    frames: int = 0
    ambiguous_frames: int = 0
    clusters: int = 0
    skipped_large: int = 0
    changed: int = 0
    """Clusters where the look-ahead chose a different hypothesis than the immediate solver."""
    rollouts: int = 0
    trigger_frames: list[int] = field(default_factory=list)
    changed_frames: list[int] = field(default_factory=list)


def find_clusters(
    rec: Stage1Record, margin: float, max_cluster: int, confirmed_only: bool = False,
    max_visibility: float = 1.0, require_all_occluded: bool = False,
) -> tuple[list[Cluster], int]:  # fmt: skip
    """Ambiguity clusters of one first-association cost matrix. Returns ``(clusters, skipped)``.

    ``confirmed_only`` drops lost tracks from the ambiguity graph (they keep their normal
    association). ``max_visibility < 1`` keeps only clusters in which at least one detection has
    estimated visibility below it (occlusion-triggered look-ahead; needs ``rec.dets_vis``)."""
    c = rec.cost
    n, m = c.shape
    if n == 0 or m == 0:
        return [], 0
    ok = np.isfinite(c) & (c < rec.thresh)
    if confirmed_only:
        ok &= ~np.asarray(rec.lost, bool).reshape(-1, 1)
    big = np.where(ok, c, np.inf)
    row_best = big.min(axis=1, keepdims=True)
    col_best = big.min(axis=0, keepdims=True)
    tie = ok & ((big <= row_best + margin) | (big <= col_best + margin))
    # union-find over rows 0..n-1 and cols n..n+m-1, via near-tie edges
    parent = list(range(n + m))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for r, col in zip(*np.nonzero(tie), strict=True):
        ra, rb = find(int(r)), find(n + int(col))
        if ra != rb:
            parent[ra] = rb
    comps: dict[int, list[int]] = {}
    for v in range(n + m):
        comps.setdefault(find(v), []).append(v)
    matched = {int(r): int(col) for r, col in rec.matches}
    out, skipped = [], 0
    for nodes in comps.values():
        rows = [v for v in nodes if v < n]
        cols = [v - n for v in nodes if v >= n]
        if len(nodes) < 3 or not rows or not cols:
            continue
        if len(rows) > max_cluster or len(cols) > max_cluster:
            skipped += 1
            continue
        vis = rec.dets_vis
        if max_visibility < 1.0 and vis is not None:
            vv = vis[cols]
            worst = float(np.max(vv)) if require_all_occluded else float(np.min(vv))
            if worst >= max_visibility:
                continue
        edges = {
            (rec.track_ids[r], col): float(c[r, col]) for r in rows for col in cols if ok[r, col]
        }
        greedy = tuple(sorted((rec.track_ids[r], matched[r]) for r in rows
                              if r in matched and matched[r] in cols))  # fmt: skip
        out.append(Cluster(tuple(rec.track_ids[r] for r in rows), tuple(cols), edges, greedy))
    return out, skipped


def enumerate_hypotheses(
    cl: Cluster, thresh: float, max_hyp: int, same_tracks: bool = False
) -> list[Hypothesis]:
    """Best partial matchings of a cluster by the padded-assignment objective (greedy included).

    ``same_tracks`` keeps only matchings that match the same set of tracks as the immediate
    choice (pure re-assignments such as identity swaps)."""
    half = thresh / 2.0
    hyps: list[tuple[float, Hypothesis]] = []

    def rec(i: int, used: frozenset[int], pairs: list[tuple[int, int]], cost: float) -> None:
        if i == len(cl.track_ids):
            unmatched_cols = len(cl.cols) - len(used)
            hyps.append((cost + half * unmatched_cols, tuple(sorted(pairs))))
            return
        tid = cl.track_ids[i]
        rec(i + 1, used, pairs, cost + half)  # this track unmatched
        for col in cl.cols:
            if col not in used and (tid, col) in cl.edges:
                rec(i + 1, used | {col}, [*pairs, (tid, col)], cost + cl.edges[(tid, col)])

    rec(0, frozenset(), [], 0.0)
    if same_tracks:
        g_tracks = {t for t, _ in cl.greedy}
        hyps = [h for h in hyps if {t for t, _ in h[1]} == g_tracks]
    hyps.sort(key=lambda h: (h[0], h[1]))
    best = [h for _, h in hyps[:max_hyp]]
    if not best:
        return [cl.greedy]
    if cl.greedy not in best:
        best[-1] = cl.greedy
    if cl.greedy in best:  # the immediate choice is evaluated first (wins ties)
        best.remove(cl.greedy)
    return [cl.greedy, *best]


def force_hook(forced: list[tuple[Cluster, Hypothesis]]) -> Any:
    """A stage-1 hook that makes each cluster take exactly the given hypothesis."""

    def hook(tracks: list[Track], cost: FloatArray) -> FloatArray:
        row_of = {t.track_id: i for i, t in enumerate(tracks)}
        orig = cost.copy()
        for cl, hyp in forced:
            rows = [row_of[t] for t in cl.track_ids if t in row_of]
            cost[rows, :] = np.inf  # cluster tracks: only their forced pair stays
            for tid, col in hyp:
                if tid in row_of:
                    cost[:, col] = np.inf  # nobody else takes that detection
            for tid, col in hyp:
                if tid in row_of:
                    cost[row_of[tid], col] = orig[row_of[tid], col]
        return cost

    return hook


class DelayedTracker:
    """Wraps :class:`ByteTrackLike` with a ``K``-frame look-ahead on ambiguous associations."""

    def __init__(
        self,
        inner: ByteTrackLike,
        *,
        max_delay_frames: int,
        margin: float,
        max_cluster: int = 4,
        max_hypotheses: int = 6,
        miss_penalty: float = 1.0,
        min_gain: float = 0.0,
        confirmed_only: bool = False,
        same_tracks: bool = False,
        score_scope: str = "cluster",
        birth_penalty: float = 0.0,
        trigger_max_visibility: float = 1.0,
        trigger_require_all: bool = False,
    ) -> None:
        if max_delay_frames < 0:
            raise ValueError("max_delay_frames must be >= 0")
        self.state = inner
        self.state.record = True
        self.K = max_delay_frames
        self.margin = margin
        self.max_cluster = max_cluster
        self.max_hyp = max_hypotheses
        self.miss_penalty = miss_penalty
        self.min_gain = min_gain
        self.confirmed_only = confirmed_only
        self.same_tracks = same_tracks
        if score_scope not in ("cluster", "all"):
            raise ValueError(f"unknown score_scope {score_scope!r}")
        self.score_scope = score_scope
        self.birth_penalty = birth_penalty
        self.trigger_max_visibility = trigger_max_visibility
        self.trigger_require_all = trigger_require_all
        self.buffer: deque[Detections] = deque()
        self.stats = DelayedStats()
        self._received = 0

    @property
    def lag(self) -> int:
        return self.K

    # -------------------------------------------------------------------------------------- API
    def update(self, dets: Detections) -> FrameResult | None:
        """Feed frame ``t``; return the committed result for frame ``t − K`` (``None`` while the
        buffer fills)."""
        self._received += 1
        self.buffer.append(dets)
        if len(self.buffer) <= self.K:
            return None
        return self._commit()

    def flush(self) -> list[FrameResult]:
        """End of stream: commit the buffered frames with whatever look-ahead remains."""
        out = []
        while self.buffer:
            out.append(self._commit())
        return out

    # ---------------------------------------------------------------------------------- internals
    def _score(self, forced: list[tuple[Cluster, Hypothesis]], d0: Detections,
               future: list[Detections], tids: tuple[int, ...]) -> float:  # fmt: skip
        sim = self.state.clone()
        id0 = sim._next_id
        sim.stage1_hook = force_hook(forced)
        sim.update(d0)
        sim.stage1_hook = None
        total = self._frame_cost(sim, tids, id0)
        for d in future:
            sim.update(d)
            total += self._frame_cost(sim, tids, id0)
        self.stats.rollouts += 1
        return total

    def _frame_cost(self, sim: ByteTrackLike, tids: tuple[int, ...], id0: int) -> float:
        c = sum(sim.match_costs.get(t, self.miss_penalty) for t in tids)
        if self.birth_penalty:
            c += self.birth_penalty * sum(t.track_id >= id0 for t in sim.tracks)
        return c

    def _commit(self) -> FrameResult:
        d0 = self.buffer.popleft()
        future = list(self.buffer)
        self.stats.frames += 1
        trial = self.state.clone()
        res = trial.update(d0)
        rec = trial.last_stage1
        clusters: list[Cluster] = []
        if rec is not None:
            clusters, skipped = find_clusters(
                rec, self.margin, self.max_cluster, self.confirmed_only,
                self.trigger_max_visibility, self.trigger_require_all,
            )  # fmt: skip
            self.stats.skipped_large += skipped
        f = trial.frame_id
        if clusters:  # recorded with K = 0 too: the "flagged windows" of the go/no-go
            self.stats.ambiguous_frames += 1
            self.stats.clusters += len(clusters)
            self.stats.trigger_frames.append(f)
        if not clusters or not future:
            self.state = trial
            return res
        assert rec is not None
        chosen: list[tuple[Cluster, Hypothesis]] = []
        any_change = False
        for cl in clusters:
            hyps = enumerate_hypotheses(cl, rec.thresh, self.max_hyp, self.same_tracks)
            if len(hyps) == 1:
                chosen.append((cl, hyps[0]))
                continue
            tids = cl.track_ids if self.score_scope == "cluster" else rec.track_ids
            scores = [self._score([(cl, h)], d0, future, tids) for h in hyps]
            best = int(np.argmin(scores))  # first minimum: the immediate choice wins ties
            if scores[0] - scores[best] <= self.min_gain:
                best = 0
            chosen.append((cl, hyps[best]))
            any_change |= best != 0
        if any_change:
            self.stats.changed += sum(h != cl.greedy for cl, h in chosen)
            self.stats.changed_frames.append(f)
            new = self.state.clone()
            new.stage1_hook = force_hook(chosen)
            res = new.update(d0)
            new.stage1_hook = None
            self.state = new
        else:
            self.state = trial
        return FrameResult(res.frame_id, res.tracks, self._pending(rec, chosen, f), res.extras)

    def _pending(self, rec: Stage1Record, chosen: list[tuple[Cluster, Hypothesis]],
                 f: int) -> tuple[PendingDecision, ...]:  # fmt: skip
        """One record per cluster track: the candidate detections, their costs and the choice."""
        out = []
        for cl, hyp in chosen:
            pick = dict(hyp)
            for tid in cl.track_ids:
                cols = [c for c in cl.cols if (tid, c) in cl.edges]
                boxes = tuple(tuple(float(v) for v in rec.dets_tlwh[c]) for c in cols)
                out.append(PendingDecision(
                    track_id=tid,
                    candidate_boxes_tlwh=boxes,  # type: ignore[arg-type]
                    candidate_costs=tuple(cl.edges[(tid, c)] for c in cols),
                    opened_frame=f,
                    deadline_frame=f + self.K,
                    chosen_index=cols.index(pick[tid]) if tid in pick else -1,
                    competing_track_ids=tuple(t for t in cl.track_ids if t != tid),
                    changed=hyp != cl.greedy,
                ))  # fmt: skip
        return tuple(out)

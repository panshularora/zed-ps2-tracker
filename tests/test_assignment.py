import itertools

import numpy as np
import pytest

from tracker.assignment import HungarianAssigner, linear_assignment

BACKENDS = ["scipy", "lap"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (3, 0)])
def test_empty(backend: str, shape: tuple[int, int]) -> None:
    m = linear_assignment(np.zeros(shape), 0.5, backend)  # type: ignore[arg-type]
    assert m.matches.shape == (0, 2)
    np.testing.assert_array_equal(m.unmatched_rows, np.arange(shape[0]))
    np.testing.assert_array_equal(m.unmatched_cols, np.arange(shape[1]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_all_gated(backend: str) -> None:
    for cost in (np.full((3, 2), 0.9), np.full((3, 2), np.inf), np.full((3, 2), np.nan)):
        m = linear_assignment(cost, 0.5, backend)  # type: ignore[arg-type]
        assert len(m.matches) == 0
        assert len(m.unmatched_rows) == 3
        assert len(m.unmatched_cols) == 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_simple_optimal(backend: str) -> None:
    cost = np.array([[0.1, 0.4], [0.2, 0.15]])
    m = linear_assignment(cost, 0.5, backend)  # type: ignore[arg-type]
    assert sorted(map(tuple, m.matches.tolist())) == [(0, 0), (1, 1)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_threshold_respected_and_partial(backend: str) -> None:
    cost = np.array([[0.1, 0.9, np.inf], [0.9, 0.9, 0.3], [0.95, 0.99, 0.99]])
    m = linear_assignment(cost, 0.5, backend)  # type: ignore[arg-type]
    assert sorted(map(tuple, m.matches.tolist())) == [(0, 0), (1, 2)]
    assert m.unmatched_rows.tolist() == [2]
    assert m.unmatched_cols.tolist() == [1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_rectangular(backend: str) -> None:
    cost = np.array([[0.3, 0.1, 0.2, 0.4]])
    m = linear_assignment(cost, 0.5, backend)  # type: ignore[arg-type]
    assert m.matches.tolist() == [[0, 1]]
    assert m.unmatched_cols.tolist() == [0, 2, 3]


def test_unmatched_is_priced_globally() -> None:
    # Greedy on 0.1 would leave row 1 unmatched (total 0.1 + 0.8 = 0.9 in unmatched cost);
    # global optimum matches (0,1),(1,0) at 0.4 + 0.4 = 0.8 < 0.9.
    cost = np.array([[0.1, 0.4], [0.4, np.inf]])
    m = linear_assignment(cost, 0.8)
    assert sorted(map(tuple, m.matches.tolist())) == [(0, 1), (1, 0)]


def _brute_force(cost: np.ndarray, thresh: float) -> float:
    n, m = cost.shape
    best = (n + m) * thresh / 2  # nothing matched
    for k in range(1, min(n, m) + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                c = cost[list(rows), list(cols)]
                if np.all(c < thresh):
                    best = min(best, c.sum() + (n + m - 2 * k) * thresh / 2)
    return best


@pytest.mark.parametrize("backend", BACKENDS)
def test_random_matches_brute_force_and_backends_agree(backend: str) -> None:
    rng = np.random.default_rng(42)
    for _ in range(60):
        n, m = rng.integers(1, 5, size=2)
        cost = rng.uniform(0, 1, (n, m))
        cost[rng.uniform(size=(n, m)) < 0.2] = np.inf
        thresh = 0.6
        res = linear_assignment(cost, thresh, backend)  # type: ignore[arg-type]
        k = len(res.matches)
        total = cost[res.matches[:, 0], res.matches[:, 1]].sum() + (n + m - 2 * k) * thresh / 2
        assert total == pytest.approx(_brute_force(cost, thresh))
        assert np.all(cost[res.matches[:, 0], res.matches[:, 1]] < thresh)
        assert len(set(res.matches[:, 0])) == k
        assert len(set(res.matches[:, 1])) == k


def test_bad_inputs() -> None:
    with pytest.raises(ValueError, match="2-D"):
        linear_assignment(np.zeros(3), 0.5)
    with pytest.raises(ValueError, match="backend"):
        linear_assignment(np.zeros((1, 1)), 0.5, "nope")  # type: ignore[arg-type]


def test_assigner_callable() -> None:
    a = HungarianAssigner()
    assert a(np.array([[0.1]]), 0.5).matches.tolist() == [[0, 0]]

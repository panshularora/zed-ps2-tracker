"""Tuning (tau_update / tau_cost) must run on MOT17 train_half only: never on val-half frames."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eval import phase3
from eval.splits import half_range

ROOT = Path(__file__).resolve().parents[1]


def test_plan_tunes_on_train_half() -> None:
    plan = yaml.safe_load((ROOT / "configs/phase3/plan.yaml").read_text())
    assert plan["tune"]["split"] == "train_half"


def test_tune_refuses_val_half(tmp_path: Path) -> None:
    plan = yaml.safe_load((ROOT / "configs/phase3/plan.yaml").read_text())
    for bad in ("val_half", "full", None):
        plan["tune"]["split"] = bad
        with pytest.raises(ValueError, match="train_half"):
            phase3.tune(plan, {"caches": {}}, tmp_path, 1, False)


@pytest.mark.parametrize("n", [525, 1050, 600, 429])
def test_train_and_val_halves_disjoint(n: int) -> None:
    tlo, thi = half_range(n, "train_half")
    vlo, vhi = half_range(n, "val_half")
    assert tlo == 1 and thi < vlo and vhi == n

from pathlib import Path

import pytest

from tracker.config import TrackerConfig, config_to_dict, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_match_baseline_yaml() -> None:
    assert load_config(ROOT / "configs/baselines/bytetrack.yaml") == TrackerConfig()


def test_inherit_and_override() -> None:
    cfg = load_config(ROOT / "configs/baselines/bytetrack_mot20.yaml")
    assert cfg.association.fuse_score is False
    assert cfg.detection.high_thresh == 0.6
    reid = load_config(ROOT / "configs/baselines/bytetrack_reid.yaml", ["appearance.alpha=0.8"])
    assert reid.appearance.enabled
    assert reid.appearance.alpha == 0.8


def test_unknown_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("association:\n  match_thres: 0.7\n")
    with pytest.raises(KeyError, match="match_thres"):
        load_config(p)


def test_type_checked() -> None:
    with pytest.raises(TypeError, match="expects bool"):
        load_config(overrides=["appearance.enabled=3"])
    assert load_config(overrides=["association.match_thresh=1"]).association.match_thresh == 1.0


def test_cycle_detected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("inherit: b.yaml\n")
    (tmp_path / "b.yaml").write_text("inherit: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml")


def test_bad_override() -> None:
    with pytest.raises(ValueError, match="key=value"):
        load_config(overrides=["nope"])


def test_roundtrip_dict() -> None:
    d = config_to_dict(TrackerConfig())
    assert d["detection"]["high_thresh"] == 0.6

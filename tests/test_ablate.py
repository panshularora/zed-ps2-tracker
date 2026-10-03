"""Ablation harness: manifest validation and report writers (no tracking runs)."""

from pathlib import Path

import pytest

from eval.ablate import load_manifest, markdown, write_csv


def test_repo_manifest_is_valid() -> None:
    m = load_manifest(Path("configs/ablations/manifest.yaml"))
    for e in m["entries"]:
        assert Path(e["config"]).exists(), e["config"]
    assert {"bytetrack", "botsort", "ocsort", "deep_ocsort", "hybrid_sort"} <= {
        e["name"] for e in m["entries"]
    }


def test_manifest_rejects_duplicates_and_unknown_keys(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text("dataset: MOT17\nentries:\n  - {name: a, config: x}\n  - {name: a, config: y}\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(p)
    p.write_text("dataset: MOT17\nentries:\n  - {name: a, config: x, typo: 1}\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_manifest(p)


def test_writers(tmp_path: Path) -> None:
    row = {"name": "a", "label": "A", "group": "g", "HOTA": 50.0, "IDF1": 56.0, "MOTA": 47.0,
           "IDSW": 191.0, "AssA": 58.0, "DetA": 43.0, "p50_ms": 0.9, "p95_ms": 1.8,
           "max_ms": 2.5, "stress": {}}  # fmt: skip
    md = markdown({"dataset": "MOT17", "split": "val_half"}, [row])
    assert "| g | A | 50.0 | 56.0 | 47.0 | 191 |" in md
    write_csv([row], tmp_path / "t.csv")
    lines = (tmp_path / "t.csv").read_text().splitlines()
    assert lines[0].startswith("group,name,label,HOTA")
    assert len(lines) == 2

"""Post-GPU pipeline end to end on a synthetic Stage A bundle (real writer, bringback, pack)."""

import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eval import ingest as ing
from eval import phase3
from eval.synth_stagea import build
from tracker.config import load_config
from tracker.io import SequenceCache, load_cache
from tracker.runner import run_sequence

PLAN = Path(__file__).resolve().parents[1] / "configs" / "phase3" / "plan.yaml"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("zed")
    return build(root, n_frames=80, n_people=10)


def test_partial_then_rest_ingest_and_phase3(bundle: Path, tmp_path: Path) -> None:
    dest, nodata = tmp_path / "stagea", tmp_path / "none"
    first = tmp_path / "first"
    first.mkdir()
    for name in ("core.tar", "MOT17__ablation.tar", "MOT20__sparsetrack_ablation.tar"):
        shutil.copy(bundle / name, first / name)
    res = ing.ingest(first, dest, nodata, delete_source=True)
    assert not list(first.glob("*.tar"))  # deleted after verification
    idx = res["index"]["caches"]
    assert set(idx) == {"MOT17/ablation/train", "MOT20/sparsetrack_ablation/train"}
    assert idx["MOT17/ablation/train"]["gt_root"] == str(dest / "gt" / "MOT17")
    assert res["sources"][-1]["still_missing"]  # the rest of the DGX bundle is reported

    out, rep = tmp_path / "out", tmp_path / "report"
    s1 = phase3.run(PLAN, dest / "index.json", out, rep, jobs=4)
    assert [r["id"] for r in s1["ran"]] == [
        "mot17_val_half",
        "mot20_val_half__sparsetrack_ablation",
    ]
    assert any(w.startswith("dancetrack_val") for w in s1["waiting"])
    assert s1["reproduction"]["verdict"] in ("REPRODUCED", "NOT REPRODUCED")
    assert (rep / "mot17_val_half" / "stress_visibility.png").exists()
    lat = json.loads((out / "mot17_val_half" / "bytetrack" / "latency.json").read_text())
    assert lat["combined"]["p95_ms"] > 0
    tuned = json.loads((rep / "tune.json").read_text())
    gng = json.loads((rep / "go_no_go_v2.json").read_text())
    assert gng["replication"]["status"].startswith(("pass", "fail"))  # MOT20 clean was ingested
    assert set(gng["rows"]) == {"k0", "k3", "k5"}
    assert 0.0 <= gng["rows"]["k5"]["flag_rate"] <= 1.0
    assert gng["rows"]["k0"]["flag_rate"] >= 0.0
    assert set(tuned["chosen"]) == {"tau_update", "tau_cost"}

    ing.ingest(bundle, dest, nodata)  # the rest (folder of tars; re-ingesting is idempotent)
    s2 = phase3.run(PLAN, dest / "index.json", out, rep, jobs=4)
    ids = {r["id"] for r in s2["ran"]}
    assert {"dancetrack_val", "mot20_val_half__mot17full_xdomain"} <= ids
    # leaked (SAW_EVAL_HALF) caches are excluded by default and say so
    assert "mot20_val_half__mot20full_SAW_EVAL_HALF" not in ids
    assert not any(r["saw_eval_half"] for r in s2["ran"])
    assert any("SAW_EVAL_HALF" in x and "leaked, excluded" in x for x in s2["excluded"])
    s3 = phase3.run(PLAN, dest / "index.json", out, rep, jobs=4, include_leaked=True)
    assert "mot20_val_half__mot20full_SAW_EVAL_HALF" in {r["id"] for r in s3["ran"]}
    saw_dir = rep / "mot20_val_half__mot20full_SAW_EVAL_HALF"
    saw = json.loads((saw_dir / "ablation.json").read_text())
    assert len(saw) == 10
    md = (saw_dir / "ablation.md").read_text()
    assert "DETECTOR SAW THE EVAL HALF" in md
    rows: list[dict[str, Any]] = json.loads((rep / "mot17_val_half" / "ablation.json").read_text())
    assert len(rows) == 14
    assert all(r["latency_source"] == "eval.latency" for r in rows)
    # the tuned taus were substituted into the gated entries
    run = json.loads((out / "mot17_val_half" / "reid_vis_gated" / "trackers" / "reid_vis_gated"
                      / "run.json").read_text())  # fmt: skip
    assert run["config"]["appearance"]["min_visibility"] == tuned["chosen"]["tau_update"]


def test_embeddings_from_ingested_cache_drive_reid(bundle: Path, tmp_path: Path) -> None:
    ing.ingest(bundle / "MOT17__ablation.tar", tmp_path / "s", tmp_path / "none")
    cdir = tmp_path / "s" / "cache" / "MOT17" / "ablation" / "train"
    c = load_cache(cdir / "MOT17-90-FRCNN.npz")
    assert c.emb is not None
    assert c.emb.shape[1] == 512
    assert c.meta.reid is not None
    e = np.random.default_rng(0).normal(size=c.emb.shape)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    shuffled = SequenceCache(c.meta, c.frame, c.tlwh, c.score, e.astype(np.float16))

    def sig(cache: SequenceCache, cfg: str) -> list[Any]:
        res = run_sequence(cache, load_config(cfg)).results
        return [[(t.track_id, tuple(np.round(t.tlwh, 2))) for t in fr.tracks] for fr in res]

    reid = "configs/baselines/bytetrack_reid.yaml"
    assert sig(c, reid) != sig(shuffled, reid)  # re-ID really reads the cached embeddings
    iou = "configs/baselines/bytetrack.yaml"
    assert sig(c, iou) == sig(shuffled, iou)


def test_tampered_tar_is_rejected(bundle: Path, tmp_path: Path) -> None:
    src = bundle / "MOT17__ablation.tar"
    bad = tmp_path / "bad.tar"
    with tarfile.open(src) as tin, tarfile.open(bad, "w") as tout:
        for ti in tin.getmembers():
            data = tin.extractfile(ti).read()  # type: ignore[union-attr]
            if ti.name.endswith("MOT17-90-FRCNN.npz"):
                data = data[:-1] + bytes([data[-1] ^ 1])
            tout.addfile(ti, io.BytesIO(data))
    with pytest.raises(ing.IngestError, match="sha256 mismatch"):
        ing.ingest(bad, tmp_path / "s", tmp_path / "none")
    assert not (
        tmp_path / "s" / "cache" / "MOT17" / "ablation" / "train" / "MOT17-90-FRCNN.npz"
    ).exists()


def test_disk_check(bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    full = shutil.disk_usage(tmp_path)._replace(free=10)
    monkeypatch.setattr(ing.shutil, "disk_usage", lambda p: full)
    with pytest.raises(ing.IngestError, match="free"):
        ing.ingest(bundle / "MOT17__ablation.tar", tmp_path / "s", tmp_path / "none")


def test_unpacked_bring_back_dir(bundle: Path, tmp_path: Path) -> None:
    res = ing.ingest(bundle.parent / "bring_back", tmp_path / "s", tmp_path / "none")
    assert len(res["index"]["caches"]) == 5
    assert res["sources"][0]["still_missing"] == []

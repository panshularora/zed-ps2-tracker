"""run.sh all, CPU-testable parts: laptop-detector phase 3, run-info stamping, the final
results bundle (build on the GPU host, verify + install locally) and the Stage A ETA."""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from eval import eta, phase3, results_bundle, run_info
from eval import ingest as ing
from eval.synth_stagea import build

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "phase3" / "plan.yaml"
LAPTOP = "MOT17/laptop_yolox_s_xdomain/train"
GPU = {"index": 0, "name": "NVIDIA GeForce RTX 3050 Laptop GPU", "memory_mib": 6144,
       "driver": "581.86", "compute_cap": "8.6"}  # fmt: skip


@pytest.fixture(scope="module")
def laptop_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Synthetic Stage A -> MOT17 cache relabelled as a laptop-detector cache -> phase 3."""
    root = tmp_path_factory.mktemp("zed")
    tars = build(root / "zed", n_frames=60, n_people=8)
    dest = root / "zed" / "stagea"
    ing.ingest(tars / "MOT17__ablation.tar", dest, root / "none")
    ing.ingest(tars / "core.tar", dest, root / "none")
    (dest / "cache" / "MOT17" / "ablation").rename(
        dest / "cache" / "MOT17" / "laptop_yolox_s_xdomain"
    )
    ing.rebuild_index(dest, root / "none")
    out, report = root / "zed" / "phase3" / "out", root / "zed" / "phase3" / "report"
    phase3.run(PLAN, dest / "index.json", out, report, jobs=4, latency=False,
               remap={"MOT17/ablation/train": LAPTOP})  # fmt: skip
    return {"root": root / "zed", "index": dest / "index.json", "report": report}


def test_laptop_detector_rows_are_tagged_and_checks_pending(laptop_run: dict[str, Path]) -> None:
    rep = laptop_run["report"]
    s = json.loads((rep / "summary.json").read_text())
    assert s["tune_cache"] == LAPTOP
    assert s["deviation"]["remap"] == {"MOT17/ablation/train": LAPTOP}
    assert [r["id"] for r in s["ran"]] == ["mot17_val_half"]
    assert s["ran"][0]["laptop_detector"] is True
    assert s["reproduction"]["verdict"] == "PENDING"
    assert s["go_no_go_v2"] == "PENDING"
    rows = json.loads((rep / "mot17_val_half" / "ablation.json").read_text())
    assert len(rows) == 14  # every tracker, baselines included, on the same detector output
    assert all(r["label"].endswith(f"[{phase3.LAPTOP_TAG}]") for r in rows)
    assert phase3.LAPTOP_TAG in (rep / "mot17_val_half" / "ablation.md").read_text()
    assert "PENDING" in (rep / "reproduction.md").read_text()
    assert "PENDING" in (rep / "go_no_go_v2.md").read_text()
    assert "not comparable to published numbers" in (rep / "README.md").read_text()


def test_apply_remap_leaves_wildcards_and_replication_alone() -> None:
    plan = phase3.load_plan(PLAN)
    p = phase3.apply_remap(plan, {"MOT17/ablation/train": LAPTOP})
    assert p["tune"]["cache"] == p["reproduction"]["cache"] == LAPTOP
    assert p["go_no_go_v2"]["primary"]["cache"] == LAPTOP
    assert "MOT20/*/train" in [c["cache"] for c in p["conditions"]]
    assert p["go_no_go_v2"]["replication"]["caches"] == plan["go_no_go_v2"]["replication"]["caches"]
    assert plan["tune"]["cache"] == "MOT17/ablation/train"  # the loaded plan is not mutated


def test_parse_nvidia_smi_is_robust() -> None:
    assert run_info.parse_nvidia_smi("NVIDIA GeForce RTX 3050 Laptop GPU, 6144, 581.86, 8.6\n") == [
        GPU
    ]
    with_units = (
        "name, memory.total [MiB], driver_version, compute_cap\n"
        "A100-SXM4-80GB, 81920 MiB, 535.1, 8.0"
    )
    g = run_info.parse_nvidia_smi(with_units)
    assert g == [{"index": 0, "name": "A100-SXM4-80GB", "memory_mib": 81920, "driver": "535.1",
                  "compute_cap": "8.0"}]  # fmt: skip
    odd = "\nGPU X, [N/A], [Not Supported]\nGPU Y, 8 GiB\n"
    g = run_info.parse_nvidia_smi(odd)
    assert [x["memory_mib"] for x in g] == [None, 8192]
    assert g[0]["driver"] is None
    assert g[1]["index"] == 1
    assert run_info.parse_nvidia_smi("") == []


def test_stamp_every_results_file_idempotently(laptop_run: dict[str, Path], tmp_path: Path) -> None:
    rep = tmp_path / "report"
    shutil.copytree(laptop_run["report"], rep)
    info = run_info.collect(json.loads(laptop_run["index"].read_text()), "laptop", GPU)
    run_info.stamp_report(rep, info)
    run_info.stamp_report(rep, info)  # twice: no duplicated header / columns
    md = (rep / "mot17_val_half" / "ablation.md").read_text()
    assert md.count(run_info.MARK_BEGIN) == 1
    assert "RTX 3050" in md
    assert "laptop" in md
    assert "800x1440" in md
    assert LAPTOP in md
    head = (rep / "mot17_val_half" / "ablation.csv").read_text().splitlines()[0].split(",")
    assert head[-6:] == list(run_info.CSV_COLS)
    assert head.count("run_gpu") == 1
    rows = json.loads((rep / "mot17_val_half" / "ablation.json").read_text())
    assert all(r["run_info"]["run_profile"] == "laptop" for r in rows)
    g = json.loads((rep / "go_no_go_v2.json").read_text())
    assert g["run_info"]["cache"] == LAPTOP
    assert g["run_info"]["run_gpu_mem_mib"] == "6144"
    for p in rep.rglob("*.md"):
        assert p.read_text().startswith(run_info.MARK_BEGIN), p
    det = info["detectors"][LAPTOP]
    assert det["precision"] in ("fp16", "fp32")
    assert det["input_size"]


def _bundle(laptop_run: dict[str, Path], tmp_path: Path) -> Path:
    zed = laptop_run["root"]
    state = zed / "run_all"
    state.mkdir(exist_ok=True)
    (state / "gpu.json").write_text(json.dumps(GPU))
    (zed / "STATUS").write_text("state: DONE\n")
    (zed / "logs").mkdir(exist_ok=True)
    (zed / "logs" / "huge.log").write_bytes(b"x" * 3_000_000)
    return results_bundle.build(zed, ROOT, laptop_run["report"], tmp_path / "final",
                                profile="laptop", label="mot17", index=laptop_run["index"],
                                state=state, max_file_mb=1.0)  # fmt: skip


def test_bundle_manifest_and_sha256(laptop_run: dict[str, Path], tmp_path: Path) -> None:
    path = _bundle(laptop_run, tmp_path)
    assert path.name.startswith("zed_results_")
    assert path.name.endswith("_laptop_mot17.tar.gz")
    assert (tmp_path / "final" / "zed_results_latest.tar.gz").resolve() == path.resolve()
    side = (tmp_path / "final" / (path.name + ".sha256")).read_text().split()
    assert side == [results_bundle.sha256_file(path), path.name]
    m = results_bundle.read_manifest(path)
    assert m["kind"] == results_bundle.KIND
    assert m["profile"] == "laptop"
    assert m["verdicts"] == {"go_no_go_v2": "PENDING", "reproduction": "PENDING"}
    assert set(m["frozen_files_sha256_lf"]) == set(results_bundle.FROZEN)
    paths = {f["path"] for f in m["files"]}
    assert "phase3/report/mot17_val_half/ablation.md" in paths
    assert "configs/phase3/plan.yaml" in paths
    assert "run/gpu.json" in paths
    assert "run/STATUS" in paths
    assert "phase3/stagea_index.json" in paths
    assert not any(p.endswith(".npz") for p in paths)  # caches stay on the GPU host
    assert any("huge.log" in s for s in m["skipped"])
    assert "logs/huge.log" not in paths
    with tarfile.open(path, "r:gz") as tf:
        sums = tf.extractfile("SHA256SUMS").read().decode()  # type: ignore[union-attr]
        for f in m["files"]:
            assert f"{f['sha256']}  {f['path']}" in sums
            data = tf.extractfile(f["path"]).read()  # type: ignore[union-attr]
            assert results_bundle.sha256_bytes(data) == f["sha256"]


def test_make_ingest_accepts_the_results_tar(
    laptop_run: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _bundle(laptop_run, tmp_path)
    report, dest = tmp_path / "docs_results_phase3", tmp_path / "dgx_results"
    report.mkdir()
    (report / "stale.md").write_text("old")
    rc = ing.main(
        ["--bring-back", str(path), "--dest", str(tmp_path / "stagea"), "--report", str(report),
         "--results-dest", str(dest), "--delete-source"]
    )  # fmt: skip
    assert rc == 0
    assert "INGEST OK" in capsys.readouterr().out
    assert (report / "README.md").exists()
    assert not (report / "stale.md").exists()
    assert (report / "mot17_val_half" / "ablation.json").exists()
    assert not path.exists()  # DELETE=1
    assert not (tmp_path / "stagea" / "index.json").exists()  # no Stage A ingest happened
    (got,) = list(dest.iterdir())
    assert (got / "RESULTS_MANIFEST.json").exists()


def test_tampered_results_bundle_is_rejected(laptop_run: dict[str, Path], tmp_path: Path) -> None:
    path = _bundle(laptop_run, tmp_path)
    bad = tmp_path / "zed_results_bad.tar.gz"
    with tarfile.open(path, "r:gz") as tin, tarfile.open(bad, "w:gz") as tout:
        for ti in tin.getmembers():
            data = tin.extractfile(ti).read()  # type: ignore[union-attr]
            if ti.name == "phase3/report/summary.json":
                data = data.replace(b"PENDING", b"GO!!!!!")
            tout.addfile(ti, io.BytesIO(data))
    with pytest.raises(results_bundle.BundleError, match="sha256 mismatch"):
        results_bundle.ingest(bad, tmp_path / "d", tmp_path / "r")
    assert not (tmp_path / "r").exists()
    # the sidecar catches a truncated copy of the tar itself
    side = path.with_name(path.name + ".sha256")
    copy = tmp_path / "copy" / path.name
    copy.parent.mkdir()
    copy.write_bytes(path.read_bytes()[:-10])
    shutil.copy(side, copy.with_name(side.name))
    with pytest.raises(results_bundle.BundleError, match=r"differs"):
        results_bundle.ingest(copy, tmp_path / "d", tmp_path / "r")


def test_eta_after_first_sequence(tmp_path: Path) -> None:
    seqs = tmp_path / "train"
    for name, n in (("MOT17-02-FRCNN", 600), ("MOT17-04-FRCNN", 1050), ("MOT17-05-FRCNN", 837)):
        (seqs / name).mkdir(parents=True)
        (seqs / name / "seqinfo.ini").write_text(f"[Sequence]\nname={name}\nseqLength={n}\n")
    log = tmp_path / "stage_a_log.jsonl"
    rec = {
        "sequence": "MOT17-02-FRCNN",
        "frames": 600,
        "stage_a_seconds": 60.0,
        "stage_a_fps": 10.0,
    }
    log.write_text(json.dumps(rec) + "\n{partial")
    e = eta.estimate(eta.read_log(log), eta.seq_lengths(seqs, "*-FRCNN"),
                     eta.parse_later(["MOT20=8931x2", "DanceTrack=?"]))  # fmt: skip
    assert e["fps"] == pytest.approx(10.0)
    assert e["remaining_frames"] == 1887
    assert e["remaining_seconds"] == pytest.approx(188.7)
    msg = eta.message(e)
    assert "10.0 frames/s" in msg
    assert "~3 min" in msg
    assert "MOT20 ~30 min" in msg
    assert "DanceTrack unknown" in msg
    assert eta.message(eta.estimate([], {}, [])).startswith("Stage A: no finished sequence")
    assert eta.fmt_dur(3 * 3600 + 5 * 60) == "3 h 05 min"

"""`run.sh all` without a GPU: dry-run output, the stage engine (skip/resume/force/fail-fast,
STATUS file) and a stubbed end-to-end laptop run with a CUDA out-of-memory fallback."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts" / "dgx" / "pipeline_lib.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def run_all(
    zed: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    e = {k: v for k, v in os.environ.items() if not k.startswith(("ZED_", "BATCH", "MOT20_V"))}
    e.update(ZED_ROOT=str(zed), **(env or {}))
    return subprocess.run(["bash", str(ROOT / "run.sh"), "all", *args], capture_output=True,
                          text=True, env=e, timeout=600)  # fmt: skip


def stages_of(out: str) -> list[str]:
    return [ln.split()[1] for ln in out.splitlines() if ln[:4].strip().rstrip(".").isdigit()]


def test_dry_run_laptop_runs_mot17_first_and_finalizes_each_dataset(tmp_path: Path) -> None:
    r = run_all(tmp_path, "--profile", "laptop", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "batch 1" in r.stdout
    assert "MOT20 variants: sparsetrack_ablation mot17full_xdomain" in r.stdout
    assert stages_of(r.stdout) == [
        "preflight", "setup", "gpucheck",
        "stagea_mot17", "ingest_mot17", "phase3_mot17", "finalize_mot17",
        "stagea_mot20", "ingest_mot20", "phase3_mot20", "finalize_mot20",
        "stagea_dancetrack", "stagea_sompt22", "ingest_dancetrack", "phase3_dancetrack",
        "finalize_dancetrack",
    ]  # fmt: skip
    assert not (tmp_path / "STATUS").exists()
    assert not (tmp_path / "run_all" / "done").exists()


def test_dry_run_dgx_and_options(tmp_path: Path) -> None:
    gate = tmp_path / "gate.mp4"
    gate.write_bytes(b"x")
    r = run_all(tmp_path, "--profile", "dgx", "--dry-run", "--gate", str(gate), "--with-pexels")
    assert r.returncode == 0, r.stderr
    assert "batch 8" in r.stdout
    assert stages_of(r.stdout) == [
        "preflight", "setup", "gpucheck", "stagea_mot17", "stagea_mot20", "stagea_dancetrack",
        "stagea_sompt22", "stagea_video", "stagea_pexels", "ingest_all", "phase3_all",
        "finalize_all",
    ]  # fmt: skip
    r = run_all(tmp_path, "--profile=laptop", "--dry-run", "--skip", "mot20,dancetrack,sompt22",
                "--until", "phase3_mot17")  # fmt: skip
    assert stages_of(r.stdout) == ["preflight", "setup", "gpucheck", "stagea_mot17",
                                   "ingest_mot17", "phase3_mot17"]  # fmt: skip
    (tmp_path / "run_all" / "done").mkdir(parents=True)
    (tmp_path / "run_all" / "done" / "setup").write_text("2026-10-09\n")
    r = run_all(tmp_path, "--profile", "laptop", "--dry-run", "--skip", "mot20,dancetrack,sompt22")
    assert "skip (done 2026-10-09)" in r.stdout
    r = run_all(tmp_path, "--profile", "laptop", "--dry-run", "--force=setup")
    assert "RUN (forced)" in r.stdout
    for bad in (["--profile", "cloud"], ["--detector", "m"], ["--force=nope"], ["--until", "x"]):
        r = run_all(tmp_path, "--dry-run", *bad)
        assert r.returncode != 0, bad


ENGINE = r"""
set -euo pipefail
source "$LIB"
pipe_exec() {
  echo "$1" >> "$ZED_ROOT/ran.txt"
  case "$1" in
    b) [[ -f "$ZED_ROOT/fail_b" ]] && { echo "b broke"; false; }; true ;;
    s) pipe_skip "no archives" ;;
  esac
}
pipe_add a once "first"
pipe_add chk always "cheap check"
pipe_add b once "second"
pipe_add s once "optional"
pipe_add c once "third"
pipe_run
"""


def engine(zed: Path, **env: str) -> subprocess.CompletedProcess[str]:
    e = {**os.environ, "ZED_ROOT": str(zed), "LIB": str(LIB), **env}
    return subprocess.run(["bash", "-c", ENGINE], capture_output=True, text=True, env=e)


def ran(zed: Path) -> list[str]:
    p = zed / "ran.txt"
    out = p.read_text().split() if p.exists() else []
    p.unlink(missing_ok=True)
    return out


def test_engine_fail_fast_resume_force_and_status(tmp_path: Path) -> None:
    (tmp_path / "fail_b").touch()
    r = engine(tmp_path)
    assert r.returncode != 0
    assert "ERROR in stage b" in r.stdout + r.stderr
    assert ran(tmp_path) == ["a", "chk", "b"]  # stopped at b, c never started
    status = (tmp_path / "STATUS").read_text()
    assert "FAILED at stage b" in status
    assert "[done]" in status
    assert "[FAILED]" in status
    assert "[pending]" in status

    (tmp_path / "fail_b").unlink()
    r = engine(tmp_path)  # resume: a is skipped, the always-stage runs again
    assert r.returncode == 0, r.stderr
    assert ran(tmp_path) == ["chk", "b", "s", "c"]
    status = (tmp_path / "STATUS").read_text()
    assert "state: DONE" in status
    assert "[skipped]" in status
    assert "no archives" in status
    assert {p.name for p in (tmp_path / "run_all" / "done").iterdir()} == {"a", "b", "c"}

    assert engine(tmp_path).returncode == 0
    assert ran(tmp_path) == ["chk", "s"]  # everything done; the skipped stage is retried

    assert engine(tmp_path, PIPE_FORCE="b").returncode == 0
    assert ran(tmp_path) == ["chk", "b", "s", "c"]  # forced b re-runs, c (downstream) too
    assert engine(tmp_path, PIPE_FORCE_ALL="1").returncode == 0
    assert ran(tmp_path) == ["a", "chk", "b", "s", "c"]
    r = engine(tmp_path, PIPE_DRY_RUN="1")
    assert r.returncode == 0
    assert ran(tmp_path) == []
    assert "DRY RUN" in r.stdout


# ---------------------------------------------------------------- stubbed end-to-end (CPU)

FAKE_SMI = """#!/usr/bin/env bash
case "$*" in
  -L) echo "GPU 0: NVIDIA GeForce RTX 3050 Laptop GPU (UUID: GPU-fake)" ;;
  *memory.used*) echo "300" ;;
  *query-gpu*) echo "NVIDIA GeForce RTX 3050 Laptop GPU, 6144, 581.86, 8.6" ;;
  *) echo "fake nvidia-smi" ;;
esac
"""
FAKE_TORCH = """
__version__ = "2.7.1+cu126"
class version: cuda = "12.6"
class _P:
    name = "NVIDIA GeForce RTX 3050 Laptop GPU"
    total_memory = 6144 * 2**20
    major, minor = 8, 6
class cuda:
    is_available = staticmethod(lambda: True)
    get_device_properties = staticmethod(lambda i: _P())
    get_device_name = staticmethod(lambda i: _P.name)
class backends:
    class cudnn:
        is_available = staticmethod(lambda: True)
        version = staticmethod(lambda: 90100)
class _T:
    def __matmul__(self, o): return self
    def all(self): return self
    def item(self): return True
float16 = "float16"
randn = lambda *a, **k: _T()
isfinite = lambda t: t
"""
# run.sh stand-in: YOLOX-X "runs out of memory" on MOT17; YOLOX-S writes synthetic caches
FAKE_RUN = """#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == mot17 ]] || { echo "unexpected target $1"; exit 9; }
if [[ "$DETECTOR" == x ]]; then
  part="$ZED_ROOT/cache/MOT17/ablation/train"
  mkdir -p "$part" && touch "$part/partial"
  echo "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.2 GiB"; exit 1
fi
[[ "$BATCH" == 1 ]] || { echo "laptop batch must be 1, got $BATCH"; exit 8; }
source "$ZED_ROOT/env/bin/activate"
python -m eval.synth_stagea --root "$ZED_ROOT" --only MOT17 --label laptop_yolox_s_SAW_EVAL_HALF \\
  --saw-eval-half --frames 50 --people 6
touch "$ZED_ROOT/data/.extracted_MOT17_train"
"""


def test_stubbed_laptop_run_with_oom_fallback(tmp_path: Path) -> None:
    zed, fake = tmp_path / "zed", tmp_path / "fake"
    (fake / "bin").mkdir(parents=True)
    (fake / "torch").mkdir()
    (fake / "bin" / "nvidia-smi").write_text(FAKE_SMI)
    (fake / "bin" / "nvidia-smi").chmod(0o755)
    (fake / "torch" / "__init__.py").write_text(FAKE_TORCH)
    run_sh = fake / "run.sh"
    run_sh.write_text(FAKE_RUN)
    env_bin = zed / "env" / "bin"
    env_bin.mkdir(parents=True)
    (env_bin / "activate").write_text(
        f'source "{ROOT}/.venv/bin/activate"\nexport PYTHONPATH="{fake}:{ROOT}"\n'
    )
    (zed / "data").mkdir(parents=True)
    (zed / "data" / ".extracted_MOT17_train").touch()
    (zed / "run_all" / "done").mkdir(parents=True)
    (zed / "run_all" / "done" / "setup").write_text("stub\n")  # setup.sh needs a real GPU
    env = {"PATH": f"{fake / 'bin'}:{os.environ['PATH']}", "RUN_SH": str(run_sh),
           "ETA_WATCH": "0", "JOBS": "4", "MIN_FREE_GB": "1"}  # fmt: skip
    args = ("--profile", "laptop", "--skip", "mot20,dancetrack,sompt22")
    if not (ROOT / ".venv" / "bin" / "activate").exists():
        pytest.skip("needs the repo .venv")
    r = run_all(zed, *args, env=env)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    status = (zed / "STATUS").read_text()
    assert "state: DONE" in status
    assert "CUDA out of memory with YOLOX-X on mot17" in status
    assert (zed / "run_all" / "detector").read_text().strip() == "s"
    assert list((zed / "cache_oom_partial").rglob("partial"))  # partial X caches moved aside
    assert not (zed / "cache" / "MOT17" / "ablation").exists()
    gpu = json.loads((zed / "run_all" / "gpu.json").read_text())
    assert gpu["name"] == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert gpu["memory_mib"] == 6144
    rep = zed / "phase3" / "report"
    s = json.loads((rep / "summary.json").read_text())
    assert s["reproduction"]["verdict"] == "PENDING"
    assert s["go_no_go_v2"] == "PENDING"
    md = (rep / "mot17_val_half" / "ablation.md").read_text()
    assert "RTX 3050" in md
    assert "laptop detector, not comparable to published numbers" in md
    tars = sorted((zed / "final").glob("zed_results_*_laptop_mot17.tar.gz"))
    assert len(tars) == 1
    with tarfile.open(tars[0], "r:gz") as tf:
        m = json.loads(tf.extractfile("RESULTS_MANIFEST.json").read())  # type: ignore[union-attr]
    assert m["profile"] == "laptop"
    assert m["run_info"]["gpu"]["memory_mib"] == 6144
    det = m["run_info"]["detectors"]["MOT17/laptop_yolox_s_SAW_EVAL_HALF/train"]
    assert det["precision"] in ("fp16", "fp32")
    assert det["input_size"] == "800x1440"

    # re-run: every once-stage is skipped (resume), only the cheap checks run again
    r = run_all(zed, *args, env=env)
    assert r.returncode == 0, r.stdout[-3000:]
    assert r.stdout.count("done earlier, skipped") == 5
    assert len(list((zed / "final").glob("zed_results_*.tar.gz"))) == 1 + 1  # + latest symlink

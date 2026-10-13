"""ByteTrack reproduction audit: detector post-processing, metadata audit, phase-3 guards."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from eval import phase3
from eval.repro_check import OFFICIAL, audit_detector
from eval.run import parse_seq_overrides
from models.yolox_detector import raw_to_tlwh
from tracker.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_raw_to_tlwh_does_not_clip_like_bytetrack() -> None:
    # one box sticking out of a 100x50 image on the left, letterbox ratio 0.5
    o = np.array([[-10.0, 5.0, 20.0, 45.0, 0.9, 0.5, 0.0],
                  [10.0, 10.0, 10.0, 20.0, 0.8, 1.0, 0.0]], np.float32)  # fmt: skip
    tlwh, score = raw_to_tlwh(o, 0.5, (50, 100))
    assert tlwh.shape == (1, 4)  # the zero-width box is dropped
    np.testing.assert_allclose(tlwh[0], [-20, 10, 60, 80])
    np.testing.assert_allclose(score, [0.45])
    clipped, _ = raw_to_tlwh(o, 0.5, (50, 100), clip=True)
    np.testing.assert_allclose(clipped[0], [0, 10, 40, 40])
    assert raw_to_tlwh(None, 1.0, (1, 1))[0].shape == (0, 4)


def test_audit_flags_old_clipped_caches_and_wrong_settings() -> None:
    ok = {**OFFICIAL, "conf_floor": 0.05}
    assert audit_detector(ok) == []
    assert audit_detector({**ok, "fp16": False}) == []  # FP32 diagnostic is allowed
    old = {k: v for k, v in ok.items() if k != "clip_boxes"}
    assert any("clip_boxes" in i for i in audit_detector(old))
    bad = audit_detector({**ok, "test_size": [608, 1088], "nms_thresh": 0.45, "conf_floor": 0.2})
    assert len(bad) == 3


def test_official_config_matches_bytetrack_bookkeeping() -> None:
    c = load_config(ROOT / "configs" / "baselines" / "bytetrack_official.yaml")
    assert c.lifecycle.predict_tentative is False
    assert c.lifecycle.removal_lag is True
    assert c.association.match_thresh == 0.9
    assert c.detection.high_thresh == 0.6
    assert c.output.min_box_area == 100
    assert c.output.max_aspect_ratio == 1.6
    assert parse_seq_overrides(["MOT17-05:lifecycle.track_buffer=14"]) == {
        "MOT17-05": ["lifecycle.track_buffer=14"]
    }
    with pytest.raises(SystemExit):
        parse_seq_overrides(["nocolon"])


@pytest.mark.parametrize("bad", ["MOT20", "MOT20=", "MOT20=x", "MOT20=0"])
def test_phase3_rejects_bad_timeouts(bad: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        phase3.main(["--index", str(tmp_path / "none.json"), "--timeout-min", bad])


def test_laptop_profile_raises_the_mot20_timeout_and_fp32_flag(tmp_path: Path) -> None:
    script = ("RUN_ALL_SOURCE_ONLY=1 source scripts/dgx/run_all.sh; PROFILE=laptop; phase3_args x;"
              " echo; PROFILE=dgx; phase3_args x; echo '|';"
              " PHASE3_TIMEOUT_MIN_MOT20=90 phase3_args x; echo;"
              " FP16=0; python() { echo \"$@\"; }; stage_a dataset --x")  # fmt: skip
    out = subprocess.run(["bash", "-c", script], cwd=ROOT, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "ZED_ROOT": str(tmp_path),
                              "HOME": str(tmp_path)}, check=True).stdout  # fmt: skip
    laptop, rest = out.split("\n", 1)
    assert "--timeout-min MOT20=240" in laptop
    dgx, env90 = rest.split("|", 1)
    assert "timeout-min" not in dgx
    assert "--timeout-min MOT20=90" in env90
    assert "--no-fp16" in env90


def test_default_mot20_variants_exclude_the_leaked_detector() -> None:
    env = (ROOT / "scripts" / "dgx" / "config.env").read_text()
    line = next(ln for ln in env.splitlines() if ln.startswith(': "${MOT20_VARIANTS'))
    assert "SAW_EVAL_HALF" not in line

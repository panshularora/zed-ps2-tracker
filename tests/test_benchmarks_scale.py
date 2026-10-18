"""Results tab GPU showcase: every count and pair is computed from the committed bundle files."""

import json
from pathlib import Path

from serve.benchmarks import gpu

ROOT = Path(__file__).resolve().parents[1] / "docs" / "results"
P3 = ROOT / "phase3"


def _abl(name: str) -> list[dict]:
    return json.loads((P3 / name / "ablation.json").read_text())


def test_scale_counts_match_the_files():
    g = gpu(ROOT)
    s = g["scale"]
    ran = [
        c for c in json.loads((P3 / "summary.json").read_text())["ran"] if not c["saw_eval_half"]
    ]
    abls = {c["id"]: _abl(c["id"]) for c in ran}
    assert s["evaluations"] == len(ran) == 4
    assert s["tracker_runs"] == sum(len(a) for a in abls.values()) == 44
    assert s["excluded"] == 1  # mot20_val_half__mot20full_SAW_EVAL_HALF, in no table
    assert all("SAW" not in d["id"] for d in g["datasets"])
    frames = {k: sum(b["frames"] for b in a[0]["stress"]["density"]) for k, a in abls.items()}
    assert s["frames"] == sum(frames.values())
    assert s["tracked_frames"] == sum(frames[k] * len(a) for k, a in abls.items())
    assert s["go_no_go"] == {"verdict": "NO-GO", "failed": 5, "total": 5}
    sm = json.loads((ROOT / "stress" / "stress_map.json").read_text())
    assert s["stress"]["cells"] == len(sm["cells"]) == 48
    assert s["stress"]["seeds_per_cell"] == 20
    assert s["stress"]["runs"] == 48 * 20 * len(sm["trackers"])


def test_rows_carry_fp_fn_from_the_same_trackeval_run():
    g = gpu(ROOT)
    m17 = next(d for d in g["datasets"] if d["id"] == "mot17_val_half")
    bt = next(r for r in m17["rows"] if r["name"] == "bytetrack")
    rep = json.loads((P3 / "reproduction.json").read_text())["ours"]
    assert (bt["FP"], bt["FN"], bt["IDSW"]) == (int(rep["FP"]), int(rep["FN"]), int(rep["IDSW"]))


def test_reid_rule_pairs_and_claim_flags_follow_the_rows():
    pairs = {p["family"]: p for p in gpu(ROOT)["reid_rule"]["pairs"]}
    bt = pairs["ByteTrack"]
    got = {r["dataset"]: (r["base"]["IDSW"], r["ours"]["IDSW"]) for r in bt["rows"]}
    assert got == {
        "mot17_val_half": (183, 167),
        "mot20_val_half__mot17full_xdomain": (1160, 1039),
        "mot20_val_half__sparsetrack_ablation": (1278, 1019),
        "dancetrack_val": (1883, 1681),
    }
    for p in pairs.values():
        assert p["fewer_idsw_everywhere"] == all(r["d_IDSW"] < 0 for r in p["rows"])
        assert p["hota_never_lower"] == all(r["d_HOTA"] >= 0 for r in p["rows"])
    assert bt["fewer_idsw_everywhere"]
    assert bt["hota_never_lower"]

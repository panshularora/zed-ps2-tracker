"""Final integrity check + bring-back bundle for Stage A (run by scripts/dgx/finalize.sh).

Checks every cache under ``CACHE``: it loads and validates (schema v1), its frame count equals
the sequence's ``seqinfo.ini``, it has 512-d embeddings, its confidence floor is ≤ 0.1, and the
``SAW_EVAL_HALF`` folder label agrees with the ``detector_saw_eval_half`` flag. Then it assembles
``BRING`` (hard links where possible, so no extra disk) with caches, gt labels, clips, results,
logs and weight checksums, and writes ``MANIFEST.json`` + ``SHA256SUMS``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from tracker.io import load_cache


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def check_cache(path: Path, seq_dirs: dict[str, Path]) -> tuple[list[str], str | None]:
    errs: list[str] = []
    try:
        c = load_cache(path)
    except Exception as e:  # report every bad file, don't stop at the first
        return [f"{path}: unreadable ({e})"], None
    m = c.meta
    seq = seq_dirs.get(m.sequence)
    if seq is not None:
        ini = (seq / "seqinfo.ini").read_text()
        n = next(int(x.split("=")[1]) for x in ini.splitlines() if x.startswith("seqLength"))
        if n != m.seq_length:
            errs.append(f"{path}: seq_length {m.seq_length} != seqinfo {n}")
    if c.emb is None or c.emb.shape[1] != 512:
        errs.append(f"{path}: missing 512-d embeddings")
    floor = float(m.detector.get("conf_floor", 1.0))
    if floor > 0.1:
        errs.append(f"{path}: conf_floor {floor} > 0.1")
    labelled = "SAW_EVAL_HALF" in str(path)
    if labelled != m.detector_saw_eval_half:
        errs.append(f"{path}: folder label vs detector_saw_eval_half={m.detector_saw_eval_half}")
    if len(c) == 0:
        errs.append(f"{path}: zero detections")
    return errs, m.sequence


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m models.bringback")
    p.add_argument("--root", type=Path, required=True, help="ZED_ROOT")
    a = p.parse_args(argv)
    root = a.root
    cache, data, bring = root / "cache", root / "data", root / "bring_back"
    seq_dirs = {d.parent.name: d.parent for d in data.rglob("seqinfo.ini")}
    seq_dirs = {k: v for k, v in seq_dirs.items() if "_seqs" in v.parts or "SOMPT22" not in v.parts}

    caches = sorted(cache.rglob("*.npz"))
    errors: list[str] = []
    have: set[str] = set()
    for c in caches:
        errs, name = check_cache(c, seq_dirs)
        errors += errs
        if name:
            have.add(name)
    if bring.exists():
        shutil.rmtree(bring)
    files: list[Path] = []
    for c in caches:
        dst = bring / "cache" / c.relative_to(cache)
        place(c, dst)
        files.append(dst)
    for log in cache.rglob("stage_a_log.jsonl"):
        dst = bring / "cache" / log.relative_to(cache)
        place(log, dst)
        files.append(dst)
    # gt + seqinfo for every sequence we have a cache for (DanceTrack/SOMPT22 labels aren't
    # downloadable separately; MOT17/MOT20 ones are, but bringing them keeps the bundle complete)
    for name, d in sorted(seq_dirs.items()):
        if name not in have:
            continue
        rel = Path(d.relative_to(data))
        for fname in ("seqinfo.ini", "gt/gt.txt"):
            if (d / fname).exists():
                dst = bring / "gt" / rel / fname
                place((d / fname).resolve(), dst)
                files.append(dst)
    for sub in ("clips", "results", "logs"):
        for src in sorted((root / sub).rglob("*")) if (root / sub).exists() else []:
            if src.is_file() and not src.name.endswith(".tmp.mp4"):
                dst = bring / sub / src.relative_to(root / sub)
                place(src, dst)
                files.append(dst)
    sums = root / "weights" / "SHA256SUMS"
    if sums.exists():
        place(sums, bring / "weights_SHA256SUMS")
        files.append(bring / "weights_SHA256SUMS")

    manifest: dict[str, Any] = {"errors": errors, "files": []}
    total: dict[str, int] = {}
    for f in files:
        size = f.stat().st_size
        top = f.relative_to(bring).parts[0]
        total[top] = total.get(top, 0) + size
        manifest["files"].append(
            {"path": str(f.relative_to(bring)), "bytes": size, "sha256": sha256(f)}
        )
    manifest["bytes_by_category"] = total
    manifest["bytes_total"] = sum(total.values())
    (bring / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    with (bring / "SHA256SUMS").open("w") as fh:
        for e in manifest["files"]:
            fh.write(f"{e['sha256']}  {e['path']}\n")

    print(f"\n{'category':<10} {'size':>10}")
    for k, v in sorted(total.items()):
        print(f"{k:<10} {v / 1e6:>8.1f} MB")
    print(
        f"{'TOTAL':<10} {manifest['bytes_total'] / 1e9:>8.2f} GB   ({len(files)} files) -> {bring}"
    )
    if errors:
        print("\nINTEGRITY ERRORS:")
        for e in errors:
            print("  " + e)
        return 1
    print(f"\nINTEGRITY OK: {len(caches)} caches verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

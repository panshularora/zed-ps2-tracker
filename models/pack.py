"""Pack ``bring_back/`` into a few uncompressed tar files for ``scp`` (run by scripts/dgx/pack.sh).

``bring_back_tars/core.tar`` holds ``MANIFEST.json``, ``SHA256SUMS``, ``weights_SHA256SUMS``,
``gt/``, ``logs/``, ``results/`` and every ``stage_a_log.jsonl``. Each cache folder
``cache/<dataset>/<detector-label>/`` becomes ``<dataset>__<label>.tar`` and the demo clips
``clips.tar``. Every tar also carries ``MANIFEST.json`` + ``SHA256SUMS``, so the CPU side
(``python -m eval.ingest``) can verify any single tar on its own. Paths inside the tars are
relative to ``bring_back/``, exactly as in ``SHA256SUMS``.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

INDEX = ("MANIFEST.json", "SHA256SUMS")


def groups(bring: Path) -> dict[str, list[Path]]:
    """Map tar name → files (relative to ``bring``) that go into it."""
    out: dict[str, list[Path]] = {"core": []}
    for f in sorted(p for p in bring.rglob("*") if p.is_file()):
        rel = f.relative_to(bring)
        top = rel.parts[0]
        if rel.name in INDEX and len(rel.parts) == 1:
            continue
        if top == "cache" and rel.suffix == ".npz" and len(rel.parts) >= 4:
            out.setdefault(f"{rel.parts[1]}__{rel.parts[2]}", []).append(rel)
        elif top == "clips":
            out.setdefault("clips", []).append(rel)
        else:
            out["core"].append(rel)
    return out


def pack(bring: Path, dest: Path) -> dict[str, int]:
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.tar"):
        old.unlink()
    sizes: dict[str, int] = {}
    for name, files in groups(bring).items():
        path = dest / f"{name}.tar"
        tmp = path.with_suffix(".tar.tmp")
        with tarfile.open(tmp, "w") as tf:
            for idx in INDEX:
                if (bring / idx).exists():
                    tf.add(bring / idx, arcname=idx)
            for rel in files:
                tf.add(bring / rel, arcname=str(rel))
        tmp.rename(path)
        sizes[path.name] = path.stat().st_size
    return sizes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m models.pack")
    p.add_argument("--root", type=Path, required=True, help="ZED_ROOT")
    a = p.parse_args(argv)
    bring = a.root / "bring_back"
    if not (bring / "SHA256SUMS").exists():
        print(f"no {bring}/SHA256SUMS: run scripts/dgx/finalize.sh first")
        return 1
    sizes = pack(bring, a.root / "bring_back_tars")
    for k, v in sorted(sizes.items()):
        print(f"{k:<45} {v / 1e6:>9.1f} MB")
    print(f"{'TOTAL':<45} {sum(sizes.values()) / 1e9:>9.2f} GB -> {a.root / 'bring_back_tars'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""``make ingest BRING_BACK=<path>``: verify a Stage A bring-back and place it where eval expects.

    python -m eval.ingest --bring-back /workspace/uploads/zed-bring-back/MOT17__ablation.tar
    python -m eval.ingest --bring-back /workspace/uploads/zed-bring-back        # dir of tars
    python -m eval.ingest --bring-back ~/zed-panshul/bring_back                 # unpacked bundle

``<path>`` is a ``bring_back/`` folder (``models/bringback.py``), one tar written by
``scripts/dgx/pack.sh`` (``models/pack.py``), or a folder of such tars. Partial bundles are the
normal case: ingest MOT17 + MOT20 first, delete the tars, and ingest the rest later. Each run
merges into the same destination and index.

For every file it places, ingest:

1. checks the sha256 against the bundle's ``SHA256SUMS`` (computed while streaming, so a tar is
   never unpacked twice); a file missing from ``SHA256SUMS`` or with a wrong hash is rejected;
2. for caches, loads and validates schema v1 and checks that the ``SAW_EVAL_HALF`` folder label
   agrees with the ``detector_saw_eval_half`` flag inside the cache;
3. refuses to start if the free disk is smaller than the files it would write, plus a margin.

Layout under ``--dest`` (default ``data/stagea``)::

    cache/<dataset>/<detector-label>/<split>/<seq>.npz   (+ stage_a_log.jsonl)
    gt/<dataset>/.../<seq>/{seqinfo.ini,gt/gt.txt}       labels shipped with the bundle
    clips/, logs/, results/, weights_SHA256SUMS
    index.json   one entry per cache folder: dataset, split, sequences, saw_eval_half, gt_root

``gt_root`` is what ``eval.run --root`` needs. It is the bundle's own labels when present,
else ``data/<dataset>`` (the MOT17/MOT20 public labels from ``make data``). ``eval.phase3``
reads ``index.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from tracker.io import load_cache

INDEX_FILES = ("MANIFEST.json", "SHA256SUMS")
DISK_MARGIN = 200 * 1024 * 1024  # leave 200 MB free


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class Member:
    rel: str
    size: int
    open: Any  # () -> binary file object


def parse_sums(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            out[rel.strip().lstrip("*").removeprefix("./")] = digest
    return out


def _dir_members(root: Path) -> Iterator[Member]:
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = f.relative_to(root).as_posix()
        yield Member(rel, f.stat().st_size, lambda f=f: f.open("rb"))


def _tar_members(tf: tarfile.TarFile) -> Iterator[Member]:
    for ti in tf.getmembers():
        if not ti.isfile():
            continue
        name = ti.name.removeprefix("./")
        if name.startswith("/") or ".." in Path(name).parts:
            raise IngestError(f"unsafe path in tar: {ti.name}")
        yield Member(name, ti.size, lambda ti=ti: tf.extractfile(ti))


def sources(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if (path / "SHA256SUMS").exists():
        return [path]
    tars = sorted(path.glob("*.tar"))
    if not tars:
        raise IngestError(f"{path}: neither a bring_back folder (SHA256SUMS) nor a folder of tars")
    return tars


def _copy_hashed(src: IO[bytes], dst: Path) -> str:
    h = hashlib.sha256()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with tmp.open("wb") as out:
        while b := src.read(1 << 20):
            h.update(b)
            out.write(b)
    tmp.replace(dst)
    return h.hexdigest()


def check_cache(path: Path, rel: str) -> dict[str, Any]:
    c = load_cache(path)  # validates schema v1
    labelled = "SAW_EVAL_HALF" in rel
    if labelled != c.meta.detector_saw_eval_half:
        raise IngestError(
            f"{rel}: folder label vs detector_saw_eval_half={c.meta.detector_saw_eval_half}"
        )
    return {"sequence": c.meta.sequence, "dataset": c.meta.dataset, "split": c.meta.split,
            "saw_eval_half": c.meta.detector_saw_eval_half, "detections": len(c),
            "has_embeddings": c.emb is not None}  # fmt: skip


def ingest_one(src: Path, dest: Path) -> dict[str, Any]:
    """Verify and place one source (tar or folder). Returns a summary."""
    with contextlib.ExitStack() as stack:
        tf = stack.enter_context(tarfile.open(src, "r:")) if src.is_file() else None
        members = list(_tar_members(tf) if tf else _dir_members(src))
        by_rel = {m.rel: m for m in members}
        if "SHA256SUMS" not in by_rel:
            raise IngestError(f"{src}: no SHA256SUMS (was it made by finalize.sh / pack.sh?)")
        with by_rel["SHA256SUMS"].open() as fh:
            sums = parse_sums(fh.read().decode())
        payload = [m for m in members if m.rel not in INDEX_FILES]
        unknown = [m.rel for m in payload if m.rel not in sums]
        if unknown:
            raise IngestError(f"{src}: files not in SHA256SUMS: {unknown[:5]}")
        need = sum(m.size for m in payload)
        dest.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(dest).free
        if need + DISK_MARGIN > free:
            raise IngestError(
                f"{src}: needs {need / 1e9:.2f} GB but only {free / 1e9:.2f} GB is free under "
                f"{dest}; delete already-ingested tars or ingest a smaller part first"
            )
        placed, caches = [], []
        for m in payload:
            target = dest / m.rel
            with m.open() as fh:
                digest = _copy_hashed(fh, target)
            if digest != sums[m.rel]:
                target.unlink()
                raise IngestError(f"{m.rel}: sha256 mismatch (corrupt copy; copy it again)")
            if m.rel.startswith("cache/") and m.rel.endswith(".npz"):
                try:
                    caches.append({"rel": m.rel, **check_cache(target, m.rel)})
                except Exception:
                    target.unlink()
                    raise
            placed.append(m.rel)
        if "MANIFEST.json" in by_rel:
            with by_rel["MANIFEST.json"].open() as fh:
                manifest = json.loads(fh.read().decode())
            (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
            listed = {f["path"] for f in manifest.get("files", [])}
            missing = sorted(p for p in listed if not (dest / p).exists())
        else:
            missing = []
        return {"source": str(src), "placed": len(placed), "bytes": need, "caches": caches,
                "still_missing": missing}  # fmt: skip


def _gt_root(dest: Path, dataset: str, seqs: list[str], data_dir: Path) -> str | None:
    """Folder that ``eval.run --root`` accepts for these sequences (labels with gt.txt)."""
    gt = dest / "gt"
    hits = [p.parent.parent for p in gt.rglob("gt/gt.txt") if p.parent.parent.name in seqs]
    if hits:
        parent = hits[0].parent  # the folder that holds the sequence folders
        # find_sequences looks in <root>/train (or <root>/val for DanceTrack) first
        return str(parent.parent if parent.name in ("train", "val") else parent)
    fallback = data_dir / dataset
    sub = fallback / ("val" if dataset.lower() == "dancetrack" else "train")
    if all((sub / s / "gt" / "gt.txt").exists() for s in seqs):
        return str(fallback)
    return None


def rebuild_index(dest: Path, data_dir: Path) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for npz in sorted((dest / "cache").rglob("*.npz")):
        key = npz.parent.relative_to(dest / "cache").as_posix()
        c = load_cache(npz)
        g = groups.setdefault(key, {"dir": str(npz.parent), "dataset": c.meta.dataset,
                                    "split": c.meta.split, "sequences": [],
                                    "saw_eval_half": c.meta.detector_saw_eval_half,
                                    "has_embeddings": c.emb is not None})  # fmt: skip
        g["sequences"].append(c.meta.sequence)
        g["has_embeddings"] = g["has_embeddings"] and c.emb is not None
    for g in groups.values():
        g["gt_root"] = _gt_root(dest, g["dataset"], g["sequences"], data_dir)
    index = {"caches": groups}
    (dest / "index.json").write_text(json.dumps(index, indent=1))
    return index


def ingest(path: Path, dest: Path, data_dir: Path, delete_source: bool = False) -> dict[str, Any]:
    out = []
    for src in sources(path):
        s = ingest_one(src, dest)
        out.append(s)
        print(f"[ok] {src.name}: {s['placed']} files, {s['bytes'] / 1e6:.1f} MB, "
              f"{len(s['caches'])} caches verified")  # fmt: skip
        if delete_source and src.is_file():
            src.unlink()
            print(f"     deleted {src}")
    index = rebuild_index(dest, data_dir)
    return {"sources": out, "index": index}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.ingest", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--bring-back", type=Path, required=True)
    p.add_argument("--dest", type=Path, default=Path("data/stagea"))
    p.add_argument("--data", type=Path, default=Path("data"), help="public MOT labels")
    p.add_argument(
        "--delete-source",
        action="store_true",
        help="delete each tar after it is verified and placed (saves disk)",
    )
    a = p.parse_args(argv)
    try:
        res = ingest(a.bring_back, a.dest, a.data, a.delete_source)
    except IngestError as e:
        print(f"INGEST FAILED: {e}")
        return 1
    print(f"\n{'cache folder':<45} {'seqs':>4}  emb  gt_root")
    for k, g in sorted(res["index"]["caches"].items()):
        flag = " SAW_EVAL_HALF" if g["saw_eval_half"] else ""
        print(f"{k:<45} {len(g['sequences']):>4}  {'yes' if g['has_embeddings'] else 'no ':<3}  "
              f"{g['gt_root'] or 'MISSING (ingest core.tar)'}{flag}")  # fmt: skip
    missing = res["sources"][-1]["still_missing"] if res["sources"] else []
    if missing:
        print(
            f"\n{len(missing)} files of the DGX bundle are not here yet (e.g. {missing[0]}); "
            "ingest the remaining tars later."
        )
    print("\nINGEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The final results bundle of ``run.sh all``: build it on the GPU host, ingest it locally.

**Build** (``scripts/dgx/finalize.sh --results``, last stage of ``run.sh all``) packs only the
small final artifacts into ONE ``zed_results_<time>_<profile>_<label>.tar.gz``:

    phase3/report/**          tables (md/csv/json), stress plots, tune, reproduction, go/no-go v2,
                              gate events, run_info.json (all stamped with detector/GPU/profile)
    phase3/stagea_index.json  the ingested Stage A index (cache folders, gt roots)
    stagea/MANIFEST.json      the Stage A bring-back manifest (sha256 of every cache)
    stagea/cache/**/stage_a_log.jsonl, stagea/weights_SHA256SUMS
    configs/**, docs/preregistration/**   the exact configs and the frozen rule used
    logs/**                   run logs (files over --max-file-mb are listed as skipped)
    run/STATUS, run/*.json, run/env.txt   the pipeline status, GPU record, environment dump
    results/**                small DGX-side tracker outputs (SOMPT22 test format), if any
    RESULTS_MANIFEST.json     kind, profile, run_info, verdicts, git commit, frozen-file sha256,
                              every file with size + sha256
    SHA256SUMS                sha256sum -c compatible

A ``<tar>.sha256`` sidecar covers the tar itself. Caches, images, clips and weights stay on the
GPU host.

**Ingest** (``make ingest BRING_BACK=<that tar>``, dispatched by ``eval/ingest.py``) checks
every file against ``SHA256SUMS`` (no unlisted files, no unsafe paths), extracts the bundle to
``data/dgx_results/<bundle>/`` and installs ``phase3/report`` as ``docs/results/phase3/``, which
the web app's Results page and ``/api/benchmarks`` read. ``make phase3`` is NOT needed
afterwards (it needs the caches, which stay on the GPU host).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

KIND = "zed-ps2-final-results"
SCHEMA = 1
MANIFEST = "RESULTS_MANIFEST.json"
SUMS = "SHA256SUMS"
FROZEN = ("configs/phase3/plan.yaml", "docs/preregistration/go_no_go_v2.md",
          "configs/ours/delayed_v2_reid.yaml")  # fmt: skip


class BundleError(RuntimeError):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None


def frozen_hashes(repo: Path) -> dict[str, str | None]:
    """sha256 of the frozen pre-registration files with line endings normalised to LF (a
    Windows clone may have CRLF; the content is what is frozen)."""
    out: dict[str, str | None] = {}
    for rel in FROZEN:
        p = repo / rel
        out[rel] = sha256_bytes(p.read_bytes().replace(b"\r\n", b"\n")) if p.exists() else None
    return out


def collect_files(
    zed_root: Path,
    repo: Path,
    report: Path,
    index: Path | None,
    state: Path | None,
    max_file_mb: float,
) -> tuple[dict[str, Path], list[str]]:
    """Map arcname -> source file. Returns (files, skipped-notes)."""
    files: dict[str, Path] = {}
    skipped: list[str] = []
    limit = max_file_mb * 1e6

    def add_tree(src: Path, prefix: str, suffixes: tuple[str, ...] | None = None,
                 capped: bool = False) -> None:  # fmt: skip
        if not src.exists():
            return
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            if suffixes and f.suffix not in suffixes:
                continue
            if f.name.endswith((".part", ".tmp")):
                continue
            arc = f"{prefix}/{f.relative_to(src).as_posix()}"
            if capped and f.stat().st_size > limit:
                skipped.append(f"{arc} ({f.stat().st_size / 1e6:.1f} MB > {max_file_mb} MB)")
                continue
            files[arc] = f

    if not (report / "summary.json").exists():
        raise BundleError(f"{report}/summary.json missing: did `make phase3` run?")
    add_tree(report, "phase3/report")
    if index is not None and index.exists():
        files["phase3/stagea_index.json"] = index
    bring = zed_root / "bring_back"
    for name in ("MANIFEST.json", "weights_SHA256SUMS"):
        if (bring / name).exists():
            files[f"stagea/{name}"] = bring / name
    for log in sorted((bring / "cache").rglob("stage_a_log.jsonl")) if bring.exists() else []:
        files[f"stagea/{log.relative_to(bring).as_posix()}"] = log
    add_tree(repo / "configs", "configs")
    add_tree(repo / "docs" / "preregistration", "docs/preregistration")
    add_tree(zed_root / "logs", "logs", capped=True)
    add_tree(zed_root / "run_all" / "logs", "logs/run_all", capped=True)
    add_tree(zed_root / "results", "results", capped=True)
    if state is not None and state.exists():
        for f in sorted(state.glob("*")):
            if f.is_file() and f.suffix in (".json", ".txt", ".env") and f.stat().st_size <= limit:
                files[f"run/{f.name}"] = f
    status = zed_root / "STATUS"
    if status.exists():
        files["run/STATUS"] = status
    return files, skipped


def build(zed_root: Path, repo: Path, report: Path, out_dir: Path, *, profile: str,
          label: str = "final", index: Path | None = None, state: Path | None = None,
          run_info: dict[str, Any] | None = None, max_file_mb: float = 20.0,
          now: dt.datetime | None = None) -> Path:  # fmt: skip
    now = now or dt.datetime.now().astimezone()
    files, skipped = collect_files(zed_root, repo, report, index, state, max_file_mb)
    summary = json.loads((report / "summary.json").read_text())
    # read each file ONCE: the same bytes are hashed and archived (a live log may still grow)
    blobs = {arc: src.read_bytes() for arc, src in sorted(files.items())}
    entries = [
        {"path": arc, "bytes": len(b), "sha256": sha256_bytes(b)} for arc, b in blobs.items()
    ]
    info = run_info
    if info is None and (report / "run_info.json").exists():
        info = json.loads((report / "run_info.json").read_text())
    rep = summary.get("reproduction") or {}
    manifest: dict[str, Any] = {
        "kind": KIND,
        "schema": SCHEMA,
        "created": now.isoformat(timespec="seconds"),
        "profile": profile,
        "label": label,
        "run_info": info or {},
        "verdicts": {
            "go_no_go_v2": summary.get("go_no_go_v2", "not computed"),
            "reproduction": rep.get("verdict", "not computed"),
        },
        "phase3_summary": summary,
        "git": {
            "commit": _git(repo, "rev-parse", "HEAD"),
            "dirty": bool(_git(repo, "status", "--porcelain", "--untracked-files=no")),
        },
        "frozen_files_sha256_lf": frozen_hashes(repo),
        "skipped": skipped,
        "files": entries,
    }
    sums = "".join(f"{e['sha256']}  {e['path']}\n" for e in entries)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    name = f"zed_results_{stamp}_{profile}_{label}.tar.gz"
    path = out_dir / name
    tmp = out_dir / (name + ".part")
    with tarfile.open(tmp, "w:gz") as tf:
        head = {MANIFEST: json.dumps(manifest, indent=1).encode(), SUMS: sums.encode()}
        for arc, data in [*head.items(), *blobs.items()]:
            ti = tarfile.TarInfo(arc)
            ti.size = len(data)
            ti.mtime = int(now.timestamp())
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
    tmp.replace(path)
    digest = sha256_file(path)
    (out_dir / (name + ".sha256")).write_text(f"{digest}  {name}\n")
    latest = out_dir / "zed_results_latest.tar.gz"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(name)
    (out_dir / "LATEST").write_text(name + "\n")
    return path


# ------------------------------------------------------------------------- local side


def read_manifest(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:*") as tf:
        fh = tf.extractfile(MANIFEST)
        if fh is None:
            raise BundleError(f"{path.name}: {MANIFEST} is not a file")
        m: dict[str, Any] = json.loads(fh.read())
    return m


def is_results_bundle(path: Path) -> bool:
    if not path.is_file() or not tarfile.is_tarfile(path):
        return False
    with tarfile.open(path, "r:*") as tf:
        try:
            tf.getmember(MANIFEST)
        except KeyError:
            return False
    return True


def _read(tf: tarfile.TarFile, m: tarfile.TarInfo) -> bytes:
    fh = tf.extractfile(m)
    if fh is None:
        raise BundleError(f"{m.name}: not a regular file")
    return fh.read()


def _safe(name: str) -> str:
    n = name.removeprefix("./")
    if n.startswith("/") or ".." in Path(n).parts:
        raise BundleError(f"unsafe path in bundle: {name}")
    return n


def ingest(path: Path, results_dest: Path, report_dest: Path,
           delete_source: bool = False) -> dict[str, Any]:  # fmt: skip
    """Verify every file of a results bundle, extract it, install its phase-3 report."""
    path = path.resolve()  # zed_results_latest.tar.gz is a symlink to the real tar
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists():
        want = sidecar.read_text().split()[0]
        if sha256_file(path) != want:
            raise BundleError(f"{path.name}: sha256 differs from {sidecar.name} (copy it again)")
    stem = path.name.removesuffix(".tar.gz").removesuffix(".tgz").removesuffix(".tar")
    dest = results_dest / stem
    tmp = results_dest / (stem + ".part")
    if tmp.exists():
        shutil.rmtree(tmp)
    with tarfile.open(path, "r:*") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        names = {_safe(m.name): m for m in members}
        if MANIFEST not in names or SUMS not in names:
            raise BundleError(f"{path.name}: no {MANIFEST} / {SUMS}")
        manifest = json.loads(_read(tf, names[MANIFEST]))
        if manifest.get("kind") != KIND:
            raise BundleError(f"{path.name}: kind {manifest.get('kind')!r} != {KIND!r}")
        sums: dict[str, str] = {}
        for line in _read(tf, names[SUMS]).decode().splitlines():
            if line.strip():
                d, rel = line.split(maxsplit=1)
                sums[rel.strip()] = d
        unknown = sorted(n for n in names if n not in sums and n not in (MANIFEST, SUMS))
        if unknown:
            raise BundleError(f"{path.name}: files not in {SUMS}: {unknown[:5]}")
        missing = sorted(n for n in sums if n not in names)
        if missing:
            raise BundleError(f"{path.name}: listed but absent: {missing[:5]}")
        for n, m in names.items():
            data = _read(tf, m)
            if n in sums and sha256_bytes(data) != sums[n]:
                shutil.rmtree(tmp, ignore_errors=True)
                raise BundleError(f"{n}: sha256 mismatch (corrupt copy; copy the tar again)")
            target = tmp / n
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.replace(dest)
    report_src = dest / "phase3" / "report"
    if report_src.exists():
        if report_dest.exists():
            shutil.rmtree(report_dest)
        report_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(report_src, report_dest)
    if delete_source:
        path.unlink()
        if sidecar.exists():
            sidecar.unlink()
    return {"dest": str(dest), "report": str(report_dest), "files": len(sums),
            "profile": manifest.get("profile"), "verdicts": manifest.get("verdicts", {}),
            "run_info": manifest.get("run_info", {})}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.results_bundle", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--zed-root", type=Path, required=True)
    b.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    b.add_argument("--report", type=Path, required=True)
    b.add_argument("--index", type=Path, default=None)
    b.add_argument("--state", type=Path, default=None, help="run_all state folder")
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--profile", required=True)
    b.add_argument("--label", default="final")
    b.add_argument("--max-file-mb", type=float, default=20.0)
    i = sub.add_parser("ingest")
    i.add_argument("--bundle", type=Path, required=True)
    i.add_argument("--results-dest", type=Path, default=Path("data/dgx_results"))
    i.add_argument("--report", type=Path, default=Path("docs/results/phase3"))
    i.add_argument("--delete-source", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.cmd == "build":
            path = build(a.zed_root, a.repo, a.report, a.out_dir, profile=a.profile,
                         label=a.label, index=a.index, state=a.state,
                         max_file_mb=a.max_file_mb)  # fmt: skip
            m = read_manifest(path)
            print(f"{path}  ({path.stat().st_size / 1e6:.2f} MB, {len(m['files'])} files)")
            for s in m["skipped"]:
                print(f"  skipped (too big): {s}")
            print(f"verdicts: {m['verdicts']}")
            print(f"RESULTS BUNDLE OK: {path}")
        else:
            r = ingest(a.bundle, a.results_dest, a.report, a.delete_source)
            print(json.dumps(r["verdicts"]))
            print(f"RESULTS INGEST OK: {r['files']} files -> {r['dest']}; report -> {r['report']}")
    except BundleError as e:
        print(f"RESULTS BUNDLE FAILED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

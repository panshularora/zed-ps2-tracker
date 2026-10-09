"""Demo clip manifest (``configs/demo/clips.yaml``) and the Pexels fetch step.

``python -m serve.clips fetch [--out data/demo/raw]`` downloads every clip from its Pexels page id
(``https://www.pexels.com/download/video/<id>/``) and normalises it to 1080p (short side) H.264,
CRF 20, no audio, at the source frame rate. ``python -m serve.blur`` then writes the face-blurred
copies that the service is allowed to serve.

These clips have no ground truth. They are for qualitative demos only, never for results tables.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "configs" / "demo" / "clips.yaml"
EVENT_STYLINGS = ("neutral", "alert")
DOWNLOAD_URL = "https://www.pexels.com/download/video/{id}/"


@dataclass(frozen=True)
class DemoClip:
    id: str
    title: str
    file: str
    page: str
    pexels_id: int | None = None
    note: str = ""
    consent: bool = False
    event_styling: str = "neutral"
    gate: tuple[float, float, float, float] | None = None
    hysteresis_px: float = 10.0
    licence: str = "Pexels License"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_styling not in EVENT_STYLINGS:
            raise ValueError(f"{self.id}: event_styling must be one of {EVENT_STYLINGS}")
        # alert styling (tailgate / hold alerts, error colour) needs recorded consent
        if self.event_styling == "alert" and not self.consent:
            raise ValueError(f"{self.id}: event_styling 'alert' requires consent: true")
        if self.gate is not None and (
            len(self.gate) != 4 or not all(0.0 <= v <= 1.0 for v in self.gate)
        ):
            raise ValueError(f"{self.id}: gate must be [x1, y1, x2, y2] in 0..1")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> list[DemoClip]:
    raw = yaml.safe_load(Path(path).read_text())
    clips = []
    seen: set[str] = set()
    for c in raw.get("clips", []):
        gate = c.pop("gate", None)
        known = set(DemoClip.__dataclass_fields__) - {"extra", "gate"}
        clip = DemoClip(
            **{k: v for k, v in c.items() if k in known},
            gate=tuple(float(v) for v in gate) if gate else None,  # type: ignore[arg-type]
            extra={k: v for k, v in c.items() if k not in known},
        )
        if clip.id in seen:
            raise ValueError(f"duplicate clip id {clip.id!r}")
        seen.add(clip.id)
        clips.append(clip)
    return clips


def normalise(src: Path, dst: Path, ffmpeg: str = "ffmpeg") -> None:
    """≤1080p short side, H.264 CRF 20, yuv420p, no audio, +faststart (source fps kept)."""
    vf = "scale='if(gt(iw,ih),-2,min(1080,iw))':'if(gt(iw,ih),min(1080,ih),-2)'"
    tmp = dst.with_suffix(".tmp.mp4")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(dst)


def fetch(clips: list[DemoClip], out: Path, keep_original: bool = False) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for c in clips:
        dst = out / c.file
        if dst.exists() and dst.stat().st_size > 0:
            print(f"have  {dst}")
            done.append(dst)
            continue
        if c.pexels_id is None:
            print(f"skip  {c.id}: no pexels_id (add the file by hand)", file=sys.stderr)
            continue
        orig = out / f"{c.id}.original.mp4"
        req = urllib.request.Request(
            DOWNLOAD_URL.format(id=c.pexels_id), headers={"User-Agent": "Mozilla/5.0"}
        )
        print(f"fetch {c.page}")
        with urllib.request.urlopen(req, timeout=120) as r, orig.open("wb") as fh:
            shutil.copyfileobj(r, fh)
        normalise(orig, dst)
        if not keep_original:
            orig.unlink()
        done.append(dst)
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download + normalise the manifest clips")
    f.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    f.add_argument("--out", type=Path, default=ROOT / "data" / "demo" / "raw")
    f.add_argument("--only", nargs="*", help="clip ids")
    f.add_argument("--keep-original", action="store_true")
    sub.add_parser("list", help="print the manifest")
    a = ap.parse_args(argv)
    clips = load_manifest(getattr(a, "manifest", DEFAULT_MANIFEST))
    if a.cmd == "list":
        for c in clips:
            print(f"{c.id:24s} {c.file:40s} consent={c.consent} styling={c.event_styling}")
        return 0
    if a.only:
        clips = [c for c in clips if c.id in set(a.only)]
    fetch(clips, a.out, a.keep_original)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

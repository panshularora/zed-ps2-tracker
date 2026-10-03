"""Typed tracker configuration loaded from plain YAML.

A YAML file may ``inherit:`` another (path relative to itself); keys are deep-merged, child
wins. Unknown keys are an error, so a typo in an ablation config can't silently run the baseline.
Dotted CLI overrides (``association.match_thresh=0.7``) are applied last.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import yaml

from tracker.kalman import KalmanConfig


@dataclass(frozen=True)
class DetectionConfig:
    high_thresh: float = 0.6
    """Detections with score > high_thresh go to the first association (ByteTrack τ_high)."""
    low_thresh: float = 0.1
    """Detections with low_thresh < score < high_thresh go to the second association (a score
    exactly equal to high_thresh is in neither set, as in ByteTrack)."""
    new_track_thresh: float = 0.7
    """Unmatched high detections above this start a new (tentative) track."""


@dataclass(frozen=True)
class AssociationConfig:
    match_thresh: float = 0.9
    """Max cost for the first (high-score) association."""
    second_match_thresh: float = 0.5
    """Max IoU cost for the second (low-score) association."""
    unconfirmed_match_thresh: float = 0.7
    """Max cost for matching tentative tracks to leftover high detections."""
    fuse_score: bool = True
    """ByteTrack's IoU × score fusion (on for MOT17, off for MOT20 in ByteTrack)."""
    duplicate_iou_dist: float = 0.15
    """Confirmed/lost track pairs closer than this (1 − IoU) are de-duplicated."""
    backend: Literal["scipy", "lap"] = "scipy"


@dataclass(frozen=True)
class LifecycleSection:
    min_hits: int = 2
    tentative_max_misses: int = 0
    track_buffer: int = 30
    """Lost-track lifetime in frames at 30 fps."""
    scale_buffer_by_fps: bool = False
    """Scale ``track_buffer`` by ``frame_rate / 30``. ByteTrack's evaluator never passes the frame
    rate, so its effective buffer is always 30 frames (default false, matching that)."""
    confirm_on_first_frame: bool = True


@dataclass(frozen=True)
class AppearanceConfig:
    enabled: bool = False
    """Use re-ID embeddings (needs a Stage A cache; public MOT det.txt has none)."""
    alpha: float = 0.9
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    min_visibility: float = 0.0
    """Skip EMA updates below this estimated visibility (0 = never skip; phase 2 hook)."""


@dataclass(frozen=True)
class VisibilityConfig:
    estimator: Literal["constant", "depth_order"] = "constant"


@dataclass(frozen=True)
class DelayedAssignmentConfig:
    """Bounded-latency ambiguity-aware assignment. **Planned (phase 2), not built.**"""

    enabled: bool = False
    max_delay_frames: int = 3
    margin: float = 0.05


@dataclass(frozen=True)
class OutputConfig:
    min_box_area: float = 100.0
    max_aspect_ratio: float = 1.6
    """Drop reported boxes with w / h above this (ByteTrack's MOT 'vertical' filter); 0 = off."""


@dataclass(frozen=True)
class TrackerConfig:
    name: str = "bytetrack_like"
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    lifecycle: LifecycleSection = field(default_factory=LifecycleSection)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    delayed: DelayedAssignmentConfig = field(default_factory=DelayedAssignmentConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


T = TypeVar("T")


def _deep_merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _from_dict(cls: type[T], data: dict[str, Any], where: str = "") -> T:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise KeyError(f"unknown config key(s) at '{where or '<root>'}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    defaults = cls()
    for name, value in data.items():
        current = getattr(defaults, name)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"'{where}{name}' must be a mapping")
            kwargs[name] = _from_dict(type(current), value, f"{where}{name}.")
        else:
            if (
                isinstance(current, float)
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                value = float(value)
            if type(current) is not type(value):
                raise TypeError(
                    f"'{where}{name}' expects {type(current).__name__}, got {type(value).__name__}"
                )
            kwargs[name] = value
    return cast(T, cls(**kwargs))


def _load_yaml_tree(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"config inheritance cycle: {[*seen, path]}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: top level must be a mapping")
    parent = raw.pop("inherit", None)
    if parent is None:
        return raw
    base = _load_yaml_tree((path.parent / str(parent)), (*seen, path))
    return _deep_merge(base, raw)


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(tree: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = _deep_merge({}, tree)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, val = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot override inside non-mapping at {key!r}")
        node[parts[-1]] = _parse_scalar(val)
    return out


def config_from_dict(data: dict[str, Any]) -> TrackerConfig:
    return _from_dict(TrackerConfig, data)


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> TrackerConfig:
    tree = _load_yaml_tree(Path(path)) if path is not None else {}
    return config_from_dict(apply_overrides(tree, overrides or []))


def config_to_dict(cfg: TrackerConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)

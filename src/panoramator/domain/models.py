from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from panoramator.projection.models import Projection

ImageArray = np.ndarray


@dataclass(slots=True)
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(slots=True)
class Frame:
    index: int
    timestamp_seconds: float
    image: ImageArray
    feature_image: ImageArray | None = None


@dataclass(slots=True)
class FrameQuality:
    sharpness: float
    difference_score: float
    accepted: bool
    reason: str


@dataclass(slots=True)
class SelectedFrame:
    frame: Frame
    quality: FrameQuality
    alternates: list[Frame] = field(default_factory=list)


@dataclass(slots=True)
class FeatureSet:
    keypoints: list
    descriptors: ImageArray | None
    backend: str


@dataclass(slots=True)
class MatchSet:
    raw_count: int
    good_matches: list
    confidence: float


@dataclass(slots=True)
class PairGeometry:
    homography: ImageArray | None
    inliers: int
    reprojection_error: float
    valid: bool
    reason: str


@dataclass(slots=True)
class CanvasModel:
    width: int
    height: int
    offset_matrix: ImageArray
    global_homographies: list[ImageArray]
    projection: Projection | None = None


@dataclass(slots=True)
class PanoramaDiagnostics:
    selected_frames: list[dict] = field(default_factory=list)
    validated_frames: list[dict] = field(default_factory=list)
    rejected_frames: list[dict] = field(default_factory=list)
    pair_metrics: list[dict] = field(default_factory=list)
    feature_backend: str = ""
    sampling_step: int = 0
    fallback_used: bool = False
    fallback_attempted: bool = False
    attempted_backends: list[str] = field(default_factory=list)
    attempted_sampling_steps: list[int] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    capture_mode: str = "linear"
    projection: str = "planar"
    strategy_confidence: float = 0.0
    strategy_reason: str = ""
    strategy_measurements: dict[str, float] = field(default_factory=dict)
    crop_policy: str = "none"
    crop_before_size: tuple[int, int] | None = None
    crop_after_size: tuple[int, int] | None = None
    crop_lost_area_fraction: float = 0.0
    trajectory: dict[str, list[float]] = field(default_factory=dict)
    seam_metrics: list[dict[str, float]] = field(default_factory=list)
    keyframe_metrics: list[dict[str, float | str]] = field(default_factory=list)
    photometric_metrics: list[dict[str, float]] = field(default_factory=list)
    status: str = "ok"


@dataclass(slots=True)
class PanoramaResult:
    image: ImageArray | None
    metadata: VideoMetadata
    diagnostics: PanoramaDiagnostics

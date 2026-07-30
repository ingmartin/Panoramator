from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np


class SurfaceKind(StrEnum):
    AUTO = "auto"
    CYLINDRICAL = "cylindrical"
    CURVED = "curved"


class UnwrapStatus(StrEnum):
    OK = "ok"
    PARTIAL_SURFACE = "partial_surface"
    OBJECT_NOT_DETECTED = "object_not_detected"
    INSUFFICIENT_TEXTURE = "insufficient_texture"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    EXCESSIVE_MOTION_BLUR = "excessive_motion_blur"
    SURFACE_MODEL_MISMATCH = "surface_model_mismatch"
    OCCLUDED_CRITICAL_AREA = "occluded_critical_area"
    UNSTABLE_CAMERA_GEOMETRY = "unstable_camera_geometry"


@dataclass(slots=True)
class SurfaceModel:
    kind: SurfaceKind
    axis_x: float = 0.5
    axis_y: float = 0.5
    radius_px: float = 0.0
    top_y: float = 0.0
    bottom_y: float = 1.0
    seam_angle_degrees: float = 180.0
    confidence: float = 0.0


@dataclass(slots=True)
class UnwrapDiagnostics:
    status: UnwrapStatus
    message: str
    recommendation: str
    surface_kind: SurfaceKind
    measurements: dict[str, float | int | str | list[float] | list[int]] = field(default_factory=dict)
    selected_frames: list[int] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["status"] = self.status.value
        result["surface_kind"] = self.surface_kind.value
        return result


@dataclass(slots=True)
class UnwrapResult:
    image: np.ndarray | None
    coverage: np.ndarray | None
    model: SurfaceModel | None
    diagnostics: UnwrapDiagnostics
    output_path: Path | None = None


@dataclass(slots=True)
class UnwrapConfig:
    surface_kind: SurfaceKind = SurfaceKind.AUTO
    allow_partial: bool = False
    sampling_step: int = 12
    max_frames: int = 48
    blur_threshold: float = 35.0
    min_object_area_ratio: float = 0.025
    min_coverage: float = 0.90
    output_height: int = 512
    output_width: int = 1536
    central_band_ratio: float = 0.55
    max_pose_residual_radians: float = 0.08
    enable_global_pose_optimization: bool = False

    def validate(self) -> None:
        self.surface_kind = SurfaceKind(self.surface_kind)
        if self.sampling_step < 1 or not 2 <= self.max_frames <= 65_535:
            raise ValueError("sampling_step must be >= 1 and max_frames must be between 2 and 65535")
        if not 0 < self.min_object_area_ratio < 1:
            raise ValueError("min_object_area_ratio must be between 0 and 1")
        if not 0 < self.min_coverage <= 1:
            raise ValueError("min_coverage must be between 0 and 1")
        if self.output_height < 32 or self.output_width < 64:
            raise ValueError("output dimensions are too small")
        if not 0.2 <= self.central_band_ratio <= 1.0:
            raise ValueError("central_band_ratio must be between 0.2 and 1.0")
        if self.max_pose_residual_radians <= 0:
            raise ValueError("max_pose_residual_radians must be > 0")

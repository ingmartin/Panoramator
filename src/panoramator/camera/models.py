from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan

from panoramator.config.models import PanoramaConfig


@dataclass(frozen=True, slots=True)
class CameraParameters:
    """Camera intrinsics required by curved panorama projections."""

    focal_length_px: float
    center_x: float
    center_y: float

    @classmethod
    def from_config(cls, config: PanoramaConfig, shape: tuple[int, int]) -> CameraParameters:
        height, width = shape
        if config.focal_length_px is not None:
            focal = config.focal_length_px
        elif config.horizontal_fov_degrees is not None:
            focal = width / (2.0 * tan(radians(config.horizontal_fov_degrees) / 2.0))
        else:
            # A conservative 60 degree estimate is preferable to an arbitrary pixel value.
            focal = width / (2.0 * tan(radians(60.0) / 2.0))
        center_x = float(width / 2.0 if config.projection_center_x is None else config.projection_center_x)
        center_y = float(height / 2.0 if config.projection_center_y is None else config.projection_center_y)
        if not 0.0 <= center_x <= width or not 0.0 <= center_y <= height:
            raise ValueError("projection center must lie within the source frame")
        return cls(
            focal_length_px=float(focal),
            center_x=center_x,
            center_y=center_y,
        )

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path


@dataclass(slots=True)
class PanoramaConfig:
    sampling_step: int = 15
    max_frames: int = 25
    downscale: float = 1.0
    feature_downscale: float = 1.0
    blur_threshold: float = 80.0
    adaptive_blur_threshold: bool = False
    adaptive_blur_percentile: float = 0.35
    enable_blur_rescue_sharpening: bool = True
    blur_rescue_sharpen_strength: float = 0.2
    blur_rescue_sharpen_sigma: float = 1.0
    frame_selection_window_size: int = 1
    min_difference: float = 8.0
    feature_backend: str = "orb"
    enable_feature_fallback: bool = True
    fallback_feature_backend: str = "sift"
    fallback_min_chain_length: int = 8
    enable_sampling_fallback: bool = True
    fallback_sampling_step: int = 8
    max_features: int = 2500
    ratio_test: float = 0.75
    min_match_count: int = 20
    min_inlier_count: int = 8
    min_inlier_ratio: float = 0.4
    motion_model: str = "affine"
    capture_mode: str = "auto"
    projection: str = "auto"
    focal_length_px: float | None = None
    horizontal_fov_degrees: float | None = None
    projection_center_x: float | None = None
    projection_center_y: float | None = None
    projection_contour_samples: int = 32
    ransac_threshold: float = 4.0
    max_reprojection_error: float = 6.0
    max_scale_deviation: float = 0.15
    max_rotation_degrees: float = 12.0
    max_homography_corner_scale: float = 2.0
    max_canvas_width: int = 12000
    max_canvas_height: int = 12000
    feather_blend_kernel: int = 21
    seam_blur_kernel: int = 1
    seam_band_width: int = 7
    enable_photometric_normalization: bool = True
    photometric_smoothing: float = 0.65
    overlap_sharpness_weight: float = 0.35
    # 24 px proved too coarse on handheld office_rotation footage: it reduced
    # stripe count but made the remaining geometric discontinuities prominent.
    rotation_min_baseline_px: float = 12.0
    rotation_min_new_coverage_ratio: float = 0.01
    photometric_gain_limit: float = 0.12
    photometric_offset_limit: float = 20.0
    crop_result: bool = True
    photo_mode: bool = False
    crop_policy: str = "auto"
    max_inscribed_crop_loss: float = 0.35
    max_inscribed_crop_width_loss: float = 0.25
    trajectory_smoothing_window: int = 5
    max_rotation_scale_correction: float = 0.02
    orbit_max_reprojection_error: float = 3.5
    orbit_min_dominant_inlier_ratio: float = 0.55
    enable_final_sharpening: bool = True
    final_sharpen_strength: float = 0.15
    final_sharpen_sigma: float = 1.0
    save_debug_artifacts: bool = True

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_json(cls, path: str | Path) -> PanoramaConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def validate(self) -> None:
        self.feature_backend = self.feature_backend.lower()
        self.fallback_feature_backend = self.fallback_feature_backend.lower()
        self.motion_model = self.motion_model.lower()
        self.capture_mode = self.capture_mode.lower()
        self.projection = self.projection.lower()
        self.crop_policy = self.crop_policy.lower()
        if self.sampling_step < 1:
            raise ValueError("sampling_step must be >= 1")
        if self.max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        if not 0 < self.downscale <= 1:
            raise ValueError("downscale must be between 0 and 1")
        if not 0 < self.feature_downscale <= 1:
            raise ValueError("feature_downscale must be between 0 and 1")
        if self.blur_threshold < 0:
            raise ValueError("blur_threshold must be >= 0")
        if not 0.0 <= self.adaptive_blur_percentile <= 1.0:
            raise ValueError("adaptive_blur_percentile must be between 0.0 and 1.0")
        if self.blur_rescue_sharpen_strength < 0:
            raise ValueError("blur_rescue_sharpen_strength must be >= 0")
        if self.blur_rescue_sharpen_sigma <= 0:
            raise ValueError("blur_rescue_sharpen_sigma must be > 0")
        if self.frame_selection_window_size < 1:
            raise ValueError("frame_selection_window_size must be >= 1")
        if self.min_difference < 0:
            raise ValueError("min_difference must be >= 0")
        if self.feature_backend not in {"orb", "sift"}:
            raise ValueError("feature_backend must be one of: orb, sift")
        if self.fallback_feature_backend not in {"orb", "sift"}:
            raise ValueError("fallback_feature_backend must be one of: orb, sift")
        if self.fallback_min_chain_length < 2:
            raise ValueError("fallback_min_chain_length must be >= 2")
        if self.fallback_sampling_step < 1:
            raise ValueError("fallback_sampling_step must be >= 1")
        if self.max_features < 1:
            raise ValueError("max_features must be >= 1")
        if not 0.0 < self.ratio_test < 1.0:
            raise ValueError("ratio_test must be between 0.0 and 1.0")
        if self.min_match_count < 1:
            raise ValueError("min_match_count must be >= 1")
        if self.min_inlier_count < 1:
            raise ValueError("min_inlier_count must be >= 1")
        if not 0.0 < self.min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be between 0.0 (exclusive) and 1.0")
        if self.motion_model not in {"translation", "partial_affine", "affine", "homography"}:
            raise ValueError("motion_model must be one of: translation, partial_affine, affine, homography")
        if self.capture_mode not in {"auto", "linear", "rotation", "orbit"}:
            raise ValueError("capture_mode must be one of: auto, linear, rotation, orbit")
        if self.projection not in {"auto", "planar", "cylindrical", "spherical"}:
            raise ValueError("projection must be one of: auto, planar, cylindrical, spherical")
        if self.crop_policy not in {"auto", "bounding", "inscribed_rectangle", "preserve_alpha"}:
            raise ValueError("crop_policy must be one of: auto, bounding, inscribed_rectangle, preserve_alpha")
        if not 0.0 <= self.max_inscribed_crop_loss < 1.0:
            raise ValueError("max_inscribed_crop_loss must be between 0.0 and 1.0")
        if not 0.0 <= self.max_inscribed_crop_width_loss < 1.0:
            raise ValueError("max_inscribed_crop_width_loss must be between 0.0 and 1.0")
        if self.trajectory_smoothing_window < 1:
            raise ValueError("trajectory_smoothing_window must be >= 1")
        if not 0.0 <= self.max_rotation_scale_correction <= 0.1:
            raise ValueError("max_rotation_scale_correction must be between 0.0 and 0.1")
        if self.orbit_max_reprojection_error <= 0:
            raise ValueError("orbit_max_reprojection_error must be > 0")
        if not 0.0 < self.orbit_min_dominant_inlier_ratio <= 1.0:
            raise ValueError("orbit_min_dominant_inlier_ratio must be between 0.0 (exclusive) and 1.0")
        if self.focal_length_px is not None and (not isfinite(self.focal_length_px) or self.focal_length_px <= 0):
            raise ValueError("focal_length_px must be > 0")
        if self.horizontal_fov_degrees is not None and (
            not isfinite(self.horizontal_fov_degrees) or not 1.0 < self.horizontal_fov_degrees < 179.0
        ):
            raise ValueError("horizontal_fov_degrees must be between 1 and 179")
        if self.focal_length_px is not None and self.horizontal_fov_degrees is not None:
            raise ValueError("Set either focal_length_px or horizontal_fov_degrees, not both")
        if self.projection_contour_samples < 4:
            raise ValueError("projection_contour_samples must be >= 4")
        for name, value in {
            "projection_center_x": self.projection_center_x,
            "projection_center_y": self.projection_center_y,
        }.items():
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.ransac_threshold <= 0:
            raise ValueError("ransac_threshold must be > 0")
        if self.max_reprojection_error <= 0:
            raise ValueError("max_reprojection_error must be > 0")
        if self.max_scale_deviation < 0:
            raise ValueError("max_scale_deviation must be >= 0")
        if self.max_rotation_degrees < 0:
            raise ValueError("max_rotation_degrees must be >= 0")
        if self.max_homography_corner_scale < 1.0:
            raise ValueError("max_homography_corner_scale must be >= 1.0")
        if self.max_canvas_width < 1 or self.max_canvas_height < 1:
            raise ValueError("max_canvas_width and max_canvas_height must be >= 1")
        if self.feather_blend_kernel < 1:
            raise ValueError("feather_blend_kernel must be >= 1")
        if self.seam_blur_kernel < 1:
            raise ValueError("seam_blur_kernel must be >= 1")
        if self.seam_band_width < 1:
            raise ValueError("seam_band_width must be >= 1")
        if not 0.0 <= self.photometric_smoothing <= 1.0:
            raise ValueError("photometric_smoothing must be between 0.0 and 1.0")
        if self.overlap_sharpness_weight < 0:
            raise ValueError("overlap_sharpness_weight must be >= 0")
        if self.rotation_min_baseline_px < 0:
            raise ValueError("rotation_min_baseline_px must be >= 0")
        if not 0.0 <= self.rotation_min_new_coverage_ratio < 1.0:
            raise ValueError("rotation_min_new_coverage_ratio must be between 0.0 and 1.0")
        if not 0.0 <= self.photometric_gain_limit <= 0.5:
            raise ValueError("photometric_gain_limit must be between 0.0 and 0.5")
        if self.photometric_offset_limit < 0:
            raise ValueError("photometric_offset_limit must be >= 0")
        if self.final_sharpen_strength < 0:
            raise ValueError("final_sharpen_strength must be >= 0")
        if self.final_sharpen_sigma <= 0:
            raise ValueError("final_sharpen_sigma must be > 0")

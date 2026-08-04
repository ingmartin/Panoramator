from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from panoramator.config.models import PanoramaConfig
from panoramator.object_unwrap import PublishProfile, SurfaceKind, UnwrapConfig

ApplyPredicate = Callable[[argparse.Namespace, str], bool]
ValueFactory = Callable[[argparse.Namespace, str], object]


@dataclass(frozen=True)
class ArgumentSpec:
    flags: tuple[str, ...]
    parser_kwargs: Mapping[str, object]
    config_attr: str | None = None
    should_apply: ApplyPredicate | None = None
    value_factory: ValueFactory | None = None

    @property
    def dest(self) -> str:
        explicit_dest = self.parser_kwargs.get("dest")
        if isinstance(explicit_dest, str):
            return explicit_dest
        for flag in reversed(self.flags):
            if flag.startswith("--"):
                return flag[2:].replace("-", "_")
        return self.flags[0].replace("-", "_")


def _flag_enabled(args: argparse.Namespace, dest: str) -> bool:
    return bool(getattr(args, dest, False))


def _argument_value_is_set(args: argparse.Namespace, dest: str) -> bool:
    return getattr(args, dest, None) is not None


def _literal_value(value: object) -> ValueFactory:
    return lambda _args, _dest: value


def _enum_value(enum_type: type[SurfaceKind] | type[PublishProfile]) -> ValueFactory:
    return lambda args, dest: enum_type(getattr(args, dest))


def add_arguments(parser: argparse.ArgumentParser, specs: Sequence[ArgumentSpec]) -> None:
    for spec in specs:
        parser.add_argument(*spec.flags, **cast(dict[str, Any], dict(spec.parser_kwargs)))


def _apply_argument_specs(config: object, args: argparse.Namespace, specs: Sequence[ArgumentSpec]) -> None:
    for spec in specs:
        if spec.config_attr is None:
            continue

        dest = spec.dest
        should_apply = spec.should_apply(args, dest) if spec.should_apply is not None else _argument_value_is_set(args, dest)
        if not should_apply:
            continue

        value = spec.value_factory(args, dest) if spec.value_factory is not None else getattr(args, dest)
        setattr(config, spec.config_attr, value)


BUILD_ARGUMENTS: tuple[ArgumentSpec, ...] = (
    ArgumentSpec(("video_path",), {"help": "Input video file"}),
    ArgumentSpec(("output_path",), {"help": "Output panorama image path"}),
    ArgumentSpec(("--config",), {"help": "Load panorama config overrides from JSON"}),
    ArgumentSpec(("--sampling-step",), {"type": int, "help": "Process every Nth frame"}, config_attr="sampling_step"),
    ArgumentSpec(("--max-frames",), {"type": int, "help": "Maximum number of selected frames to use"}, config_attr="max_frames"),
    ArgumentSpec(("--downscale",), {"type": float, "help": "Global frame downscale factor before stitching"}, config_attr="downscale"),
    ArgumentSpec(("--feature-downscale",), {"type": float, "help": "Extra downscale factor before feature extraction"}, config_attr="feature_downscale"),
    ArgumentSpec(
        ("--feature-backend",),
        {"choices": ["orb", "sift"], "help": "Primary feature detector and descriptor backend"},
        config_attr="feature_backend",
    ),
    ArgumentSpec(("--blur-threshold",), {"type": float, "help": "Reject frames blurrier than this score"}, config_attr="blur_threshold"),
    ArgumentSpec(
        ("--adaptive-blur-threshold",),
        {"action": "store_true", "help": "Estimate blur threshold from the input video"},
        config_attr="adaptive_blur_threshold",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(
        ("--blur-rescue-sharpening",),
        {"action": "store_true", "help": "Enable pre-filter sharpening for borderline blurry frames"},
        config_attr="enable_blur_rescue_sharpening",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(
        ("--no-blur-rescue-sharpening",),
        {"action": "store_true", "help": "Disable pre-filter sharpening for blurry frames"},
        config_attr="enable_blur_rescue_sharpening",
        should_apply=_flag_enabled,
        value_factory=_literal_value(False),
    ),
    ArgumentSpec(
        ("--blur-rescue-sharpen-strength",),
        {"type": float, "help": "Sharpening strength used by blur rescue"},
        config_attr="blur_rescue_sharpen_strength",
    ),
    ArgumentSpec(
        ("--blur-rescue-sharpen-sigma",),
        {"type": float, "help": "Sharpening blur radius used by blur rescue"},
        config_attr="blur_rescue_sharpen_sigma",
    ),
    ArgumentSpec(
        ("--frame-selection-window-size",),
        {"type": int, "help": "Local frame selection window size"},
        config_attr="frame_selection_window_size",
    ),
    ArgumentSpec(
        ("--motion-model",),
        {"choices": ["translation", "partial_affine", "affine", "homography"], "help": "Geometric motion model used between frames"},
        config_attr="motion_model",
    ),
    ArgumentSpec(
        ("--capture-mode",),
        {"choices": ["auto", "linear", "rotation"], "help": "Expected camera capture pattern"},
        config_attr="capture_mode",
    ),
    ArgumentSpec(
        ("--projection",),
        {"choices": ["auto", "planar", "cylindrical", "spherical"], "help": "Projection model for the output panorama"},
        config_attr="projection",
    ),
    ArgumentSpec(("--focal-length-px",), {"type": float, "help": "Explicit focal length in pixels for projection"}, config_attr="focal_length_px"),
    ArgumentSpec(
        ("--horizontal-fov-degrees",),
        {"type": float, "help": "Explicit horizontal field of view in degrees"},
        config_attr="horizontal_fov_degrees",
    ),
    ArgumentSpec(("--feather-blend-kernel",), {"type": int, "help": "Feather blending kernel size"}, config_attr="feather_blend_kernel"),
    ArgumentSpec(("--seam-blur-kernel",), {"type": int, "help": "Blur kernel size applied along seams"}, config_attr="seam_blur_kernel"),
    ArgumentSpec(("--seam-band-width",), {"type": int, "help": "Width of the seam smoothing band in pixels"}, config_attr="seam_band_width"),
    ArgumentSpec(
        ("--photometric-normalization",),
        {"action": "store_true", "help": "Enable local exposure normalization across overlaps"},
        config_attr="enable_photometric_normalization",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(
        ("--no-photometric-normalization",),
        {"action": "store_true", "help": "Disable local exposure normalization across overlaps"},
        config_attr="enable_photometric_normalization",
        should_apply=_flag_enabled,
        value_factory=_literal_value(False),
    ),
    ArgumentSpec(
        ("--global-photometric-normalization",),
        {"action": "store_true", "help": "Enable global exposure normalization across the full panorama"},
        config_attr="enable_global_photometric_normalization",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(
        ("--no-global-photometric-normalization",),
        {"action": "store_true", "help": "Disable global exposure normalization across the full panorama"},
        config_attr="enable_global_photometric_normalization",
        should_apply=_flag_enabled,
        value_factory=_literal_value(False),
    ),
    ArgumentSpec(("--photometric-smoothing",), {"type": float, "help": "Smoothing factor for photometric correction"}, config_attr="photometric_smoothing"),
    ArgumentSpec(
        ("--overlap-sharpness-weight",),
        {"type": float, "help": "Weight sharp overlaps more strongly during blending"},
        config_attr="overlap_sharpness_weight",
    ),
    ArgumentSpec(
        ("--rotation-min-baseline-px",),
        {"type": float, "help": "Minimum baseline in pixels for rotation capture acceptance"},
        config_attr="rotation_min_baseline_px",
    ),
    ArgumentSpec(
        ("--rotation-min-new-coverage-ratio",),
        {"type": float, "help": "Minimum new coverage ratio required for rotation capture"},
        config_attr="rotation_min_new_coverage_ratio",
    ),
    ArgumentSpec(("--photometric-gain-limit",), {"type": float, "help": "Maximum allowed gain correction"}, config_attr="photometric_gain_limit"),
    ArgumentSpec(
        ("--photometric-offset-limit",),
        {"type": float, "help": "Maximum allowed brightness offset correction"},
        config_attr="photometric_offset_limit",
    ),
    ArgumentSpec(
        ("--narrow-gap-fill",),
        {"action": "store_true", "help": "Enable filling narrow transparent gaps after stitching"},
        config_attr="enable_narrow_gap_fill",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(
        ("--no-narrow-gap-fill",),
        {"action": "store_true", "help": "Disable filling narrow transparent gaps after stitching"},
        config_attr="enable_narrow_gap_fill",
        should_apply=_flag_enabled,
        value_factory=_literal_value(False),
    ),
    ArgumentSpec(("--max-narrow-gap-width",), {"type": int, "help": "Maximum gap width eligible for automatic fill"}, config_attr="max_narrow_gap_width"),
    ArgumentSpec(("--photo-crop-margin-px",), {"type": int, "help": "Extra crop margin in pixels for photo mode"}, config_attr="photo_crop_margin_px"),
    ArgumentSpec(
        ("--photo-mode",),
        {"action": "store_true", "help": "Prefer photo-style cropping and post-processing"},
        config_attr="photo_mode",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(
        ("--crop-policy",),
        {"choices": ["auto", "bounding", "inscribed_rectangle", "preserve_alpha"], "help": "Crop strategy for the final panorama"},
        config_attr="crop_policy",
    ),
    ArgumentSpec(("--max-inscribed-crop-loss",), {"type": float, "help": "Maximum allowed area loss for inscribed crop"}, config_attr="max_inscribed_crop_loss"),
    ArgumentSpec(
        ("--max-inscribed-crop-width-loss",),
        {"type": float, "help": "Maximum allowed width loss for inscribed crop"},
        config_attr="max_inscribed_crop_width_loss",
    ),
    ArgumentSpec(
        ("--final-sharpening",),
        {"action": "store_true", "help": "Enable final output sharpening"},
        config_attr="enable_final_sharpening",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(
        ("--no-final-sharpening",),
        {"action": "store_true", "help": "Disable final output sharpening"},
        config_attr="enable_final_sharpening",
        should_apply=_flag_enabled,
        value_factory=_literal_value(False),
    ),
    ArgumentSpec(("--final-sharpen-strength",), {"type": float, "help": "Final output sharpening strength"}, config_attr="final_sharpen_strength"),
    ArgumentSpec(("--final-sharpen-sigma",), {"type": float, "help": "Final output sharpening blur radius"}, config_attr="final_sharpen_sigma"),
    ArgumentSpec(
        ("--feature-fallback",),
        {"action": "store_true", "help": "Allow retry with a fallback feature backend"},
        config_attr="enable_feature_fallback",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(
        ("--no-feature-fallback",),
        {"action": "store_true", "help": "Disable retry with a fallback feature backend"},
        config_attr="enable_feature_fallback",
        should_apply=_flag_enabled,
        value_factory=_literal_value(False),
    ),
    ArgumentSpec(
        ("--fallback-feature-backend",),
        {"choices": ["orb", "sift"], "help": "Feature backend used for fallback retry"},
        config_attr="fallback_feature_backend",
    ),
    ArgumentSpec(
        ("--fallback-min-chain-length",),
        {"type": int, "help": "Minimum frame chain length required to avoid backend fallback"},
        config_attr="fallback_min_chain_length",
    ),
    ArgumentSpec(
        ("--sampling-fallback",),
        {"action": "store_true", "help": "Allow retry with a denser frame sampling step"},
        config_attr="enable_sampling_fallback",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(
        ("--no-sampling-fallback",),
        {"action": "store_true", "help": "Disable retry with a denser frame sampling step"},
        config_attr="enable_sampling_fallback",
        should_apply=_flag_enabled,
        value_factory=_literal_value(False),
    ),
    ArgumentSpec(("--fallback-sampling-step",), {"type": int, "help": "Sampling step used for fallback retry"}, config_attr="fallback_sampling_step"),
    ArgumentSpec(
        ("--save-debug-artifacts",),
        {"action": "store_true", "help": "Write intermediate debug artifacts to disk"},
        config_attr="save_debug_artifacts",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(
        ("--no-save-debug-artifacts",),
        {"action": "store_true", "help": "Disable writing intermediate debug artifacts"},
        config_attr="save_debug_artifacts",
        should_apply=_flag_enabled,
        value_factory=_literal_value(False),
    ),
)

UNWRAP_ARGUMENTS: tuple[ArgumentSpec, ...] = (
    ArgumentSpec(("video_path",), {"help": "Input video file"}),
    ArgumentSpec(("output_path",), {"help": "Output surface image path"}),
    ArgumentSpec(("--config",), {"help": "Load unwrap config overrides from JSON"}),
    ArgumentSpec(
        ("--surface",),
        {"dest": "surface_kind", "choices": [kind.value for kind in SurfaceKind], "default": "auto", "help": "Target surface model to unwrap"},
        config_attr="surface_kind",
        should_apply=_argument_value_is_set,
        value_factory=_enum_value(SurfaceKind),
    ),
    ArgumentSpec(
        ("--publish-profile",),
        {"choices": [profile.value for profile in PublishProfile], "help": "Preset quality-vs-coverage tradeoff"},
        config_attr="publish_profile",
        value_factory=_enum_value(PublishProfile),
    ),
    ArgumentSpec(
        ("--allow-partial",),
        {"action": "store_true", "help": "Allow partial output when full surface coverage is not available"},
        config_attr="allow_partial",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(("--sampling-step",), {"type": int, "help": "Process every Nth frame"}, config_attr="sampling_step"),
    ArgumentSpec(("--max-frames",), {"type": int, "help": "Maximum number of selected frames to use"}, config_attr="max_frames"),
    ArgumentSpec(("--blur-threshold",), {"type": float, "help": "Reject frames blurrier than this score"}, config_attr="blur_threshold"),
    ArgumentSpec(
        ("--min-object-area-ratio",),
        {"type": float, "help": "Minimum detected object area ratio per frame"},
        config_attr="min_object_area_ratio",
    ),
    ArgumentSpec(("--min-coverage",), {"type": float, "help": "Minimum coverage required for a successful unwrap"}, config_attr="min_coverage"),
    ArgumentSpec(("--output-width",), {"type": int, "help": "Output surface width in pixels"}, config_attr="output_width"),
    ArgumentSpec(("--output-height",), {"type": int, "help": "Output surface height in pixels"}, config_attr="output_height"),
    ArgumentSpec(
        ("--crop-result",),
        {"action": "store_true", "help": "Crop empty margins from the final surface image"},
        config_attr="crop_result",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(
        ("--save-debug-artifacts",),
        {"action": "store_true", "help": "Write intermediate debug artifacts to disk"},
        config_attr="save_debug_artifacts",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(
        ("--no-save-debug-artifacts",),
        {"action": "store_true", "help": "Disable writing intermediate debug artifacts"},
        config_attr="save_debug_artifacts",
        should_apply=_flag_enabled,
        value_factory=_literal_value(False),
    ),
    ArgumentSpec(
        ("--photo-mode",),
        {"action": "store_true", "help": "Prefer photo-style cropping and output cleanup"},
        config_attr="photo_mode",
        should_apply=_flag_enabled,
        value_factory=_literal_value(True),
    ),
    ArgumentSpec(("--photo-crop-margin-px",), {"type": int, "help": "Extra crop margin in pixels for photo mode"}, config_attr="photo_crop_margin_px"),
    ArgumentSpec(
        ("--photo-crop-max-loss",),
        {"type": float, "help": "Maximum allowed area loss during photo-mode cropping"},
        config_attr="photo_crop_max_loss",
    ),
    ArgumentSpec(
        ("--photo-crop-max-width-loss",),
        {"type": float, "help": "Maximum allowed width loss during photo-mode cropping"},
        config_attr="photo_crop_max_width_loss",
    ),
    ArgumentSpec(("--central-band-ratio",), {"type": float, "help": "Relative width of the central analysis band"}, config_attr="central_band_ratio"),
    ArgumentSpec(
        ("--max-pose-residual-radians",),
        {"type": float, "help": "Maximum allowed pose residual in radians"},
        config_attr="max_pose_residual_radians",
    ),
    ArgumentSpec(
        ("--min-accepted-pose-pair-fraction",),
        {"type": float, "help": "Minimum fraction of accepted frame pairs for pose solving"},
        config_attr="min_accepted_pose_pair_fraction",
    ),
    ArgumentSpec(
        ("--max-mosaic-boundary-mean-error",),
        {"type": float, "help": "Maximum mean boundary alignment error in pixels"},
        config_attr="max_mosaic_boundary_mean_error",
    ),
    ArgumentSpec(
        ("--max-mosaic-boundary-severe-fraction",),
        {"type": float, "help": "Maximum fraction of boundary marked as severely misaligned"},
        config_attr="max_mosaic_boundary_severe_fraction",
    ),
    ArgumentSpec(
        ("--mosaic-boundary-severe-error",),
        {"type": float, "help": "Severe mosaic seam error threshold in pixels"},
        config_attr="mosaic_boundary_severe_error",
    ),
    ArgumentSpec(
        ("--max-mosaic-boundary-severe-footprint",),
        {"type": float, "help": "Maximum severe seam footprint before failing the unwrap"},
        config_attr="max_mosaic_boundary_severe_footprint",
    ),
    ArgumentSpec(
        ("--no-temporal-decimation",),
        {"action": "store_true", "help": "Keep all candidate frames instead of pruning similar ones"},
        config_attr="enable_temporal_decimation",
        should_apply=_flag_enabled,
        value_factory=_literal_value(False),
    ),
    ArgumentSpec(
        ("--temporal-decimation-max-mask-iou",),
        {"type": float, "help": "Maximum mask IoU before two frames are treated as redundant"},
        config_attr="temporal_decimation_max_mask_iou",
    ),
    ArgumentSpec(
        ("--temporal-decimation-min-band-difference",),
        {"type": float, "help": "Minimum band difference required to keep a new frame"},
        config_attr="temporal_decimation_min_band_difference",
    ),
    ArgumentSpec(
        ("--temporal-decimation-min-bbox-shift",),
        {"type": float, "help": "Minimum bounding-box shift required to keep a new frame"},
        config_attr="temporal_decimation_min_bbox_shift",
    ),
    ArgumentSpec(
        ("--min-rectification-column-fraction",),
        {"type": float, "help": "Minimum usable rectified column fraction"},
        config_attr="min_rectification_column_fraction",
    ),
    ArgumentSpec(
        ("--rectification-smoothing-window",),
        {"type": int, "help": "Smoothing window size for rectification"},
        config_attr="rectification_smoothing_window",
    ),
    ArgumentSpec(
        ("--max-rectification-axis-step",),
        {"type": float, "help": "Maximum per-step change along the rectification axis"},
        config_attr="max_rectification_axis_step",
    ),
    ArgumentSpec(
        ("--no-global-pose-optimization",),
        {"action": "store_true", "help": "Disable the experimental multi-frame pose gate"},
        config_attr="enable_global_pose_optimization",
        should_apply=_flag_enabled,
        value_factory=_literal_value(False),
    ),
)


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    add_arguments(parser, BUILD_ARGUMENTS)


def add_unwrap_arguments(parser: argparse.ArgumentParser) -> None:
    add_arguments(parser, UNWRAP_ARGUMENTS)


def apply_build_overrides(config: PanoramaConfig, args: argparse.Namespace) -> None:
    _apply_argument_specs(config, args, BUILD_ARGUMENTS)


def apply_unwrap_overrides(config: UnwrapConfig, args: argparse.Namespace) -> None:
    _apply_argument_specs(config, args, UNWRAP_ARGUMENTS)

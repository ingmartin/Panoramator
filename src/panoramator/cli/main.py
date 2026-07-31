from __future__ import annotations

import argparse
from pathlib import Path

from panoramator.application.use_cases import PanoramaBuilder
from panoramator.config.models import PanoramaConfig
from panoramator.object_unwrap import (
    ObjectUnwrapper,
    SurfaceKind,
    UnwrapConfig,
    UnwrapStatus,
)


def build_command(args: argparse.Namespace) -> int:
    config = PanoramaConfig()
    if args.config:
        config = PanoramaConfig.from_json(args.config)

    if args.sampling_step is not None:
        config.sampling_step = args.sampling_step
    if args.max_frames is not None:
        config.max_frames = args.max_frames
    if args.downscale is not None:
        config.downscale = args.downscale
    if args.feature_downscale is not None:
        config.feature_downscale = args.feature_downscale
    if args.feature_backend is not None:
        config.feature_backend = args.feature_backend
    if args.blur_threshold is not None:
        config.blur_threshold = args.blur_threshold
    if args.adaptive_blur_threshold:
        config.adaptive_blur_threshold = True
    if args.blur_rescue_sharpening:
        config.enable_blur_rescue_sharpening = True
    if args.no_blur_rescue_sharpening:
        config.enable_blur_rescue_sharpening = False
    if args.blur_rescue_sharpen_strength is not None:
        config.blur_rescue_sharpen_strength = args.blur_rescue_sharpen_strength
    if args.blur_rescue_sharpen_sigma is not None:
        config.blur_rescue_sharpen_sigma = args.blur_rescue_sharpen_sigma
    if args.frame_selection_window_size is not None:
        config.frame_selection_window_size = args.frame_selection_window_size
    if args.motion_model is not None:
        config.motion_model = args.motion_model
    if getattr(args, "capture_mode", None) is not None:
        config.capture_mode = args.capture_mode
    if getattr(args, "projection", None) is not None:
        config.projection = args.projection
    if getattr(args, "focal_length_px", None) is not None:
        config.focal_length_px = args.focal_length_px
    if getattr(args, "horizontal_fov_degrees", None) is not None:
        config.horizontal_fov_degrees = args.horizontal_fov_degrees
    if args.feather_blend_kernel is not None:
        config.feather_blend_kernel = args.feather_blend_kernel
    if args.seam_blur_kernel is not None:
        config.seam_blur_kernel = args.seam_blur_kernel
    if args.seam_band_width is not None:
        config.seam_band_width = args.seam_band_width
    if args.photometric_normalization:
        config.enable_photometric_normalization = True
    if args.no_photometric_normalization:
        config.enable_photometric_normalization = False
    if getattr(args, "global_photometric_normalization", False):
        config.enable_global_photometric_normalization = True
    if getattr(args, "no_global_photometric_normalization", False):
        config.enable_global_photometric_normalization = False
    if args.photometric_smoothing is not None:
        config.photometric_smoothing = args.photometric_smoothing
    if args.overlap_sharpness_weight is not None:
        config.overlap_sharpness_weight = args.overlap_sharpness_weight
    if getattr(args, "rotation_min_baseline_px", None) is not None:
        config.rotation_min_baseline_px = args.rotation_min_baseline_px
    if getattr(args, "rotation_min_new_coverage_ratio", None) is not None:
        config.rotation_min_new_coverage_ratio = args.rotation_min_new_coverage_ratio
    if getattr(args, "photometric_gain_limit", None) is not None:
        config.photometric_gain_limit = args.photometric_gain_limit
    if getattr(args, "photometric_offset_limit", None) is not None:
        config.photometric_offset_limit = args.photometric_offset_limit
    if getattr(args, "narrow_gap_fill", False):
        config.enable_narrow_gap_fill = True
    if getattr(args, "no_narrow_gap_fill", False):
        config.enable_narrow_gap_fill = False
    if getattr(args, "max_narrow_gap_width", None) is not None:
        config.max_narrow_gap_width = args.max_narrow_gap_width
    if getattr(args, "photo_crop_margin_px", None) is not None:
        config.photo_crop_margin_px = args.photo_crop_margin_px
    if args.photo_mode:
        config.photo_mode = True
    if getattr(args, "crop_policy", None) is not None:
        config.crop_policy = args.crop_policy
    if getattr(args, "max_inscribed_crop_loss", None) is not None:
        config.max_inscribed_crop_loss = args.max_inscribed_crop_loss
    if getattr(args, "max_inscribed_crop_width_loss", None) is not None:
        config.max_inscribed_crop_width_loss = args.max_inscribed_crop_width_loss
    if args.final_sharpening:
        config.enable_final_sharpening = True
    if args.no_final_sharpening:
        config.enable_final_sharpening = False
    if args.final_sharpen_strength is not None:
        config.final_sharpen_strength = args.final_sharpen_strength
    if args.final_sharpen_sigma is not None:
        config.final_sharpen_sigma = args.final_sharpen_sigma
    if args.feature_fallback:
        config.enable_feature_fallback = True
    if args.no_feature_fallback:
        config.enable_feature_fallback = False
    if args.fallback_feature_backend is not None:
        config.fallback_feature_backend = args.fallback_feature_backend
    if args.fallback_min_chain_length is not None:
        config.fallback_min_chain_length = args.fallback_min_chain_length
    if args.sampling_fallback:
        config.enable_sampling_fallback = True
    if args.no_sampling_fallback:
        config.enable_sampling_fallback = False
    if args.fallback_sampling_step is not None:
        config.fallback_sampling_step = args.fallback_sampling_step
    if getattr(args, "save_debug_artifacts", False):
        config.save_debug_artifacts = True
    if getattr(args, "no_save_debug_artifacts", False):
        config.save_debug_artifacts = False

    config.validate()

    result = PanoramaBuilder(config).build_from_video(args.video_path, args.output_path)
    if result.diagnostics.status != "orbit_not_supported_reliably":
        print(f"Panorama saved to: {args.output_path}")
    else:
        print("Panorama was not written: capture mode is not supported reliably")
    print(f"Selected frames: {len(result.diagnostics.selected_frames)}")
    print(f"Rejected frames: {len(result.diagnostics.rejected_frames)}")
    print(f"Feature backend: {result.diagnostics.feature_backend}")
    print(f"Sampling step: {result.diagnostics.sampling_step}")
    print(f"Fallback used: {result.diagnostics.fallback_used}")
    print(f"Capture mode: {result.diagnostics.capture_mode}")
    print(f"Projection: {result.diagnostics.projection}")
    print(f"Status: {result.diagnostics.status}")
    print(f"Video FPS: {result.metadata.fps}")
    return 0


def inspect_video_command(args: argparse.Namespace) -> int:
    from panoramator.io.video import OpenCVVideoSource

    source = OpenCVVideoSource(args.video_path, PanoramaConfig())
    metadata = source.open()
    source.close()
    print(f"path={metadata.path}")
    print(f"fps={metadata.fps}")
    print(f"frame_count={metadata.frame_count}")
    print(f"width={metadata.width}")
    print(f"height={metadata.height}")
    return 0


def unwrap_command(args: argparse.Namespace) -> int:
    config = UnwrapConfig()
    if getattr(args, "config", None):
        config = UnwrapConfig.from_json(args.config)

    if getattr(args, "surface_kind", None) is not None:
        config.surface_kind = SurfaceKind(args.surface_kind)
    if getattr(args, "allow_partial", False):
        config.allow_partial = True
    if getattr(args, "sampling_step", None) is not None:
        config.sampling_step = args.sampling_step
    if getattr(args, "max_frames", None) is not None:
        config.max_frames = args.max_frames
    if getattr(args, "blur_threshold", None) is not None:
        config.blur_threshold = args.blur_threshold
    if getattr(args, "min_object_area_ratio", None) is not None:
        config.min_object_area_ratio = args.min_object_area_ratio
    if getattr(args, "min_coverage", None) is not None:
        config.min_coverage = args.min_coverage
    if getattr(args, "output_width", None) is not None:
        config.output_width = args.output_width
    if getattr(args, "output_height", None) is not None:
        config.output_height = args.output_height
    if getattr(args, "crop_result", False):
        config.crop_result = True
    if getattr(args, "photo_mode", False):
        config.photo_mode = True
    if getattr(args, "photo_crop_margin_px", None) is not None:
        config.photo_crop_margin_px = args.photo_crop_margin_px
    if getattr(args, "photo_crop_max_loss", None) is not None:
        config.photo_crop_max_loss = args.photo_crop_max_loss
    if getattr(args, "photo_crop_max_width_loss", None) is not None:
        config.photo_crop_max_width_loss = args.photo_crop_max_width_loss
    if getattr(args, "save_debug_artifacts", False):
        config.save_debug_artifacts = True
    if getattr(args, "no_save_debug_artifacts", False):
        config.save_debug_artifacts = False
    if getattr(args, "central_band_ratio", None) is not None:
        config.central_band_ratio = args.central_band_ratio
    if getattr(args, "max_pose_residual_radians", None) is not None:
        config.max_pose_residual_radians = args.max_pose_residual_radians
    if getattr(args, "min_accepted_pose_pair_fraction", None) is not None:
        config.min_accepted_pose_pair_fraction = args.min_accepted_pose_pair_fraction
    if getattr(args, "max_mosaic_boundary_mean_error", None) is not None:
        config.max_mosaic_boundary_mean_error = args.max_mosaic_boundary_mean_error
    if getattr(args, "max_mosaic_boundary_severe_fraction", None) is not None:
        config.max_mosaic_boundary_severe_fraction = args.max_mosaic_boundary_severe_fraction
    if getattr(args, "mosaic_boundary_severe_error", None) is not None:
        config.mosaic_boundary_severe_error = args.mosaic_boundary_severe_error
    if getattr(args, "max_mosaic_boundary_severe_footprint", None) is not None:
        config.max_mosaic_boundary_severe_footprint = args.max_mosaic_boundary_severe_footprint
    if getattr(args, "no_temporal_decimation", False):
        config.enable_temporal_decimation = False
    if getattr(args, "temporal_decimation_max_mask_iou", None) is not None:
        config.temporal_decimation_max_mask_iou = args.temporal_decimation_max_mask_iou
    if getattr(args, "temporal_decimation_min_band_difference", None) is not None:
        config.temporal_decimation_min_band_difference = args.temporal_decimation_min_band_difference
    if getattr(args, "temporal_decimation_min_bbox_shift", None) is not None:
        config.temporal_decimation_min_bbox_shift = args.temporal_decimation_min_bbox_shift
    if getattr(args, "min_rectification_column_fraction", None) is not None:
        config.min_rectification_column_fraction = args.min_rectification_column_fraction
    if getattr(args, "rectification_smoothing_window", None) is not None:
        config.rectification_smoothing_window = args.rectification_smoothing_window
    if getattr(args, "max_rectification_axis_step", None) is not None:
        config.max_rectification_axis_step = args.max_rectification_axis_step
    if getattr(args, "no_global_pose_optimization", False):
        config.enable_global_pose_optimization = False
    config.validate()
    result = ObjectUnwrapper(config).unwrap_video(args.video_path, args.output_path)
    print(f"Status: {result.diagnostics.status.value}")
    print(f"Surface: {result.diagnostics.surface_kind.value}")
    if result.output_path is not None:
        print(f"Unwrap saved to: {result.output_path}")
    print(f"Selected frames: {len(result.diagnostics.selected_frames)}")
    print(f"Sampling step: {config.sampling_step}")
    print(result.diagnostics.message)
    if result.diagnostics.recommendation:
        print(f"Recommendation: {result.diagnostics.recommendation}")
    return 0 if result.output_path is not None and result.diagnostics.status in {UnwrapStatus.OK, UnwrapStatus.PARTIAL_SURFACE} else 2


def export_config_command(args: argparse.Namespace) -> int:
    PanoramaConfig().save(args.output_path)
    print(f"Config saved to: {args.output_path}")
    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="panoramator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build panorama from video")
    build.add_argument("video_path")
    build.add_argument("output_path")
    build.add_argument("--config")
    build.add_argument("--sampling-step", type=int)
    build.add_argument("--max-frames", type=int)
    build.add_argument("--downscale", type=float)
    build.add_argument("--feature-downscale", type=float)
    build.add_argument("--feature-backend", choices=["orb", "sift"])
    build.add_argument("--blur-threshold", type=float)
    build.add_argument("--adaptive-blur-threshold", action="store_true")
    build.add_argument("--blur-rescue-sharpening", action="store_true")
    build.add_argument("--no-blur-rescue-sharpening", action="store_true")
    build.add_argument("--blur-rescue-sharpen-strength", type=float)
    build.add_argument("--blur-rescue-sharpen-sigma", type=float)
    build.add_argument("--frame-selection-window-size", type=int)
    build.add_argument("--motion-model", choices=["translation", "partial_affine", "affine", "homography"])
    build.add_argument("--capture-mode", choices=["auto", "linear", "rotation", "orbit"])
    build.add_argument("--projection", choices=["auto", "planar", "cylindrical", "spherical"])
    build.add_argument("--focal-length-px", type=float)
    build.add_argument("--horizontal-fov-degrees", type=float)
    build.add_argument("--feather-blend-kernel", type=int)
    build.add_argument("--seam-blur-kernel", type=int)
    build.add_argument("--seam-band-width", type=int)
    build.add_argument("--photometric-normalization", action="store_true")
    build.add_argument("--no-photometric-normalization", action="store_true")
    build.add_argument("--global-photometric-normalization", action="store_true")
    build.add_argument("--no-global-photometric-normalization", action="store_true")
    build.add_argument("--photometric-smoothing", type=float)
    build.add_argument("--overlap-sharpness-weight", type=float)
    build.add_argument("--rotation-min-baseline-px", type=float)
    build.add_argument("--rotation-min-new-coverage-ratio", type=float)
    build.add_argument("--photometric-gain-limit", type=float)
    build.add_argument("--photometric-offset-limit", type=float)
    build.add_argument("--narrow-gap-fill", action="store_true")
    build.add_argument("--no-narrow-gap-fill", action="store_true")
    build.add_argument("--max-narrow-gap-width", type=int)
    build.add_argument("--photo-crop-margin-px", type=int)
    build.add_argument("--photo-mode", action="store_true")
    build.add_argument("--crop-policy", choices=["auto", "bounding", "inscribed_rectangle", "preserve_alpha"])
    build.add_argument("--max-inscribed-crop-loss", type=float)
    build.add_argument("--max-inscribed-crop-width-loss", type=float)
    build.add_argument("--final-sharpening", action="store_true")
    build.add_argument("--no-final-sharpening", action="store_true")
    build.add_argument("--final-sharpen-strength", type=float)
    build.add_argument("--final-sharpen-sigma", type=float)
    build.add_argument("--feature-fallback", action="store_true")
    build.add_argument("--no-feature-fallback", action="store_true")
    build.add_argument("--fallback-feature-backend", choices=["orb", "sift"])
    build.add_argument("--fallback-min-chain-length", type=int)
    build.add_argument("--sampling-fallback", action="store_true")
    build.add_argument("--no-sampling-fallback", action="store_true")
    build.add_argument("--fallback-sampling-step", type=int)
    build.add_argument("--save-debug-artifacts", action="store_true")
    build.add_argument("--no-save-debug-artifacts", action="store_true")
    build.set_defaults(func=build_command)

    unwrap = subparsers.add_parser("unwrap", help="Build a surface map from video")
    unwrap.add_argument("video_path")
    unwrap.add_argument("output_path")
    unwrap.add_argument("--config")
    unwrap.add_argument("--surface", dest="surface_kind", choices=[kind.value for kind in SurfaceKind], default="auto")
    unwrap.add_argument("--allow-partial", action="store_true")
    unwrap.add_argument("--sampling-step", type=int)
    unwrap.add_argument("--max-frames", type=int)
    unwrap.add_argument("--blur-threshold", type=float)
    unwrap.add_argument("--min-object-area-ratio", type=float)
    unwrap.add_argument("--min-coverage", type=float)
    unwrap.add_argument("--output-width", type=int)
    unwrap.add_argument("--output-height", type=int)
    unwrap.add_argument("--crop-result", action="store_true")
    unwrap.add_argument("--save-debug-artifacts", action="store_true")
    unwrap.add_argument("--no-save-debug-artifacts", action="store_true")
    unwrap.add_argument("--photo-mode", action="store_true")
    unwrap.add_argument("--photo-crop-margin-px", type=int)
    unwrap.add_argument("--photo-crop-max-loss", type=float)
    unwrap.add_argument("--photo-crop-max-width-loss", type=float)
    unwrap.add_argument("--central-band-ratio", type=float)
    unwrap.add_argument("--max-pose-residual-radians", type=float)
    unwrap.add_argument("--min-accepted-pose-pair-fraction", type=float)
    unwrap.add_argument("--max-mosaic-boundary-mean-error", type=float)
    unwrap.add_argument("--max-mosaic-boundary-severe-fraction", type=float)
    unwrap.add_argument("--mosaic-boundary-severe-error", type=float)
    unwrap.add_argument("--max-mosaic-boundary-severe-footprint", type=float)
    unwrap.add_argument("--no-temporal-decimation", action="store_true")
    unwrap.add_argument("--temporal-decimation-max-mask-iou", type=float)
    unwrap.add_argument("--temporal-decimation-min-band-difference", type=float)
    unwrap.add_argument("--temporal-decimation-min-bbox-shift", type=float)
    unwrap.add_argument("--min-rectification-column-fraction", type=float)
    unwrap.add_argument("--rectification-smoothing-window", type=int)
    unwrap.add_argument("--max-rectification-axis-step", type=float)
    unwrap.add_argument(
        "--no-global-pose-optimization",
        action="store_true",
        help="Disable the conservative multi-frame pose gate (experimental; prevents status ok)",
    )
    unwrap.set_defaults(func=unwrap_command)

    inspect_video = subparsers.add_parser("inspect-video", help="Inspect video metadata")
    inspect_video.add_argument("video_path")
    inspect_video.set_defaults(func=inspect_video_command)

    export_config = subparsers.add_parser("export-config", help="Export default config")
    export_config.add_argument("output_path", nargs="?", default=str(Path("panoramator.config.json")))
    export_config.set_defaults(func=export_config_command)
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

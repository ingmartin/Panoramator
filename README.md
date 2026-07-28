# panoramator

Python package for building panoramas from video with an extensible architecture.

Capture modes are independent from projection: `--capture-mode` accepts `auto`, `linear`,
`rotation`, or `orbit`; `--projection` accepts `auto`, `planar`, `cylindrical`, or
`spherical`. Ambiguous automatic input preserves the compatible `linear + planar` pipeline.
Use `--capture-mode rotation` for a camera rotating in place; it selects a cylindrical
surface. `--focal-length-px` or `--horizontal-fov-degrees` optionally refine the camera model.

## Status

Published on PyPI and installable with pip.

## License

This repository is published under the MIT license. See `LICENSE`.

## Current Features

* video input via OpenCV;
* key frame selection by frame step, sharpness, and simple visual difference;
* slight conditional frame sharpening when a frame narrowly misses the sharpness threshold;
* optional reduced-resolution feature extraction while keeping full-resolution warping and blending;
* smarter local frame selection by sharpest candidate in a window;
* ORB or SIFT feature extraction;
* feature matching and geometry estimation with `translation`, `partial_affine`, `affine`, or `homography`;
* global canvas construction;
* feather blending with additional seam smoothing, photometric normalization, and detail-aware overlap weighting;
* automatic black border cropping;
* optional `photo-mode` crop of a planar panorama to the fully visible rectangle without internal black wedges;
* final mild panorama sharpening;
* `ORB -> SIFT` fallback when the valid frame chain is too short;
* denser `sampling_step` fallback when a second pass is needed;
* CLI for execution and diagnostics.

The default motion model is `affine`. For many video panorama cases, `partial_affine` is worth trying when you need tighter control over deformation between adjacent frames.
By default, `blur_threshold` is fixed. The `--adaptive-blur-threshold` option enables an adaptive mode where the effective threshold is reduced according to the sharpness distribution of sampled frames from the current video.

## Installation

```bash
python -m pip install panoramator
```

## Using as a Python Package

You can use the module from another Python application as a regular package. After installation, import the main classes and run panorama generation:

```python
from pathlib import Path

from panoramator import PanoramaBuilder, PanoramaConfig

config = PanoramaConfig(
    sampling_step=15,
    max_frames=25,
    motion_model="partial_affine",
    crop_result=True,
)

builder = PanoramaBuilder(config)
result = builder.build_from_video(
    video_path=Path("VID_20260709_140742.mp4"),
    output_path=Path("output.png"),
)

print(result.metadata)
```

## Development Setup

```bash
python -m pip install -e ".[dev]"
```

## Tests

```bash
pytest -q
```

## Configuration Parameters

All parameters can be set through `PanoramaConfig`, a JSON config file, or partially through the CLI.

### Input Frames

* `sampling_step` - step between frame indices during the initial video scan. Lower values mean more frames and a better chance of finding good pairs, but slower processing. Default: `15`.
* `max_frames` - maximum number of sampled frames passed into the rest of the pipeline. Default: `25`.
* `downscale` - scale factor applied to frames before processing and final warping. `1.0` means original resolution. Default: `1.0`.
* `feature_downscale` - extra scale factor used only for feature extraction and matching. Helps keep warping full-resolution while making geometry estimation cheaper. Default: `1.0`.

### Frame Selection

* `blur_threshold` - fixed sharpness threshold. Frames below it are treated as too blurry. Default: `80.0`.
* `adaptive_blur_threshold` - enables adaptive relaxation of `blur_threshold` using the sharpness distribution of the current video. Default: `False`.
* `adaptive_blur_percentile` - quantile used to compute the adaptive blur threshold. Only used when `adaptive_blur_threshold` is enabled. Default: `0.35`.
* `enable_blur_rescue_sharpening` - tries a slight unsharp-mask pass before rejecting a soft frame by blur threshold. Default: `True`.
* `blur_rescue_sharpen_strength` - strength of the rescue sharpening pass. Higher values are more aggressive. Default: `0.2`.
* `blur_rescue_sharpen_sigma` - Gaussian sigma used by the rescue sharpening pass. Default: `1.0`.
* `frame_selection_window_size` - selects the sharpest valid frame inside each local window instead of taking every valid candidate. Default: `1`.
* `min_difference` - minimum difference from the previously selected frame. Helps avoid nearly identical frames. Default: `8.0`.

### Features and Fallback

* `feature_backend` - primary feature backend. Supported values are `orb` and `sift`. Default: `orb`.
* `enable_feature_fallback` - enables automatic `ORB -> SIFT` fallback when the valid chain is too short. Default: `True`.
* `fallback_feature_backend` - backend used during feature fallback. Default: `sift`.
* `fallback_min_chain_length` - minimum valid chain length below which feature fallback starts. Default: `8`.
* `max_features` - maximum number of keypoints requested from the detector. Default: `2500`.
* `ratio_test` - Lowe ratio test coefficient used when filtering matches. Lower values apply stricter filtering. Default: `0.75`.
* `min_match_count` - minimum number of good matches required before geometry estimation is attempted. Default: `20`.
* `min_inlier_count` - minimum number of RANSAC-consistent matches required to accept a pair. Default: `8`.
* `min_inlier_ratio` - minimum fraction of good matches that must be RANSAC inliers. Default: `0.4`.

### Sampling Fallback

* `enable_sampling_fallback` - allows a second pass with denser frame sampling when the first pass produces a weak chain. Default: `True`.
* `fallback_sampling_step` - alternative denser frame step for the fallback pass. It usually makes sense to keep it smaller than the primary `sampling_step`. Default: `8`.

### Geometry

* `motion_model` - motion model between adjacent frames. Supported values are `translation`, `partial_affine`, `affine`, and `homography`. Default: `affine`.
* `ransac_threshold` - reprojection threshold used inside RANSAC during transform estimation. Default: `4.0`.
* `max_reprojection_error` - maximum mean reprojection error accepted for a valid pair. Default: `6.0`.
* `max_scale_deviation` - maximum allowed scale deviation from `1.0` in the estimated transform. Helps reject implausible matches. Default: `0.15`.
* `max_rotation_degrees` - maximum allowed rotation between neighboring frames. Default: `12.0`.
* `max_homography_corner_scale` - maximum allowed projected width or height of one frame relative to its source size in `homography` mode. Default: `2.0`.

### Canvas and Stitching

* `max_canvas_width` - hard limit for final canvas width. Helps prevent excessive memory use. Default: `12000`.
* `max_canvas_height` - hard limit for final canvas height. Helps prevent excessive memory use. Default: `12000`.

### Blending and Seams

* `feather_blend_kernel` - width of the weight smoothing zone near warped frame borders. Default: `21`.
* `seam_blur_kernel` - strength of local blur along seam areas. Default: `1` (disabled) to avoid softening details at frame joins.
* `seam_band_width` - width of the band around seam boundaries where local smoothing is allowed. Default: `7`.
* `enable_photometric_normalization` - matches brightness and contrast between neighboring selected frames before warping. Default: `True`.
* `photometric_smoothing` - how strongly neighboring frames are normalized toward each other. Default: `0.65`.
* `overlap_sharpness_weight` - how much blending should favor locally sharper content in overlap areas. Default: `0.35`.

### Postprocessing and Artifacts

* `crop_result` - enables automatic cropping of black borders after stitching. Default: `True`.
* `photo_mode` - crops a planar panorama more aggressively to the largest rectangle fully inside the visible area. Cylindrical and spherical panoramas safely fall back to outer-boundary cropping because a strict rectangle can discard most of the result. Default: `False`.
* `enable_final_sharpening` - applies a final mild unsharp-mask pass to the completed panorama. Default: `True`.
* `final_sharpen_strength` - strength of the final panorama sharpening pass. Default: `0.15`.
* `final_sharpen_sigma` - Gaussian sigma used by the final panorama sharpening pass. Default: `1.0`.
* `save_debug_artifacts` - saves a debug directory with the effective config and run report. Default: `True`.

### Full Config Example

```json
{
  "sampling_step": 15,
  "max_frames": 25,
  "downscale": 1.0,
  "feature_downscale": 1.0,
  "blur_threshold": 80.0,
  "adaptive_blur_threshold": false,
  "adaptive_blur_percentile": 0.35,
  "enable_blur_rescue_sharpening": true,
  "blur_rescue_sharpen_strength": 0.2,
  "blur_rescue_sharpen_sigma": 1.0,
  "frame_selection_window_size": 1,
  "min_difference": 8.0,
  "feature_backend": "orb",
  "enable_feature_fallback": true,
  "fallback_feature_backend": "sift",
  "fallback_min_chain_length": 8,
  "enable_sampling_fallback": true,
  "fallback_sampling_step": 8,
  "max_features": 2500,
  "ratio_test": 0.75,
  "min_match_count": 20,
  "min_inlier_count": 8,
  "min_inlier_ratio": 0.4,
  "motion_model": "affine",
  "ransac_threshold": 4.0,
  "max_reprojection_error": 6.0,
  "max_scale_deviation": 0.15,
  "max_rotation_degrees": 12.0,
  "max_homography_corner_scale": 2.0,
  "max_canvas_width": 12000,
  "max_canvas_height": 12000,
  "feather_blend_kernel": 21,
  "seam_blur_kernel": 1,
  "seam_band_width": 7,
  "enable_photometric_normalization": true,
  "photometric_smoothing": 0.65,
  "overlap_sharpness_weight": 0.35,
  "crop_result": true,
  "photo_mode": false,
  "enable_final_sharpening": true,
  "final_sharpen_strength": 0.15,
  "final_sharpen_sigma": 1.0,
  "save_debug_artifacts": true
}
```

## Quick Start

```bash
panoramator build VID_20260709_140742.mp4 output.png
```

For videos where the fixed blur threshold is too strict:

```bash
panoramator build VID_20260709_140742.mp4 output.png --adaptive-blur-threshold
```

If frames are just slightly too soft, keep the adaptive threshold and tune the rescue sharpening instead of lowering the blur threshold too much:

```bash
panoramator build VID_20260709_140742.mp4 output.png --adaptive-blur-threshold --blur-rescue-sharpen-strength 0.25 --blur-rescue-sharpen-sigma 1.0
```

If you want a quality-oriented compromise between speed and full-resolution output, reduce only feature resolution and enable windowed frame selection:

```bash
panoramator build VID_20260709_140742.mp4 output.png --feature-downscale 0.5 --frame-selection-window-size 3
```

If seam lines are visible in the panorama, you can tune feather width and very local seam blur separately:

```bash
panoramator build VID_20260709_140742.mp4 output.png --seam-blur-kernel 7 --seam-band-width 9 --feather-blend-kernel 25
```

If you need a photo-like planar frame without any black corners after warping, enable `photo-mode`:

```bash
panoramator build VID_20260709_140742.mp4 output.png --photo-mode
```

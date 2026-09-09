<p align="center">
  <img src="https://github.com/ingmartin/Panoramator/raw/main/assets/logo.svg" width="180" alt="Panoramator">
</p>

<h1 align="center">Panoramator</h1>

<p align="center">
Python package for building panoramas from video and unwrapping the observed surface of a rotating object.
</p>

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
![Code style](https://img.shields.io/badge/code%20style-ruff-purple.svg)
![PyPI](https://img.shields.io/pypi/v/panoramator.svg)

</div>

<p align="center">
OpenCV powered • Scene panoramas and object unwrap • Python CLI
</p>

## What It Does

`panoramator` covers two related, but different tasks:

| Task | Command | Use when |
| --- | --- | --- |
| Scene panorama | `build` | Camera moves along a scene or rotates in place |
| Observed-surface unwrap | `unwrap` | Camera orbits one object and you want its visible surface as a flat map |

If the goal is a wide scene, use `build`. If the goal is to inspect the outside of one object, use `unwrap`.

## Quick Start

### Install

```bash
pip install panoramator
```

### Build a panorama

```bash
panoramator build video.mp4 output.png
```

### Unwrap an object surface

```bash
panoramator unwrap video.mp4 surface.png --surface cylindrical
```

## Typical Workflows

### 1. Handheld sweep across a scene

```bash
panoramator build input.mp4 output.png
```

### 2. Camera rotates in place

```bash
panoramator build input.mp4 output.png --capture-mode rotation --horizontal-fov-degrees 70
```

### 3. Orbit around one object

```bash
panoramator unwrap input.mp4 surface.png --surface cylindrical
```

### 4. Clean crop for presentation-ready output

```bash
panoramator build input.mp4 output.png --photo-mode --photo-crop-margin-px 5
```

## Examples And Visuals

Temporary demo assets already live in [`docs/public-demo`](docs/public-demo). These animated previews can be replaced later with final real captures:

| Scenario | Input | Result |
| --- | --- | --- |
| Linear scene panorama<br>camera translates across the scene | [<img src="docs/public-demo/build-linear-input.gif" alt="Linear scene input animation" width="220">](docs/public-demo/build-linear-input.mp4) | [<img src="docs/public-demo/build-linear-reference.png" alt="Linear scene panorama reference result" width="220">](docs/public-demo/build-linear-reference.png) |
| Rotation panorama<br>camera rotates in place around one viewpoint | [<img src="docs/public-demo/build-rotation-input.gif" alt="Rotation input animation" width="220">](docs/public-demo/build-rotation-input.mp4) | [<img src="docs/public-demo/build-rotation-reference.png" alt="Rotation panorama reference result" width="220">](docs/public-demo/build-rotation-reference.png) |
| Object unwrap<br>camera orbits one object to flatten its visible surface | [<img src="docs/public-demo/unwrap-cylinder-input.gif" alt="Object orbit input animation" width="220">](docs/public-demo/unwrap-cylinder-input.mp4) | [<img src="docs/public-demo/unwrap-cylinder-reference.png" alt="Object unwrap reference result" width="220">](docs/public-demo/unwrap-cylinder-reference.png) |

When videos are ready, the README will work best if each scenario shows:

1. a short input clip or GIF;
2. the resulting panorama or surface map;
3. one sentence explaining why this mode was chosen.

## Practical Recipes

### Soft footage

Use adaptive blur filtering before lowering thresholds manually:

```bash
panoramator build input.mp4 output.png --adaptive-blur-threshold
```

If frames are only slightly soft, keep the adaptive threshold and tune rescue sharpening:

```bash
panoramator build input.mp4 output.png \
  --adaptive-blur-threshold \
  --blur-rescue-sharpen-strength 0.25 \
  --blur-rescue-sharpen-sigma 1.0
```

### Faster feature extraction with full-resolution output

```bash
panoramator build input.mp4 output.png --feature-downscale 0.5 --frame-selection-window-size 3
```

### Visible seams

```bash
panoramator build input.mp4 output.png --seam-blur-kernel 7 --seam-band-width 9 --feather-blend-kernel 25
```

### Cleaner unwrap crop

```bash
panoramator unwrap input.mp4 surface.png \
  --surface cylindrical \
  --photo-mode \
  --photo-crop-margin-px 5
```

## Output And Debug Artifacts

Both commands can save a debug directory with the effective config and diagnostics.

For `unwrap`, the `*_debug` directory typically includes:

* `run.json`
* `effective_config.json`
* coverage maps
* source and error maps
* intermediate mosaics

Disable debug output when you only need the final image:

```bash
panoramator build input.mp4 output.png --no-save-debug-artifacts
```

## Current Features

* video input via OpenCV;
* scene panorama building for `linear` and `rotation` capture;
* observed-surface `unwrap` pipeline for orbital object capture;
* ORB or SIFT feature extraction with geometry validation and fallback sampling;
* planar, cylindrical, and spherical projection surfaces with optional camera calibration;
* photometric normalization, seam handling, crop policies, and sharpening;
* debug artifacts and local private acceptance workflows.

## Important Concepts

### Capture mode and projection are different

`--capture-mode` describes how the camera moved:

* `auto`
* `linear`
* `rotation`

`--projection` describes how the result is represented:

* `auto`
* `planar`
* `cylindrical`
* `spherical`

For a camera rotating in place, use `--capture-mode rotation`. For orbit around one object, do not use `build`; use `unwrap`.

### First settings to try

Most users do not need the full parameter list first. In practice, these options matter most:

* `--capture-mode rotation` for in-place camera rotation;
* `--horizontal-fov-degrees` or `--focal-length-px` for curved projection calibration;
* `--adaptive-blur-threshold` for soft footage;
* `--feature-downscale` to reduce feature cost without shrinking final output;
* `--photo-mode` for presentation-friendly crops;
* `--no-save-debug-artifacts` when diagnostics are not needed.

## Using As A Python Package

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
    video_path=Path("input.mp4"),
    output_path=Path("output.png"),
)

print(result.metadata)
print(result.diagnostics.status)
print(result.diagnostics.output_files)
```

## Installation For Development

```bash
python -m pip install -e ".[dev]"
```

## Verifying Cylindrical Unwrap

Use an explicit surface kind when checking a cylindrical object. The command below keeps the normal quality gates enabled and writes the diagnostics needed to inspect the result:

```bash
panoramator unwrap input.mp4 output/cylinder.png \
  --surface cylindrical \
  --publish-profile balanced_publish \
  --sampling-step 12 \
  --max-frames 60 \
  --output-width 1536 \
  --output-height 512 \
  --interpolate-gaps \
  --max-interpolation-gap-px 96 \
  --save-debug-artifacts
```

For a fast pipeline smoke test, reduce the workload and explicitly allow a partial image. A sparse sample such as `--sampling-step 24 --max-frames 12` is intentionally not comparable with a full orbit: it may return `unstable_camera_geometry`, and `--allow-partial` does not override that geometry safety gate. To guarantee a diagnostic image from a sparse run, use `--surface-output-mode observed_surface`.

```bash
panoramator unwrap input.mp4 output/cylinder-smoke.png \
  --surface cylindrical \
  --sampling-step 24 \
  --max-frames 12 \
  --output-width 768 \
  --output-height 256 \
  --surface-output-mode observed_surface \
  --allow-partial \
  --save-debug-artifacts
```

The cylindrical renderer uses the mask as confidence for source-column selection and geometry checks, not as a hard foreground cutout. Texture details inside a selected surface band—such as ears, outlines, and bright printed areas—are therefore preserved. Short internal gaps are filled by interpolation between observed pixels; large external frame regions are still not invented. The `curved` path remains separate and does not inherit this publication policy.

If transparent stripes still interfere with visual inspection, enable bounded gap interpolation:

```bash
panoramator unwrap input.mp4 output/cylinder-filled.png \
  --surface cylindrical \
  --surface-output-mode observed_surface \
  --interpolate-gaps \
  --max-interpolation-gap-px 96 \
  --allow-partial
```

Interpolation is applied only between two observed regions in the same row or column. Outer margins and gaps wider than `--max-interpolation-gap-px` remain transparent.

For a real cylindrical acceptance run, do not add `--allow-partial`. Inspect `output/cylinder_debug/run.json` and check that `status` is `ok`, `surface_kind` is `cylindrical`, and `measurements.rendering` and `measurements.primary_renderer` are both `inverse_cylindrical_atlas`. Also check `coverage_fraction`, `pose_residual_radians`, and the accepted/rejected pose-pair counts. The image-space `mosaic` and `angular_mosaic` files are diagnostics; the saved surface image is produced by the primary cylindrical renderer.

Run the curved regression separately on a known curved-object clip:

```bash
panoramator unwrap curved-input.mp4 output/curved.png \
  --surface curved \
  --publish-profile balanced_publish \
  --save-debug-artifacts
```

## Tests

```bash
python -m pytest -q
python -m pytest -q --cov=panoramator --cov-report=term-missing
```

Private acceptance fixtures are intentionally separate from the default reproducible suite:

```bash
PANORAMATOR_RUN_PRIVATE_VIDEO=1 python3 -m pytest tests/test_private_orbit_acceptance.py -q
PANORAMATOR_RUN_PRIVATE_VIDEO=1 python3 -m pytest tests/test_private_panorama_acceptance.py -q
```

## Documentation

* Showcase: [`docs/showcase.md`](docs/showcase.md)
* Roadmap: [`docs/roadmap.md`](docs/roadmap.md)
* Russian README: [`README.ru.md`](README.ru.md)

## Full Configuration

The CLI and `PanoramaConfig` expose many tuning parameters for frame sampling, geometry, blending, cropping, sharpening, and diagnostics.

For day-to-day use, start with the recipes above and inspect:

```bash
panoramator build --help
panoramator unwrap --help
```

If needed, parameters can be set through:

* `PanoramaConfig`
* a JSON config file
* CLI flags and targeted overrides

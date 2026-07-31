from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.io.video import OpenCVVideoSource
from panoramator.postprocess.crop import crop_with_policy

from .analyzer import VideoAnalyzer
from .coverage import coverage_fraction
from .curved.builder import CurvedSurfaceFallbackBuilder
from .cylinder.builder import CylinderUnwrapBuilder
from .diagnostics import write_artifacts
from .models import (
    SurfaceKind,
    UnwrapConfig,
    UnwrapDiagnostics,
    UnwrapResult,
    UnwrapStatus,
)


class ObjectUnwrapper:
    """Build an honest object-surface texture from an already recorded video."""

    def __init__(self, config: UnwrapConfig | None = None) -> None:
        self.config = config or UnwrapConfig()
        self.config.validate()

    def unwrap_video(self, video_path: str | Path, output_path: str | Path) -> UnwrapResult:
        output = Path(output_path)
        # Surface coordinates are rendered at a fixed atlas resolution; loading
        # every high-resolution video frame only exhausts memory and prevents
        # dense temporal tracking on ordinary machines.
        frame_config = replace(
            PanoramaConfig(), sampling_step=self.config.sampling_step, max_frames=self.config.max_frames, downscale=0.5
        )
        source = OpenCVVideoSource(video_path, frame_config)
        try:
            source.open()
            frames = source.iter_frames()
        finally:
            source.close()
        analysis = VideoAnalyzer().analyze(frames, self.config)
        if analysis.status is not None:
            return self._failure(output, analysis.status, analysis.message, analysis.recommendation, analysis.kind)

        if analysis.kind is SurfaceKind.CYLINDRICAL:
            image, coverage, model, measurements, artifacts = CylinderUnwrapBuilder().build(analysis.frames, self.config)
            fallback = False
        else:
            image, coverage, model, measurements, artifacts = CurvedSurfaceFallbackBuilder().build(analysis.frames, self.config)
            fallback = True
        surface_coverage = measurements.get("surface_coverage_fraction", coverage_fraction(coverage))
        fraction = float(surface_coverage) if isinstance(surface_coverage, (int, float)) else coverage_fraction(coverage)
        measurements["surface_coverage_fraction"] = fraction
        measurements["frame_count"] = len(analysis.frames)
        selected = [
            {"frame_index": item.frame.index, "timestamp_seconds": item.frame.timestamp_seconds}
            for item in analysis.frames
        ]
        pose_residual = measurements.get("pose_residual_radians")
        accepted_pairs = measurements.get("accepted_pose_pairs")
        required_pairs = max(2, int(np.ceil((len(analysis.frames) - 1) * self.config.min_accepted_pose_pair_fraction)))
        geometry_rejected = (
            self.config.enable_global_pose_optimization
            and (
                not isinstance(pose_residual, (int, float))
                or not np.isfinite(pose_residual)
                or pose_residual > self.config.max_pose_residual_radians
                or not isinstance(accepted_pairs, int)
                or accepted_pairs < required_pairs
                or measurements.get("repeated_observation_detected") == 1
            )
        )
        planar_mosaic = artifacts.get("mosaic")
        planar_coverage = artifacts.get("mosaic_coverage")
        has_planar_fallback = isinstance(planar_mosaic, np.ndarray) and isinstance(planar_coverage, np.ndarray)
        quality_gate_passed = measurements.get("quality_gate_passed") == 1
        rectification_applied = measurements.get("rectification_applied") == 1
        if not self.config.enable_global_pose_optimization:
            status = UnwrapStatus.PARTIAL_SURFACE
            message = "The experimental renderer was used without a global pose quality gate."
            recommendation = "Enable global pose optimization before accepting a complete surface map."
        elif geometry_rejected:
            if rectification_applied:
                status = UnwrapStatus.PARTIAL_SURFACE
                message = "A rectified observed band is available, but the cylindrical trajectory is not independently confirmed."
                recommendation = "Use this partial rectified band, or record a slower closed orbit for a cylindrical atlas."
            elif has_planar_fallback:
                image, coverage = cast(np.ndarray, planar_mosaic), cast(np.ndarray, planar_coverage)
                status = UnwrapStatus.PARTIAL_SURFACE
                message = "A connected image-space mosaic is available, but the cylindrical surface trajectory is not confirmed."
                recommendation = "Use this partial observed band, or record a slower closed orbit for a cylindrical atlas."
            else:
                status = UnwrapStatus.UNSTABLE_CAMERA_GEOMETRY
                message = "The tracked views do not agree on one stable surface trajectory."
                recommendation = "Record a slower orbit with more overlap and less camera shake."
        elif not quality_gate_passed:
            status = UnwrapStatus.PARTIAL_SURFACE
            message = "The baseline mosaic was published without rectification because some source boundaries remain unstable."
            recommendation = "Use this observed band, or record a slower orbit with stronger overlap for cleaner rectification."
        elif fallback:
            status = UnwrapStatus.PARTIAL_SURFACE
            message = "Only the observed side band is available; a mesh UV reconstruction is not sufficiently supported by the video."
            recommendation = "Record the surface from more viewpoints with overlapping frames."
        else:
            status = UnwrapStatus.PARTIAL_SURFACE
            if rectification_applied:
                message = "A partial observed surface band was assembled and rectified from the baseline mosaic."
            else:
                message = "A partial observed surface band was assembled, but the mosaic did not support one global rectification."
            recommendation = "Use this observed band, or record a slower full orbit for future cylindrical confirmation."
        diagnostics = UnwrapDiagnostics(
            status,
            message,
            recommendation,
            analysis.kind,
            measurements,
            selected_frames=selected,
            validated_frames=selected,
            output_files=[],
            sampling_step=self.config.sampling_step,
            max_frames=self.config.max_frames,
            allow_partial=self.config.allow_partial,
        )
        if status in {UnwrapStatus.INSUFFICIENT_COVERAGE, UnwrapStatus.UNSTABLE_CAMERA_GEOMETRY} or (
            status is UnwrapStatus.PARTIAL_SURFACE and not self.config.allow_partial
        ):
            if self.config.save_debug_artifacts:
                write_artifacts(output, self.config, diagnostics, coverage, artifacts)
            return UnwrapResult(None, coverage, model, diagnostics)
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = coverage
        if self.config.photo_mode:
            bgra, _crop_policy, _crop_loss = crop_with_policy(
                bgra,
                coverage,
                "inscribed_rectangle",
                max_inscribed_loss=1.0,
                max_inscribed_width_loss=1.0,
                force_inscribed=True,
                inscribed_margin=self.config.photo_crop_margin_px,
            )
            coverage = bgra[:, :, 3].copy()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() not in {".png", ".webp", ".tiff"}:
            raise ValueError("object unwrap output must support alpha: PNG, WebP, or TIFF")
        if not cv2.imwrite(str(output), bgra):
            raise RuntimeError(f"Failed to write unwrap image: {output}")
        diagnostics.output_files = [str(output)]
        if self.config.save_debug_artifacts:
            diagnostics.output_files.extend(write_artifacts(output, self.config, diagnostics, coverage, artifacts))
        return UnwrapResult(bgra, coverage, model, diagnostics, output)

    def _failure(self, output: Path, status: UnwrapStatus, message: str, recommendation: str, kind: SurfaceKind) -> UnwrapResult:
        diagnostics = UnwrapDiagnostics(
            status,
            message,
            recommendation,
            kind,
            sampling_step=self.config.sampling_step,
            max_frames=self.config.max_frames,
            allow_partial=self.config.allow_partial,
        )
        if self.config.save_debug_artifacts:
            write_artifacts(output, self.config, diagnostics, None)
        return UnwrapResult(None, None, None, diagnostics)

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.io.video import OpenCVVideoSource
from panoramator.postprocess.crop import crop_with_policy

from .analyzer import Analysis, VideoAnalyzer
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
    SurfaceModel,
)


@dataclass(slots=True)
class _SurfaceBuild:
    image: np.ndarray
    coverage: np.ndarray
    model: SurfaceModel
    measurements: dict[str, float | int | str | list[float] | list[int]]
    artifacts: dict[str, object]
    fallback_used: bool


@dataclass(slots=True)
class _PublicationDecision:
    status: UnwrapStatus
    message: str
    recommendation: str
    image: np.ndarray
    coverage: np.ndarray


class ObjectUnwrapper:
    """Build an honest object-surface texture from an already recorded video."""

    def __init__(self, config: UnwrapConfig | None = None) -> None:
        self.config = config or UnwrapConfig()
        self.config.validate()

    def unwrap_video(self, video_path: str | Path, output_path: str | Path) -> UnwrapResult:
        output = Path(output_path)
        analysis = self._analyze_video(video_path)
        if analysis.status is not None:
            return self._failure(output, analysis.status, analysis.message, analysis.recommendation, analysis.kind)

        build = self._build_surface(analysis)
        decision = self._decide_publication(analysis, build)
        selected = [
            {"frame_index": item.frame.index, "timestamp_seconds": item.frame.timestamp_seconds}
            for item in analysis.frames
        ]
        diagnostics = UnwrapDiagnostics(
            decision.status,
            decision.message,
            decision.recommendation,
            analysis.kind,
            build.measurements,
            selected_frames=selected,
            validated_frames=selected,
            rejected_frames=list(analysis.rejected_frames or []),
            output_files=[],
            sampling_step=self.config.sampling_step,
            max_frames=self.config.max_frames,
            allow_partial=self.config.allow_partial,
        )
        if self._blocks_publication(decision.status):
            if self.config.save_debug_artifacts:
                write_artifacts(output, self.config, diagnostics, decision.coverage, build.artifacts)
            return UnwrapResult(None, decision.coverage, build.model, diagnostics)
        bgra, coverage = self._render_publishable_surface(decision, build.measurements)
        self._write_publishable_surface(output, bgra)
        diagnostics.output_files = [str(output)]
        if self.config.save_debug_artifacts:
            diagnostics.output_files.extend(write_artifacts(output, self.config, diagnostics, coverage, build.artifacts))
        return UnwrapResult(bgra, coverage, build.model, diagnostics, output)

    def _analyze_video(self, video_path: str | Path) -> Analysis:
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
        return VideoAnalyzer().analyze(frames, self.config)

    def _build_surface(self, analysis: Analysis) -> _SurfaceBuild:
        if analysis.kind is SurfaceKind.CYLINDRICAL:
            image, coverage, model, measurements, artifacts = CylinderUnwrapBuilder().build(analysis.frames, self.config)
            fallback_used = False
        else:
            image, coverage, model, measurements, artifacts = CurvedSurfaceFallbackBuilder().build(analysis.frames, self.config)
            fallback_used = True
        surface_coverage = measurements.get("surface_coverage_fraction", coverage_fraction(coverage))
        fraction = float(surface_coverage) if isinstance(surface_coverage, (int, float)) else coverage_fraction(coverage)
        measurements["surface_coverage_fraction"] = fraction
        measurements.update(analysis.measurements or {})
        measurements["frame_count"] = len(analysis.frames)
        return _SurfaceBuild(image, coverage, model, measurements, artifacts, fallback_used)

    def _decide_publication(self, analysis: Analysis, build: _SurfaceBuild) -> _PublicationDecision:
        pose_residual = build.measurements.get("pose_residual_radians")
        accepted_pairs = build.measurements.get("accepted_pose_pairs")
        required_pairs = max(2, int(np.ceil((len(analysis.frames) - 1) * self.config.min_accepted_pose_pair_fraction)))
        geometry_rejected = (
            self.config.enable_global_pose_optimization
            and (
                not isinstance(pose_residual, (int, float))
                or not np.isfinite(pose_residual)
                or pose_residual > self.config.max_pose_residual_radians
                or not isinstance(accepted_pairs, int)
                or accepted_pairs < required_pairs
                or build.measurements.get("repeated_observation_detected") == 1
            )
        )
        planar_mosaic = build.artifacts.get("mosaic")
        planar_coverage = build.artifacts.get("mosaic_coverage")
        has_planar_fallback = isinstance(planar_mosaic, np.ndarray) and isinstance(planar_coverage, np.ndarray)
        quality_gate_passed = build.measurements.get("quality_gate_passed") == 1
        rectification_applied = build.measurements.get("rectification_applied") == 1

        if not self.config.enable_global_pose_optimization:
            return _PublicationDecision(
                UnwrapStatus.PARTIAL_SURFACE,
                "The experimental renderer was used without a global pose quality gate.",
                "Enable global pose optimization before accepting a complete surface map.",
                build.image,
                build.coverage,
            )
        if geometry_rejected:
            if rectification_applied:
                return _PublicationDecision(
                    UnwrapStatus.PARTIAL_SURFACE,
                    "A rectified observed band is available, but the cylindrical trajectory is not independently confirmed.",
                    "Use this partial rectified band, or record a slower closed orbit for a cylindrical atlas.",
                    build.image,
                    build.coverage,
                )
            if has_planar_fallback:
                return _PublicationDecision(
                    UnwrapStatus.PARTIAL_SURFACE,
                    "A connected image-space mosaic is available, but the cylindrical surface trajectory is not confirmed.",
                    "Use this partial observed band, or record a slower closed orbit for a cylindrical atlas.",
                    cast(np.ndarray, planar_mosaic),
                    cast(np.ndarray, planar_coverage),
                )
            return _PublicationDecision(
                UnwrapStatus.UNSTABLE_CAMERA_GEOMETRY,
                "The tracked views do not agree on one stable surface trajectory.",
                "Record a slower orbit with more overlap and less camera shake.",
                build.image,
                build.coverage,
            )
        if not quality_gate_passed:
            return _PublicationDecision(
                UnwrapStatus.PARTIAL_SURFACE,
                "The baseline mosaic was published without rectification because some source boundaries remain unstable.",
                "Use this observed band, or record a slower orbit with stronger overlap for cleaner rectification.",
                build.image,
                build.coverage,
            )
        if build.fallback_used:
            return _PublicationDecision(
                UnwrapStatus.PARTIAL_SURFACE,
                "Only the observed side band is available; a mesh UV reconstruction is not sufficiently supported by the video.",
                "Record the surface from more viewpoints with overlapping frames.",
                build.image,
                build.coverage,
            )
        if rectification_applied:
            message = "A partial observed surface band was assembled and rectified from the baseline mosaic."
        else:
            message = "A partial observed surface band was assembled, but the mosaic did not support one global rectification."
        return _PublicationDecision(
            UnwrapStatus.PARTIAL_SURFACE,
            message,
            "Use this observed band, or record a slower full orbit for future cylindrical confirmation.",
            build.image,
            build.coverage,
        )

    def _blocks_publication(self, status: UnwrapStatus) -> bool:
        return status in {UnwrapStatus.INSUFFICIENT_COVERAGE, UnwrapStatus.UNSTABLE_CAMERA_GEOMETRY} or (
            status is UnwrapStatus.PARTIAL_SURFACE and not self.config.allow_partial
        )

    def _render_publishable_surface(
        self,
        decision: _PublicationDecision,
        measurements: dict[str, float | int | str | list[float] | list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        bgra = cv2.cvtColor(decision.image, cv2.COLOR_BGR2BGRA)
        coverage = decision.coverage.copy()
        bgra[:, :, 3] = coverage
        bgra, coverage = self._apply_photo_mode(decision.status, bgra, coverage, measurements)
        bgra, coverage = self._apply_result_crop(bgra, coverage, measurements)
        return bgra, coverage

    def _apply_photo_mode(
        self,
        status: UnwrapStatus,
        bgra: np.ndarray,
        coverage: np.ndarray,
        measurements: dict[str, float | int | str | list[float] | list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.config.photo_mode:
            return bgra, coverage
        # Presentation cleanup is allowed for any published partial result.
        # The safety boundary is the crop-loss policy, not the upstream
        # geometry status: users may still want a cleaner alpha on a
        # baseline mosaic or a planar fallback without pretending the
        # geometry became better than it is.
        photo_mode_eligible = status in {UnwrapStatus.OK, UnwrapStatus.PARTIAL_SURFACE}
        measurements["photo_mode_eligible"] = int(photo_mode_eligible)
        if not photo_mode_eligible:
            measurements["photo_mode_applied"] = 0
            measurements["photo_mode_crop_policy"] = "skipped_ineligible"
            measurements["photo_mode_crop_loss"] = 0.0
            return bgra, coverage
        cropped, crop_policy, crop_loss = crop_with_policy(
            bgra,
            coverage,
            "inscribed_rectangle",
            max_inscribed_loss=self.config.photo_crop_max_loss,
            max_inscribed_width_loss=self.config.photo_crop_max_width_loss,
            force_inscribed=False,
            inscribed_margin=self.config.photo_crop_margin_px,
        )
        if crop_policy == "inscribed_rectangle":
            bgra = cropped
            coverage = bgra[:, :, 3].copy()
            measurements["photo_mode_applied"] = 1
        else:
            measurements["photo_mode_applied"] = 0
        measurements["photo_mode_crop_policy"] = crop_policy
        measurements["photo_mode_crop_loss"] = float(crop_loss)
        return bgra, coverage

    def _apply_result_crop(
        self,
        bgra: np.ndarray,
        coverage: np.ndarray,
        measurements: dict[str, float | int | str | list[float] | list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.config.crop_result:
            return bgra, coverage
        bgra, crop_policy, crop_loss = crop_with_policy(
            bgra,
            coverage,
            "preserve_alpha",
            max_inscribed_loss=1.0,
            max_inscribed_width_loss=1.0,
        )
        coverage = bgra[:, :, 3].copy()
        measurements["crop_result_applied"] = 1
        measurements["crop_result_policy"] = crop_policy
        measurements["crop_result_loss"] = float(crop_loss)
        return bgra, coverage

    def _write_publishable_surface(self, output: Path, bgra: np.ndarray) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() not in {".png", ".webp", ".tiff"}:
            raise ValueError("object unwrap output must support alpha: PNG, WebP, or TIFF")
        if not cv2.imwrite(str(output), bgra):
            raise RuntimeError(f"Failed to write unwrap image: {output}")

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

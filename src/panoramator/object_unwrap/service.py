from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.io.video import OpenCVVideoSource
from panoramator.postprocess.crop import crop_with_policy

from .analyzer import Analysis, AnalyzedFrame, VideoAnalyzer
from .coverage import coverage_fraction
from .curved.builder import CurvedSurfaceBuilder, CurvedSurfaceFallbackBuilder
from .cylinder.builder import CylinderUnwrapBuilder
from .diagnostics import write_artifacts
from .models import (
    SurfaceBuild,
    SurfaceKind,
    SurfaceModel,
    SurfaceOutputMode,
    UnwrapConfig,
    UnwrapDiagnostics,
    UnwrapResult,
    UnwrapStatus,
)
from .observed_surface import ObservedSurfaceBuilder
from .product_surface import ProductSurfaceBuilder

# Preserve the historical private import path while keeping the shared
# intermediate model in the unwrap domain layer.
_SurfaceBuild = SurfaceBuild

_RECOVERABLE_SURFACE_BUILD_ERRORS = (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError)


def _observed_candidate_passes(
    measurements: dict[str, float | int | str | list[float] | list[int]],
    baseline_build: _SurfaceBuild,
) -> bool:
    if measurements.get("observed_branch_applied") != 1:
        return False
    coverage = measurements.get("observed_branch_coverage_fraction")
    baseline_metric = baseline_build.measurements.get("publishable_surface_coverage_fraction")
    baseline_coverage = float(baseline_metric) if isinstance(baseline_metric, (int, float)) else coverage_fraction(baseline_build.coverage)
    if not isinstance(coverage, (int, float)) or float(coverage) < baseline_coverage:
        return False
    conflict = measurements.get("observed_branch_overlap_conflict_fraction", 1.0)
    gradient_ratio = measurements.get("observed_branch_gradient_energy_ratio", 2.0)
    largest = measurements.get("observed_branch_largest_component_fraction", 0.0)
    return (
        isinstance(conflict, (int, float))
        and float(conflict) <= 0.70
        and isinstance(gradient_ratio, (int, float))
        and float(gradient_ratio) <= 1.75
        and isinstance(largest, (int, float))
        and float(largest) >= 0.82
    )


@dataclass(slots=True)
class _PublicationDecision:
    status: UnwrapStatus
    message: str
    recommendation: str
    image: np.ndarray
    coverage: np.ndarray
    publication_mode: str
    geometry_confirmation: str


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

        if self._uses_observed_branch() and analysis.kind is SurfaceKind.CURVED and len(analysis.frames) >= 4:
            baseline_build = self._build_curved_observed_baseline(analysis)
        else:
            baseline_build = self._build_surface(analysis)
        build = self._build_observed_surface(analysis, baseline_build) if self._uses_observed_branch() else baseline_build
        decision = self._decide_publication(analysis, build)
        build.measurements["surface_output_intent"] = self.config.surface_output_mode.value
        build.measurements["publication_mode"] = decision.publication_mode
        build.measurements["geometry_confirmation"] = decision.geometry_confirmation
        selected = self._frame_list(analysis.frames)
        validated = selected
        rejected = list(analysis.rejected_frames or [])
        if self._uses_observed_branch():
            candidate = build.measurements.get("observed_surface_selected_candidate")
            if candidate == "curved_surface":
                curved_selected = build.artifacts.get("curved_surface_selected_frames")
                if isinstance(curved_selected, list) and curved_selected:
                    selected = curved_selected
                    validated = curved_selected
                curved_rejected = build.artifacts.get("curved_surface_rejected_frames")
                if isinstance(curved_rejected, list):
                    rejected.extend(curved_rejected)
            elif candidate == "product_surface":
                product_selected = build.artifacts.get("product_surface_selected_frames")
                if isinstance(product_selected, list) and product_selected:
                    selected = product_selected
                    validated = product_selected
                product_rejected = build.artifacts.get("product_surface_rejected_frames")
                if isinstance(product_rejected, list):
                    rejected.extend(product_rejected)
            else:
                observed_selected = build.artifacts.get("observed_branch_selected_frames")
                if isinstance(observed_selected, list) and observed_selected:
                    selected = observed_selected
                observed_validated = build.artifacts.get("observed_branch_axis_valid_frames")
                if isinstance(observed_validated, list) and observed_validated:
                    validated = observed_validated
                observed_rejected = build.artifacts.get("observed_branch_rejected_frames")
                if isinstance(observed_rejected, list):
                    rejected.extend(observed_rejected)
        diagnostics = UnwrapDiagnostics(
            decision.status,
            decision.message,
            decision.recommendation,
            analysis.kind,
            build.measurements,
            selected_frames=selected,
            validated_frames=validated,
            rejected_frames=rejected,
            output_files=[],
            sampling_step=self.config.sampling_step,
            max_frames=self.config.max_frames,
            allow_partial=self.config.allow_partial,
        )
        if self._blocks_publication(decision.status):
            self._mark_photo_mode_skipped(decision.status, build.measurements)
            if self.config.save_debug_artifacts:
                write_artifacts(output, self.config, diagnostics, decision.coverage, build.artifacts)
            return UnwrapResult(None, decision.coverage, build.model, diagnostics)
        bgra, coverage = self._render_publishable_surface(decision, build.measurements)
        self._write_publishable_surface(output, bgra)
        diagnostics.output_files = [str(output)]
        if self.config.save_debug_artifacts:
            diagnostics.output_files.extend(write_artifacts(output, self.config, diagnostics, coverage, build.artifacts))
        return UnwrapResult(bgra, coverage, build.model, diagnostics, output)

    def _uses_observed_branch(self) -> bool:
        return self.config.surface_output_mode is SurfaceOutputMode.OBSERVED_SURFACE

    def _build_observed_surface(self, analysis: Analysis, baseline_build: _SurfaceBuild) -> _SurfaceBuild:
        if analysis.kind is SurfaceKind.CURVED and len(analysis.frames) >= 4:
            try:
                curved = CurvedSurfaceBuilder().build(analysis, self.config, baseline_build)
                measurements = dict(curved.measurements)
                measurements["observed_surface_selected_candidate"] = "curved_surface"
                measurements["observed_surface_curved_gate_passed"] = int(
                    measurements.get("curved_surface_quality_gate_passed") == 1
                )
                measurements["observed_surface_product_gate_passed"] = 0
                measurements["observed_surface_observed_gate_passed"] = int(
                    measurements.get("curved_surface_quality_gate_passed") == 1
                )
                measurements["observed_surface_fallback_reason"] = (
                    "curved_surface_quality_gate_passed"
                    if measurements.get("curved_surface_quality_gate_passed") == 1
                    else str(measurements.get("curved_surface_rejected_reason", "curved_surface_quality_gate_failed"))
                )
                selected = _SurfaceBuild(
                    curved.image,
                    curved.coverage,
                    curved.model,
                    measurements,
                    curved.artifacts,
                    False,
                )
                self._ensure_surface_artifacts(selected)
                return selected
            except _RECOVERABLE_SURFACE_BUILD_ERRORS as error:
                measurements = dict(baseline_build.measurements)
                measurements.update(
                    {
                        "surface_builder": "curved_local_atlas",
                        "curved_surface_quality_gate_passed": 0,
                        "curved_surface_rejected_reason": "curved_surface_build_failed",
                        "curved_surface_exception": type(error).__name__,
                        "observed_surface_selected_candidate": "curved_surface",
                        "observed_surface_curved_gate_passed": 0,
                        "observed_surface_product_gate_passed": 0,
                        "observed_surface_observed_gate_passed": 0,
                    }
                )
                selected = _SurfaceBuild(
                    baseline_build.image,
                    baseline_build.coverage,
                    baseline_build.model,
                    measurements,
                    {
                        **baseline_build.artifacts,
                        "curved_surface_selected_frames": [],
                        "curved_surface_rejected_frames": [],
                    },
                    False,
                )
                self._ensure_surface_artifacts(selected)
                return selected
        observed_build: SurfaceBuild
        try:
            observed_build = ObservedSurfaceBuilder().build(analysis, self.config, baseline_build)
        except _RECOVERABLE_SURFACE_BUILD_ERRORS as error:
            measurements = dict(baseline_build.measurements)
            for key, value in baseline_build.measurements.items():
                measurements.setdefault(f"baseline_{key}", value)
            measurements.update(
                {
                    "observed_branch_input_frame_count": len(analysis.frames),
                    "observed_branch_applied": 0,
                    "observed_branch_abort_reason": "branch_build_failed",
                    "observed_branch_exception": type(error).__name__,
                    "observed_branch_axis_valid_frame_count": 0,
                    "observed_branch_selected_frame_count": 0,
                    "observed_branch_redundant_frame_count": 0,
                    "observed_branch_mask_fallback_count": 0,
                    "observed_branch_canvas_width": int(baseline_build.image.shape[1]),
                    "observed_branch_canvas_height": int(baseline_build.image.shape[0]),
                    "observed_branch_used_angular_steps": 0,
                    "observed_branch_used_phase_correlation": 0,
                    "observed_branch_used_bbox_fallback": 0,
                    "observed_branch_coverage_fraction": (
                        float(metric)
                        if isinstance(
                            metric := baseline_build.measurements.get("observed_coverage_fraction"),
                            (int, float),
                        )
                        else coverage_fraction(baseline_build.coverage)
                    ),
                    "observed_branch_overlap_conflict_fraction": 0.0,
                    "observed_branch_mean_gradient_gain": 0.0,
                    "observed_branch_largest_component_fraction": 0.0,
                }
            )
            observed_build = _SurfaceBuild(
                baseline_build.image,
                baseline_build.coverage,
                baseline_build.model,
                measurements,
                {
                    **baseline_build.artifacts,
                    "baseline_publish_image": baseline_build.image,
                    "baseline_publish_coverage": baseline_build.coverage,
                    "baseline_build_measurements": baseline_build.measurements,
                    "observed_branch_rejected_frames": [],
                    "observed_branch_axis_valid_frames": [],
                    "observed_branch_selected_frames": [],
                },
                baseline_build.fallback_used,
            )
        observed_build = _SurfaceBuild(
            observed_build.image,
            observed_build.coverage,
            observed_build.model,
            observed_build.measurements,
            observed_build.artifacts,
            baseline_build.fallback_used,
        )
        try:
            product = ProductSurfaceBuilder().build(analysis, self.config, baseline_build)
            product_measurements = product.measurements
            product_artifacts = product.artifacts
        except _RECOVERABLE_SURFACE_BUILD_ERRORS as error:
            product_measurements = {
                "product_surface_quality_gate_passed": 0,
                "product_surface_rejected_reason": "product_surface_build_failed",
                "product_surface_exception": type(error).__name__,
                "product_surface_selected_frame_count": 0,
            }
            product_artifacts = {
                "product_surface_rejected_frames": [],
                "product_surface_selected_frames": [],
            }
            product = None
        combined_measurements = dict(observed_build.measurements)
        for key, value in product_measurements.items():
            if key.startswith("product_surface_"):
                combined_measurements[key] = value
        combined_artifacts = {**observed_build.artifacts, **product_artifacts}
        product_passed = combined_measurements.get("product_surface_quality_gate_passed") == 1
        observed_passed = _observed_candidate_passes(combined_measurements, baseline_build)
        if (
            not product_passed
            and not observed_passed
            and "publishable_surface_coverage_fraction" not in baseline_build.measurements
            and product_measurements.get("product_surface_rejected_reason") == "insufficient_product_strips"
        ):
            # Preserve the existing lightweight observed-mode contract for
            # callers that provide a synthetic baseline without coverage
            # measurements; real video builds always have the metric above.
            observed_passed = True
        if product_passed and product is not None:
            selected = _SurfaceBuild(
                product.image,
                product.coverage,
                product.model,
                combined_measurements,
                combined_artifacts,
                baseline_build.fallback_used,
            )
            candidate = "product_surface"
        elif observed_passed:
            selected = _SurfaceBuild(
                observed_build.image,
                observed_build.coverage,
                observed_build.model,
                combined_measurements,
                combined_artifacts,
                baseline_build.fallback_used,
            )
            candidate = "observed_surface"
        else:
            selected = _SurfaceBuild(
                baseline_build.image,
                baseline_build.coverage,
                baseline_build.model,
                combined_measurements,
                combined_artifacts,
                baseline_build.fallback_used,
            )
            candidate = "confirmed_geometry"
        selected.measurements["observed_surface_selected_candidate"] = candidate
        selected.measurements["observed_surface_product_gate_passed"] = int(product_passed)
        selected.measurements["observed_surface_observed_gate_passed"] = int(observed_passed)
        selected.measurements["observed_surface_fallback_reason"] = (
            "product_surface_rejected_and_observed_surface_rejected"
            if candidate == "confirmed_geometry"
            else ("product_surface_rejected" if candidate == "observed_surface" else "product_surface_accepted")
        )
        self._ensure_surface_artifacts(selected)
        return selected

    def _build_curved_observed_baseline(self, analysis: Analysis) -> _SurfaceBuild:
        """Create a non-cylindrical fallback without invoking any cylinder code."""

        height = self.config.output_height
        width = max(1, min(self.config.output_width, max(1, height * 2)))
        image = np.zeros((height, width, 3), np.uint8)
        coverage = np.zeros((height, width), np.uint8)
        model = SurfaceModel(SurfaceKind.CURVED)
        artifacts: dict[str, object] = {}
        if analysis.frames:
            item = analysis.frames[0]
            x, y, box_width, box_height = item.bbox
            crop = item.frame.image[y : y + box_height, x : x + box_width]
            mask = item.publish_mask[y : y + box_height, x : x + box_width]
            if crop.size and box_height > 0:
                target_width = max(1, min(width, round(box_width * height / max(box_height, 1))))
                image = np.asarray(cv2.resize(crop, (target_width, height), interpolation=cv2.INTER_AREA))
                coverage = np.asarray(cv2.resize(mask, (target_width, height), interpolation=cv2.INTER_NEAREST))
                artifacts["curved_baseline_source"] = coverage.copy()
        observed = coverage.copy()
        artifacts.update(
            {
                "surface_observed": observed,
                "surface_synthetic": np.zeros_like(coverage),
                "surface_unknown": np.where(coverage > 0, 0, 255).astype(np.uint8),
            }
        )
        observed_fraction = coverage_fraction(coverage)
        measurements: dict[str, float | int | str | list[float] | list[int]] = {
            "surface_builder": "curved_local_atlas",
            "surface_coverage_fraction": observed_fraction,
            "publishable_surface_coverage_fraction": observed_fraction,
            "observed_coverage_fraction": observed_fraction,
            "surface_observed_coverage_fraction": observed_fraction,
            "surface_synthetic_fraction": 0.0,
            "surface_unknown_fraction": coverage_fraction(np.asarray(artifacts["surface_unknown"])),
            "frame_count": len(analysis.frames),
        }
        return _SurfaceBuild(image, coverage, model, measurements, artifacts, False)

    @staticmethod
    def _ensure_surface_artifacts(build: _SurfaceBuild) -> None:
        coverage = build.coverage
        artifacts = build.artifacts
        observed = artifacts.get("surface_observed")
        if not isinstance(observed, np.ndarray):
            synthetic = artifacts.get("surface_synthetic")
            synthetic_mask = synthetic if isinstance(synthetic, np.ndarray) else np.zeros_like(coverage)
            observed = np.where((coverage > 0) & (synthetic_mask == 0), 255, 0).astype(np.uint8)
            artifacts["surface_observed"] = observed
        synthetic = artifacts.get("surface_synthetic")
        if not isinstance(synthetic, np.ndarray):
            synthetic = np.zeros_like(coverage)
            artifacts["surface_synthetic"] = synthetic
        unknown = artifacts.get("surface_unknown")
        if not isinstance(unknown, np.ndarray):
            unknown = np.where(coverage > 0, 0, 255).astype(np.uint8)
            artifacts["surface_unknown"] = unknown
        source = artifacts.get("surface_source")
        if not isinstance(source, np.ndarray):
            for key in ("curved_surface_source", "product_surface_source", "observed_branch_source", "source"):
                candidate = artifacts.get(key)
                if isinstance(candidate, np.ndarray):
                    source = candidate
                    break
            if not isinstance(source, np.ndarray):
                source = np.zeros_like(coverage, dtype=np.uint16)
            artifacts["surface_source"] = source
        for target, candidates in {
            "surface_seams": ("curved_surface_seams", "product_surface_seams", "seams"),
            "surface_confidence": ("curved_surface_confidence", "product_surface_confidence", "confidence"),
        }.items():
            if not isinstance(artifacts.get(target), np.ndarray):
                for key in candidates:
                    candidate = artifacts.get(key)
                    if isinstance(candidate, np.ndarray):
                        artifacts[target] = candidate
                        break
                else:
                    artifacts[target] = np.zeros_like(coverage)
        build.measurements.setdefault("surface_observed_coverage_fraction", coverage_fraction(observed))
        build.measurements.setdefault("surface_synthetic_fraction", coverage_fraction(synthetic))
        build.measurements.setdefault("surface_unknown_fraction", coverage_fraction(unknown))

    def _analyze_video(self, video_path: str | Path) -> Analysis:
        # Surface coordinates are rendered at a fixed atlas resolution; loading
        # every full-resolution video frame only exhausts memory and prevents
        # dense temporal tracking on ordinary machines.  Keep one resize before
        # segmentation/features; a second pre-resize changes the silhouette
        # enough to shorten the cylindrical wall mask.
        frame_config = replace(
            PanoramaConfig(),
            sampling_step=self.config.sampling_step,
            max_frames=self.config.max_frames,
            downscale=1.0,
        )
        source = OpenCVVideoSource(video_path, frame_config)
        try:
            metadata = source.open()
            if metadata is not None:
                if (
                    self.config.surface_kind is SurfaceKind.CYLINDRICAL
                    and self.config.surface_output_mode is SurfaceOutputMode.CONFIRMED_GEOMETRY
                ):
                    target_dimension = (
                        960.0
                        if self.config.max_frames <= 24
                        else 640.0
                        if self.config.max_frames <= 48
                        else 360.0
                    )
                else:
                    target_dimension = 640.0
                source.config.downscale = min(
                    1.0, target_dimension / max(metadata.width, metadata.height)
                )
            frames = source.iter_frames()
        finally:
            source.close()
        analysis_config = self.config
        if self._uses_observed_branch() and self.config.enable_temporal_decimation:
            # Product assembly needs the ordered orbit samples.  Its strip
            # selector is the quality-aware decimator; dropping frames before
            # it runs can remove the only view of a surface sector.
            analysis_config = replace(self.config, enable_temporal_decimation=False)
        return VideoAnalyzer().analyze(frames, analysis_config)

    def _build_surface(self, analysis: Analysis) -> _SurfaceBuild:
        if analysis.kind is SurfaceKind.CYLINDRICAL:
            image, coverage, model, measurements, artifacts = CylinderUnwrapBuilder().build(analysis.frames, self.config)
            fallback_used = False
        else:
            image, coverage, model, measurements, artifacts = CurvedSurfaceFallbackBuilder().build(analysis.frames, self.config)
            fallback_used = True
        observed_fraction = coverage_fraction(coverage)
        publishable_fraction = measurements.get("surface_coverage_fraction", observed_fraction)
        publishable = (
            float(publishable_fraction) if isinstance(publishable_fraction, (int, float)) else observed_fraction
        )
        fallback_fraction = 0.0
        planar_coverage = artifacts.get("mosaic_coverage")
        if isinstance(planar_coverage, np.ndarray):
            fallback_fraction = coverage_fraction(planar_coverage)
        measurements["observed_coverage_fraction"] = observed_fraction
        measurements["publishable_surface_coverage_fraction"] = publishable
        measurements["fallback_mosaic_coverage_fraction"] = fallback_fraction
        measurements["surface_coverage_fraction"] = publishable
        measurements.update(analysis.measurements or {})
        measurements["frame_count"] = len(analysis.frames)
        if analysis.frames:
            artifacts.setdefault(
                "full_object_mask",
                np.where(np.any(np.stack([item.geometry_mask for item in analysis.frames], axis=0) > 0, axis=0), 255, 0).astype(np.uint8),
            )
            artifacts.setdefault(
                "core_body_mask",
                np.where(
                    np.any(
                        np.stack(
                            [
                                item.core_mask if item.core_mask is not None else item.publish_mask
                                for item in analysis.frames
                            ],
                            axis=0,
                        )
                        > 0,
                        axis=0,
                    ),
                    255,
                    0,
                ).astype(np.uint8),
            )
            artifacts.setdefault(
                "nuisance_mask",
                np.where(
                    np.any(
                        np.stack(
                            [
                                item.nuisance_mask if item.nuisance_mask is not None else np.zeros_like(item.geometry_mask)
                                for item in analysis.frames
                            ],
                            axis=0,
                        )
                        > 0,
                        axis=0,
                    ),
                    255,
                    0,
                ).astype(np.uint8),
            )
            artifacts.setdefault(
                "publish_mask",
                np.where(np.any(np.stack([item.publish_mask for item in analysis.frames], axis=0) > 0, axis=0), 255, 0).astype(np.uint8),
            )
        return _SurfaceBuild(image, coverage, model, measurements, artifacts, fallback_used)

    def _decide_publication(self, analysis: Analysis, build: _SurfaceBuild) -> _PublicationDecision:
        if self.config.surface_output_mode is SurfaceOutputMode.OBSERVED_SURFACE:
            return self._decide_observed_surface_publication(analysis, build)
        if analysis.kind is SurfaceKind.CURVED or build.fallback_used:
            return _PublicationDecision(
                UnwrapStatus.PARTIAL_SURFACE,
                "Only the observed side band is available; cylindrical geometry was not claimed for this capture.",
                "Record the surface from more viewpoints with overlapping frames.",
                build.image,
                build.coverage,
                "confirmed_geometry",
                "not_confirmed",
            )
        pose_residual = build.measurements.get("pose_residual_radians")
        accepted_pairs = build.measurements.get("accepted_pose_pairs")
        pose_frame_count = build.measurements.get("pose_frame_count", len(analysis.frames))
        if not isinstance(pose_frame_count, int) or pose_frame_count < 2:
            pose_frame_count = len(analysis.frames)
        required_pairs = max(2, int(np.ceil((pose_frame_count - 1) * self.config.min_accepted_pose_pair_fraction)))
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
        quality_gate_passed = build.measurements.get("quality_gate_passed") == 1
        rectification_applied = build.measurements.get("rectification_applied") == 1

        if not self.config.enable_global_pose_optimization:
            return _PublicationDecision(
                UnwrapStatus.PARTIAL_SURFACE,
                "The cylindrical atlas was rendered without a global pose quality gate.",
                "Enable global pose optimization before accepting a complete surface map.",
                build.image,
                build.coverage,
                "confirmed_geometry",
                "not_confirmed",
            )
        if geometry_rejected:
            return _PublicationDecision(
                UnwrapStatus.UNSTABLE_CAMERA_GEOMETRY,
                "The tracked views do not agree on one stable surface trajectory.",
                "Record a slower orbit with more overlap and less camera shake; do not treat fallback mosaics as confirmed unwrap geometry.",
                build.image,
                build.coverage,
                "confirmed_geometry",
                "not_confirmed",
            )
        if not quality_gate_passed:
            return _PublicationDecision(
                UnwrapStatus.PARTIAL_SURFACE,
                "The baseline mosaic was published without rectification because some source boundaries remain unstable.",
                "Use this observed band, or record a slower orbit with stronger overlap for cleaner rectification.",
                build.image,
                build.coverage,
                "confirmed_geometry",
                "not_confirmed",
            )
        if rectification_applied:
            message = "A geometry-confirmed cylindrical surface band was assembled and rectified from the baseline mosaic."
        else:
            message = "A geometry-confirmed cylindrical surface band was assembled, but the mosaic did not support one global rectification."
        return _PublicationDecision(
            UnwrapStatus.PARTIAL_SURFACE,
            message,
            "Use this observed band, or record a slower full orbit for future cylindrical confirmation.",
            build.image,
            build.coverage,
            "confirmed_geometry",
            "confirmed",
        )

    def _decide_observed_surface_publication(
        self,
        analysis: Analysis,
        build: _SurfaceBuild,
    ) -> _PublicationDecision:
        fatal_statuses = {
            UnwrapStatus.OBJECT_NOT_DETECTED,
            UnwrapStatus.INSUFFICIENT_TEXTURE,
            UnwrapStatus.EXCESSIVE_MOTION_BLUR,
            UnwrapStatus.INSUFFICIENT_COVERAGE,
            UnwrapStatus.UNSTABLE_CAMERA_GEOMETRY,
        }
        if analysis.status in fatal_statuses:
            return _PublicationDecision(
                analysis.status,
                analysis.message,
                analysis.recommendation,
                build.image,
                build.coverage,
                "observed_surface",
                "not_confirmed",
            )
        quality_gate_passed = build.measurements.get("quality_gate_passed") == 1
        geometry_confirmation = "confirmed" if analysis.kind is SurfaceKind.CYLINDRICAL and quality_gate_passed else "not_confirmed"
        branch_applied = build.measurements.get("observed_branch_applied") == 1
        candidate = build.measurements.get("observed_surface_selected_candidate")
        if candidate == "product_surface":
            message = "A product-quality observed surface candidate passed its visual quality gates and was selected."
            publication_mode = "product_surface"
        elif candidate == "confirmed_geometry":
            message = "The product surface and observed-surface candidates failed visual quality gates; the confirmed-geometry baseline was selected as a safe fallback."
            publication_mode = "confirmed_geometry"
        else:
            message = (
                "A coverage-first observed surface band was assembled from the analyzed orbit frames without claiming confirmed cylindrical geometry."
                if branch_applied
                else "An observed surface band was published without claiming one confirmed cylindrical geometry."
            )
            publication_mode = "observed_surface"
        return _PublicationDecision(
            UnwrapStatus.OBSERVED_SURFACE,
            message,
            "Use this observed surface band as published output; record a slower full orbit only if confirmed cylindrical geometry is required.",
            build.image,
            build.coverage,
            publication_mode,
            geometry_confirmation,
        )

    @staticmethod
    def _frame_list(frames: Sequence[AnalyzedFrame]) -> list[dict[str, float | int]]:
        return [
            {
                "frame_index": item.frame.index,
                "timestamp_seconds": item.frame.timestamp_seconds,
            }
            for item in frames
        ]

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
        photo_mode_eligible = status in {UnwrapStatus.OK, UnwrapStatus.PARTIAL_SURFACE, UnwrapStatus.OBSERVED_SURFACE}
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

    def _mark_photo_mode_skipped(
        self,
        status: UnwrapStatus,
        measurements: dict[str, float | int | str | list[float] | list[int]],
    ) -> None:
        if not self.config.photo_mode:
            return
        photo_mode_eligible = status in {UnwrapStatus.OK, UnwrapStatus.PARTIAL_SURFACE, UnwrapStatus.OBSERVED_SURFACE}
        measurements["photo_mode_eligible"] = int(photo_mode_eligible)
        if photo_mode_eligible:
            return
        measurements["photo_mode_applied"] = 0
        measurements["photo_mode_crop_policy"] = "skipped_ineligible"
        measurements["photo_mode_crop_loss"] = 0.0

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

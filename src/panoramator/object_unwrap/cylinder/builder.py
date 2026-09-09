from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np

from ..analyzer import AnalyzedFrame
from ..coverage import coverage_fraction, least_covered_seam
from ..image_pose_graph import ImagePoseGraph, build_image_pose_graph
from ..models import SurfaceModel, UnwrapConfig
from ..planar_mosaic import build_planar_mosaic
from ..rectification import estimate_strip, evaluate_mosaic_quality, rectify_mosaic
from .fitting import fit_cylinder
from .mapper import (
    angular_increment,
    central_band,
    cylindrical_motion_increment,
    feature_shift,
    flow_angular_increment,
    horizontal_shift,
    normalized_wall,
)
from .pose import (
    accumulate_vertical_offsets,
    select_cylindrical_keyframes,
    smooth_cylindrical_steps,
    solve_monotonic_trajectory,
    stabilize_motion_steps,
)
from .renderer import render_adaptive_slit_cylindrical_atlas


class CylinderUnwrapBuilder:
    def build(
        self, frames: list[AnalyzedFrame], config: UnwrapConfig
    ) -> tuple[
        np.ndarray, np.ndarray, SurfaceModel, dict[str, float | int | str | list[float] | list[int]], dict[str, object]
    ]:
        fit = None
        image_pose_graph = None
        planar_mosaic = None
        fragments: list[tuple[np.ndarray, np.ndarray]] = []
        if not config.enable_global_pose_optimization:
            fragments = [
                central_band(
                    *normalized_wall(item.frame.image, item.publish_mask, item.bbox, config.output_height),
                    config.central_band_ratio,
                )
                for item in frames
            ]
        half_view_angle = float(np.arcsin(config.central_band_ratio))
        observations: list[tuple[float, float]] = []
        vertical_deltas: list[float] = []
        motion_matches: list[int] = []
        motion_inliers: list[int] = []
        for index, (previous, current) in enumerate(itertools.pairwise(frames)):
            if config.enable_global_pose_optimization:
                step, response, vertical_delta, match_count, inlier_count = cylindrical_motion_increment(
                    previous.frame.image,
                    current.frame.image,
                    previous.geometry_mask,
                    current.geometry_mask,
                    previous.bbox,
                    current.bbox,
                )
            else:
                previous_fragment, previous_fragment_mask = fragments[index]
                current_fragment, current_fragment_mask = fragments[index + 1]
                step, response = angular_increment(
                    previous_fragment,
                    previous_fragment_mask,
                    current_fragment,
                    current_fragment_mask,
                    config.central_band_ratio,
                )
                if response < 0.25:
                    shift, fallback_response = feature_shift(
                        previous_fragment,
                        previous_fragment_mask,
                        current_fragment,
                        current_fragment_mask,
                    )
                    if fallback_response >= 0.25:
                        step = float(-shift / max(previous_fragment.shape[1], 1) * 2.0 * half_view_angle)
                        response = fallback_response
                    else:
                        shift, response = horizontal_shift(previous_fragment, current_fragment)
                        step = float(-shift / 256.0 * half_view_angle)
                vertical_delta = 0.0
                match_count = 0
                inlier_count = 0
            observations.append((step, response))
            vertical_deltas.append(vertical_delta)
            motion_matches.append(match_count)
            motion_inliers.append(inlier_count)
        responses = [response for _, response in observations]
        if config.enable_global_pose_optimization:
            dense_steps = smooth_cylindrical_steps(
                stabilize_motion_steps([step for step, _ in observations])
            )
            dense_angles = [0.0]
            for step in dense_steps:
                dense_angles.append(dense_angles[-1] + step)
            retained_indices = select_cylindrical_keyframes(
                frames, dense_steps, motion_inliers
            )
            selected_angles = [dense_angles[index] for index in retained_indices]
            if len(retained_indices) >= 2 and len(retained_indices) < len(frames):
                frames = [frames[index] for index in retained_indices]
                observations = []
                vertical_deltas = []
                motion_matches = []
                motion_inliers = []
                for previous, current in itertools.pairwise(frames):
                    step, response, vertical_delta, match_count, inlier_count = cylindrical_motion_increment(
                        previous.frame.image,
                        current.frame.image,
                        previous.geometry_mask,
                        current.geometry_mask,
                        previous.bbox,
                        current.bbox,
                    )
                    observations.append((step, response))
                    vertical_deltas.append(vertical_delta)
                    motion_matches.append(match_count)
                    motion_inliers.append(inlier_count)
            stabilized_steps = smooth_cylindrical_steps(
                stabilize_motion_steps([step for step, _ in observations])
            )
            stabilized_observations = list(
                zip(stabilized_steps, [response for _, response in observations], strict=True)
            )
            trajectory = solve_monotonic_trajectory(stabilized_observations)
            angles = selected_angles if len(selected_angles) == len(frames) else trajectory.angles
            steps = [right - left for left, right in itertools.pairwise(angles)]
            trajectory = replace(
                trajectory,
                angles=angles,
                steps=steps,
                sweep_radians=abs(angles[-1] - angles[0]),
                repeated_observation=abs(angles[-1] - angles[0]) > 2.0 * np.pi * 1.05,
            )
            pose_residual = trajectory.residual_radians
        else:
            raw_steps = [step for step, _ in observations]
            nonzero = [abs(step) for step in raw_steps if abs(step) > 1e-4]
            baseline = float(np.median(nonzero)) if nonzero else half_view_angle * 0.18
            direction = 1.0 if sum(raw_steps) >= 0 else -1.0
            steps = [direction * float(np.clip(abs(step), baseline * 0.35, baseline * 2.0)) for step in raw_steps]
            angles = [0.0]
            for step in steps:
                angles.append(angles[-1] + step)
            pose_residual = 0.0
            trajectory = None
        fit = fit_cylinder(frames)
        if len(frames) > 60:
            image_pose_graph = ImagePoseGraph([])
            planar_mosaic = None
        else:
            image_pose_graph = build_image_pose_graph(frames)
            planar_mosaic = build_planar_mosaic(
                frames,
                image_pose_graph.edges,
                config.output_height,
                config.publish_profile,
            )
        fragments = [
            central_band(
                *normalized_wall(item.frame.image, item.publish_mask, item.bbox, config.output_height),
                config.central_band_ratio,
            )
            for item in frames
        ]
        min_angle = min(angles) - half_view_angle
        max_angle = max(angles) + half_view_angle
        angle_span = max(max_angle - min_angle, 1e-6)
        atlas_width, pixels_per_radian = self._atlas_width(fit.boxes, angle_span, config)
        canvas, coverage, source_map, local_error, inverse_measurements = render_adaptive_slit_cylindrical_atlas(
            frames,
            angles,
            config.output_height,
            atlas_width,
            accumulate_vertical_offsets(vertical_deltas),
            config.interpolate_gaps,
            config.max_interpolation_gap_px,
        )
        artifacts: dict[str, object] = {
            "inverse_cylindrical": canvas.copy(),
            "inverse_cylindrical_coverage": coverage.copy(),
            "inverse_cylindrical_source": source_map.copy(),
            "inverse_cylindrical_error": local_error.copy(),
        }
        if trajectory is not None and planar_mosaic is not None:
            # Keep image-space mosaics for diagnostics and quality gates. The
            # inverse cylindrical atlas above is the only published renderer.
            mosaic, mosaic_coverage, mosaic_source, mosaic_error = planar_mosaic
            gate = evaluate_mosaic_quality(
                mosaic,
                mosaic_coverage,
                mosaic_source,
                mosaic_error,
                max_mean_boundary_error=config.max_mosaic_boundary_mean_error,
                max_severe_boundary_fraction=config.max_mosaic_boundary_severe_fraction,
                severe_error_threshold=config.mosaic_boundary_severe_error,
                max_severe_boundary_footprint=config.max_mosaic_boundary_severe_footprint
                * config.publish_profile_settings()["severe_footprint_multiplier"],
                max_anchor_conflict_footprint=config.max_mosaic_anchor_conflict_footprint
                * config.publish_profile_settings()["anchor_conflict_multiplier"],
                max_owner_instability=config.max_mosaic_owner_instability
                * config.publish_profile_settings()["owner_instability_multiplier"],
            )
            artifacts.update(
                {
                    "mosaic": mosaic,
                    "mosaic_coverage": mosaic_coverage,
                    "mosaic_source": mosaic_source,
                    "mosaic_error": mosaic_error,
                    "mosaic_boundary": gate.boundary_map,
                    "mosaic_saliency": gate.saliency_map,
                    "mosaic_saliency_error": gate.saliency_error_map,
                    "mosaic_overlap_conflict": gate.overlap_conflict_map,
                    "mosaic_owner_transition": gate.owner_transition_map,
                    "mosaic_owner_instability": gate.owner_instability_map,
                    "mosaic_seam_risk": gate.seam_risk_map,
                }
            )
            if gate.passed and trajectory.accepted_pairs:
                strip = estimate_strip(
                    mosaic_coverage,
                    min_column_fraction=float(
                        np.clip(
                            config.min_rectification_column_fraction
                            + config.publish_profile_settings()["rectification_column_fraction_delta"],
                            0.1,
                            1.0,
                        )
                    ),
                    smoothing_window=config.rectification_smoothing_window,
                    max_axis_step=config.max_rectification_axis_step,
                )
                if strip is not None:
                    rectified, rectified_coverage, rectified_source, reprojection_error = rectify_mosaic(
                        mosaic,
                        mosaic_coverage,
                        mosaic_source,
                        mosaic_error,
                        strip,
                        config.output_height,
                        config.output_width,
                    )
                    artifacts["rectified_mosaic"] = rectified
                    artifacts["rectified_mosaic_coverage"] = rectified_coverage
                    artifacts["rectified_mosaic_source"] = rectified_source
                    artifacts["rectified_mosaic_error"] = reprojection_error
                    measurements = {
                        **gate.measurements,
                        **strip.measurements,
                        "rectification_applied": 1,
                        "publish_profile": config.publish_profile.value,
                    }
                else:
                    artifacts["mosaic_error"] = mosaic_error
                    measurements = {**gate.measurements, "rectification_applied": 0, "publish_profile": config.publish_profile.value}
            else:
                artifacts["mosaic_error"] = mosaic_error
                measurements = {**gate.measurements, "rectification_applied": 0, "publish_profile": config.publish_profile.value}
        else:
            measurements = {"quality_gate_passed": 0, "rectification_applied": 0, "publish_profile": config.publish_profile.value}
        seam = least_covered_seam(coverage)
        # Preserve chronological orientation: x=0 corresponds to the first
        # observation.  Moving the seam is a presentation choice and must not
        # silently reorder the surface sequence.
        fit.model.seam_angle_degrees = seam / atlas_width * angle_span / (2 * np.pi) * 360.0
        pose_pairs = [
            {
                "left_frame": frames[index].frame.index,
                "right_frame": frames[index + 1].frame.index,
                "delta_radians": float(delta),
                "confidence": float(confidence),
                "accepted": bool(trajectory.accepted[index]) if trajectory else False,
                "rejection_reason": trajectory.rejection_reasons[index] if trajectory else "global_pose_disabled",
            }
            for index, (delta, confidence) in enumerate(observations)
        ]
        artifacts.update({
            "source": source_map,
            "reprojection_error": np.clip(local_error, 0, 255).astype(np.uint8),
            "pose_pairs": pose_pairs,
            "image_pose_graph": image_pose_graph.edges,
        })
        if planar_mosaic is not None:
            mosaic, mosaic_coverage, mosaic_source, mosaic_error = planar_mosaic
            artifacts.update(
                {
                    "mosaic": mosaic,
                    "mosaic_coverage": mosaic_coverage,
                    "mosaic_source": mosaic_source,
                    "mosaic_error": mosaic_error,
                    "image_space_mosaic": mosaic,
                    "image_space_mosaic_coverage": mosaic_coverage,
                    "image_space_mosaic_source": mosaic_source,
                    "image_space_mosaic_error": mosaic_error,
                }
            )
        if trajectory is not None:
            feature_mosaic, feature_coverage, feature_source, feature_error = self._feature_mosaic(
                fragments, angles, min_angle, angle_span, atlas_width
            )
            artifacts.update(
                {
                    "angular_mosaic": feature_mosaic,
                    "angular_mosaic_coverage": feature_coverage,
                    "angular_mosaic_source": feature_source,
                    "angular_mosaic_error": feature_error,
                }
            )
        return canvas, coverage, fit.model, {
            "coverage_fraction": coverage_fraction(coverage),
            "surface_coverage_fraction": min(1.0, angle_span / (2.0 * np.pi)),
            "observed_angle_degrees": angle_span / (2.0 * np.pi) * 360.0,
            "atlas_width": atlas_width,
            "pixels_per_radian": pixels_per_radian,
            "match_response": responses,
            "feature_matches": motion_matches,
            "feature_inliers": motion_inliers,
            "angular_steps": steps,
            "pose_residual_radians": pose_residual,
            "accepted_pose_pairs": (trajectory.accepted_pairs if trajectory else 0),
            "rejected_pose_pairs": (len(observations) - trajectory.accepted_pairs if trajectory else len(observations)),
            "trajectory_sweep_degrees": (
                trajectory.sweep_radians / (2.0 * np.pi) * 360.0 if trajectory else 0.0
            ),
            "pose_frame_count": len(frames),
            "repeated_observation_detected": (
                int(trajectory.repeated_observation) if trajectory else 0
            ),
            "mapping": "surface_angle_height",
            "rendering": "inverse_cylindrical_atlas",
            "primary_renderer": "inverse_cylindrical_atlas",
            **inverse_measurements,
            "image_pose_graph_edges": len(image_pose_graph.edges),
            "image_pose_graph_valid_edges": image_pose_graph.valid_edges,
            "planar_mosaic_available": int(planar_mosaic is not None),
            **measurements,
        }, artifacts

    @staticmethod
    def _feature_mosaic(
        fragments: list[tuple[np.ndarray, np.ndarray]],
        angles: list[float],
        min_angle: float,
        angle_span: float,
        atlas_width: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compose the observed central bands in one common angular canvas.

        This is deliberately an affine-free *single* surface coordinate system:
        the accepted global angles choose a centre for each feature patch, and
        masks/feather weights resolve overlap.  Thus an invalid pair cannot
        create a thin independently projected strip in the published result.
        """
        height = fragments[0][0].shape[0]
        accum = np.zeros((height, atlas_width, 3), np.float32)
        total_weight = np.zeros((height, atlas_width), np.float32)
        owner_weight = np.zeros((height, atlas_width), np.float32)
        source = np.zeros((height, atlas_width), np.uint16)
        error = np.zeros((height, atlas_width), np.float32)
        for index, ((image, mask), angle) in enumerate(zip(fragments, angles, strict=True), start=1):
            width = image.shape[1]
            centre = round((angle - min_angle) / max(angle_span, 1e-9) * (atlas_width - 1))
            left, right = max(0, centre - width // 2), min(atlas_width, centre - width // 2 + width)
            source_left, source_right = left - (centre - width // 2), right - (centre - width // 2)
            patch = image[:, source_left:source_right]
            valid = mask[:, source_left:source_right] > 0
            if not np.any(valid):
                continue
            horizontal = np.minimum(np.arange(source_left, source_right) + 1, width - np.arange(source_left, source_right))
            feather = np.clip(horizontal / max(width * 0.18, 1.0), 0.03, 1.0).astype(np.float32)
            weight = valid.astype(np.float32) * feather[None, :]
            old = total_weight[:, left:right] > 0
            difference = np.mean(np.abs(accum[:, left:right] / np.maximum(total_weight[:, left:right, None], 1e-6) - patch), axis=2)
            error_region = error[:, left:right]
            conflict = old & valid
            error_region[conflict] = np.maximum(error_region[conflict], difference[conflict])
            accum[:, left:right] += patch.astype(np.float32) * weight[..., None]
            total_weight[:, left:right] += weight
            owner_region = owner_weight[:, left:right]
            source_region = source[:, left:right]
            replace = weight > owner_region
            source_region[replace] = index
            owner_region[replace] = weight[replace]
        canvas = np.clip(accum / np.maximum(total_weight[..., None], 1e-6), 0, 255).astype(np.uint8)
        coverage = np.where(total_weight > 0, 255, 0).astype(np.uint8)
        return canvas, coverage, source, np.clip(error, 0, 255).astype(np.uint8)

    @staticmethod
    def _atlas_width(boxes: list[tuple[int, int, int, int]], angle_span: float, config: UnwrapConfig) -> tuple[int, float]:
        """Choose atlas width in the same physical scale as its height.

        Near the centre of a cylindrical view, ``dx = radius * d_angle``.
        Scaling vertical pixels by ``output_height / source_height`` therefore
        defines the only aspect-preserving scale for the unwrapped horizontal
        coordinate.  ``output_width`` is a resolution ceiling, not permission
        to stretch a partial angular observation to a fixed rectangle.
        """
        source_height = float(np.median([box[3] for box in boxes]))
        radius = float(np.median([box[2] for box in boxes])) * 0.5
        pixels_per_radian = config.output_height / max(source_height, 1.0) * max(radius, 1.0)
        width = round(angle_span * pixels_per_radian)
        return int(np.clip(width, 64, config.output_width)), pixels_per_radian

    def _global_angles(
        self, fragments: list[tuple[np.ndarray, np.ndarray]], central_band_ratio: float
    ) -> tuple[list[float], list[float], list[float], float]:
        """Solve all reliable relative azimuth constraints simultaneously."""
        constraints: list[tuple[int, int, float, float]] = []
        responses: list[float] = []
        for left_index in range(len(fragments) - 1):
            for right_index in range(left_index + 1, min(len(fragments), left_index + 4)):
                left, left_mask = fragments[left_index]
                right, right_mask = fragments[right_index]
                if right_index == left_index + 1:
                    delta, confidence = flow_angular_increment(left, left_mask, right, central_band_ratio)
                else:
                    delta, confidence = angular_increment(left, left_mask, right, right_mask, central_band_ratio)
                if confidence >= 0.45:
                    constraints.append((left_index, right_index, delta, confidence))
                    responses.append(confidence)
        if not constraints:
            return [0.0] * len(fragments), responses, [], float("inf")
        signed = [delta for _, _, delta, _ in constraints if abs(delta) > 1e-4]
        direction = 1.0 if not signed or float(np.median(signed)) >= 0 else -1.0
        constraints = [item for item in constraints if abs(item[2]) < 1e-4 or item[2] * direction > 0]
        count = len(fragments)
        typical_step = float(np.median([abs(delta) for _, _, delta, _ in constraints])) if constraints else 0.05
        # Weak temporal regularisation connects intervals with no usable visual
        # tracks and prevents a local false match from reversing the orbit.
        constraints.extend((index, index + 1, direction * typical_step, 0.08) for index in range(count - 1))
        robust_weights = np.array([confidence for _, _, _, confidence in constraints], dtype=float)
        solution = np.zeros(count, dtype=float)
        residuals = np.zeros(len(constraints), dtype=float)
        for _ in range(6):
            matrix = np.zeros((len(constraints) + 1, count), dtype=float)
            target = np.zeros(len(constraints) + 1, dtype=float)
            for row, (left_index, right_index, delta, _) in enumerate(constraints):
                weight = np.sqrt(robust_weights[row])
                matrix[row, left_index] = -weight
                matrix[row, right_index] = weight
                target[row] = weight * delta
            matrix[-1, 0] = 10.0
            solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
            residuals = np.array([solution[right] - solution[left] - delta for left, right, delta, _ in constraints])
            scale = max(float(np.median(np.abs(residuals))) * 1.4826, 0.015)
            base_weights = np.array([confidence for _, _, _, confidence in constraints])
            robust_weights = base_weights * np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residuals), 1e-6))
        monotonic = np.maximum.accumulate(solution * direction)
        solution = monotonic * direction
        steps = [float(solution[index + 1] - solution[index]) for index in range(count - 1)]
        return solution.tolist(), responses, steps, float(np.sqrt(np.mean(residuals**2)))

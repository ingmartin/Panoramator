from __future__ import annotations

import cv2
import numpy as np

from ..analyzer import AnalyzedFrame
from ..coverage import coverage_fraction, least_covered_seam
from ..image_pose_graph import build_image_pose_graph
from ..planar_mosaic import build_planar_mosaic
from ..models import SurfaceModel, UnwrapConfig
from .fitting import fit_cylinder
from .mapper import angular_increment, central_band, feature_shift, flow_angular_increment, horizontal_shift, normalized_wall
from .pose import solve_monotonic_trajectory


class CylinderUnwrapBuilder:
    def build(
        self, frames: list[AnalyzedFrame], config: UnwrapConfig
    ) -> tuple[
        np.ndarray, np.ndarray, SurfaceModel, dict[str, float | int | str | list[float] | list[int]], dict[str, object]
    ]:
        fit = fit_cylinder(frames)
        image_pose_graph = build_image_pose_graph(frames)
        planar_mosaic = build_planar_mosaic(frames, image_pose_graph.edges, config.output_height)
        fragments = [
            central_band(*normalized_wall(item.frame.image, item.mask, item.bbox, config.output_height), config.central_band_ratio)
            for item in frames
        ]
        half_view_angle = float(np.arcsin(config.central_band_ratio))
        observations: list[tuple[float, float]] = []
        for (previous, previous_mask), (current, current_mask) in zip(fragments, fragments[1:], strict=False):
            if config.enable_global_pose_optimization:
                step, response = flow_angular_increment(previous, previous_mask, current, config.central_band_ratio)
                if response < 0.35:
                    step, response = angular_increment(previous, previous_mask, current, current_mask, config.central_band_ratio)
            else:
                step, response = angular_increment(previous, previous_mask, current, current_mask, config.central_band_ratio)
                if response < 0.25:
                    shift, fallback_response = feature_shift(previous, previous_mask, current, current_mask)
                    if fallback_response >= 0.25:
                        step = float(-shift / max(previous.shape[1], 1) * 2.0 * half_view_angle)
                        response = fallback_response
                    else:
                        shift, response = horizontal_shift(previous, current)
                        step = float(-shift / 256.0 * half_view_angle)
            observations.append((step, response))
        responses = [response for _, response in observations]
        if config.enable_global_pose_optimization:
            trajectory = solve_monotonic_trajectory(observations)
            angles, steps = trajectory.angles, trajectory.steps
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
        min_angle = min(angles) - half_view_angle
        max_angle = max(angles) + half_view_angle
        angle_span = max(max_angle - min_angle, 1e-6)
        atlas_width, pixels_per_radian = self._atlas_width(fit.boxes, angle_span, config)
        canvas = np.zeros((config.output_height, atlas_width, 3), np.uint8)
        weights = np.zeros((config.output_height, atlas_width), np.float32)
        # A pixel-owner map makes the selected source explicit.  Zero means an
        # unobserved pixel; frame numbers are stored one-based so it is suitable
        # for a lossless grayscale PNG.
        source_map = np.zeros((config.output_height, atlas_width), np.uint16)
        local_error = np.zeros((config.output_height, atlas_width), np.float32)
        target_angles = np.linspace(min_angle, max_angle, atlas_width, dtype=np.float32)
        for frame_offset, (item, (_, _), angle) in enumerate(zip(frames, fragments, angles, strict=True), start=1):
            local_angles = target_angles - angle
            visible = np.abs(local_angles) <= half_view_angle
            # Orthographic cylindrical projection: x / r = sin(theta).  The
            # inverse mapping preserves the geometry of the observed texture
            # instead of linearly stretching each source strip.
            x, y, width, height = item.bbox
            axis_x = x + (width - 1) * 0.5
            radius = max((width - 1) * 0.5, 1.0)
            # Each source view has its own fitted axis, radius and vertical
            # interval.  This avoids stretching a view to another view's box.
            map_x_row = (axis_x + radius * np.sin(local_angles)).astype(np.float32)
            map_x_row[~visible] = -1.0
            map_x = np.repeat(map_x_row[None, :], config.output_height, axis=0)
            source_y = np.linspace(y, y + height - 1, config.output_height, dtype=np.float32)
            source_map_y = np.repeat(source_y[:, None], atlas_width, axis=1)
            mapped_image = cv2.remap(item.frame.image, map_x, source_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mapped_mask = cv2.remap(item.mask, map_x, source_map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            column_weights = np.cos(local_angles).astype(np.float32)
            column_weights[~visible] = 0.0
            for target_x in np.flatnonzero(visible):
                valid = mapped_mask[:, target_x] > 0
                competing = valid & (weights[:, target_x] > 0)
                if np.any(competing):
                    difference = np.mean(
                        np.abs(
                            canvas[competing, target_x].astype(np.float32)
                            - mapped_image[competing, target_x].astype(np.float32)
                        ),
                        axis=1,
                    )
                    local_error[competing, target_x] = np.maximum(local_error[competing, target_x], difference)
                replace = valid & (column_weights[target_x] > weights[:, target_x])
                canvas[replace, target_x] = mapped_image[replace, target_x]
                weights[replace, target_x] = column_weights[target_x]
                source_map[replace, target_x] = frame_offset
        coverage = np.where(weights > 0, 255, 0).astype(np.uint8)
        if trajectory is not None and trajectory.accepted_pairs:
            # The per-frame cylindrical remap above is retained only as a
            # diagnostic fallback.  A pose-validated result is composed first
            # as one mask-aware feature mosaic; it is not a row of unrelated
            # source strips warped independently into the final atlas.
            canvas, coverage, source_map, reprojection_error = self._feature_mosaic(
                fragments, angles, min_angle, angle_span, atlas_width
            )
            local_error = reprojection_error.astype(np.float32)
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
        artifacts: dict[str, object] = {
            "source": source_map,
            "reprojection_error": np.clip(local_error, 0, 255).astype(np.uint8),
            "pose_pairs": pose_pairs,
            "image_pose_graph": image_pose_graph.edges,
        }
        if planar_mosaic is not None:
            mosaic, mosaic_coverage, mosaic_source, mosaic_error = planar_mosaic
            artifacts.update({"mosaic": mosaic, "mosaic_coverage": mosaic_coverage, "mosaic_source": mosaic_source, "mosaic_error": mosaic_error})
        return canvas, coverage, fit.model, {
            "coverage_fraction": coverage_fraction(coverage),
            "surface_coverage_fraction": min(1.0, angle_span / (2.0 * np.pi)),
            "observed_angle_degrees": angle_span / (2.0 * np.pi) * 360.0,
            "atlas_width": atlas_width,
            "pixels_per_radian": pixels_per_radian,
            "match_response": responses,
            "angular_steps": steps,
            "pose_residual_radians": pose_residual,
            "accepted_pose_pairs": (trajectory.accepted_pairs if trajectory else 0),
            "rejected_pose_pairs": (len(observations) - trajectory.accepted_pairs if trajectory else len(observations)),
            "trajectory_sweep_degrees": (
                trajectory.sweep_radians / (2.0 * np.pi) * 360.0 if trajectory else 0.0
            ),
            "repeated_observation_detected": (
                int(trajectory.repeated_observation) if trajectory else 0
            ),
            "mapping": "surface_angle_height",
            "rendering": "feature_mosaic_then_global_rectification" if trajectory is not None else "experimental_frame_projection",
            "image_pose_graph_edges": len(image_pose_graph.edges),
            "image_pose_graph_valid_edges": image_pose_graph.valid_edges,
            "planar_mosaic_available": int(planar_mosaic is not None),
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
            centre = int(round((angle - min_angle) / max(angle_span, 1e-9) * (atlas_width - 1)))
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
        width = int(round(angle_span * pixels_per_radian))
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

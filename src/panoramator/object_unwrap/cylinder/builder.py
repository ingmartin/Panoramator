from __future__ import annotations

import cv2
import numpy as np

from ..analyzer import AnalyzedFrame
from ..coverage import coverage_fraction, least_covered_seam
from ..models import SurfaceModel, UnwrapConfig
from .fitting import fit_cylinder
from .mapper import angular_increment, central_band, feature_shift, flow_angular_increment, horizontal_shift, normalized_wall


class CylinderUnwrapBuilder:
    def build(self, frames: list[AnalyzedFrame], config: UnwrapConfig) -> tuple[np.ndarray, np.ndarray, SurfaceModel, dict[str, float | int | str | list[float]]]:
        fit = fit_cylinder(frames)
        fragments = [
            central_band(*normalized_wall(item.frame.image, item.mask, item.bbox, config.output_height), config.central_band_ratio)
            for item in frames
        ]
        half_view_angle = float(np.arcsin(config.central_band_ratio))
        responses: list[float] = []
        raw_steps: list[float] = []
        for (previous, previous_mask), (current, current_mask) in zip(fragments, fragments[1:], strict=False):
            step, response = angular_increment(previous, previous_mask, current, current_mask, config.central_band_ratio)
            if response < 0.25:
                shift, fallback_response = feature_shift(previous, previous_mask, current, current_mask)
                if fallback_response >= 0.25:
                    step = float(-shift / max(previous.shape[1], 1) * 2.0 * half_view_angle)
                    response = fallback_response
                else:
                    shift, response = horizontal_shift(previous, current)
                    step = float(-shift / 256.0 * half_view_angle)
            responses.append(response)
            raw_steps.append(step)
        nonzero = [abs(step) for step in raw_steps if abs(step) > 1e-4]
        baseline = float(np.median(nonzero)) if nonzero else half_view_angle * 0.18
        direction = 1.0 if sum(raw_steps) >= 0 else -1.0
        lower, upper = baseline * 0.35, baseline * 2.0
        steps = [direction * float(np.clip(abs(step), lower, upper)) for step in raw_steps]
        angles = [0.0]
        for step in steps:
            angles.append(angles[-1] + step)
        pose_residual = 0.0
        min_angle = min(angles) - half_view_angle
        max_angle = max(angles) + half_view_angle
        angle_span = max(max_angle - min_angle, 1e-6)
        canvas = np.zeros((config.output_height, config.output_width, 3), np.uint8)
        weights = np.zeros((config.output_height, config.output_width), np.float32)
        target_angles = np.linspace(min_angle, max_angle, config.output_width, dtype=np.float32)
        for item, (_, _), angle in zip(frames, fragments, angles, strict=True):
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
            source_map_y = np.repeat(source_y[:, None], config.output_width, axis=1)
            mapped_image = cv2.remap(item.frame.image, map_x, source_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mapped_mask = cv2.remap(item.mask, map_x, source_map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            column_weights = np.cos(local_angles).astype(np.float32)
            column_weights[~visible] = 0.0
            for target_x in np.flatnonzero(visible):
                valid = mapped_mask[:, target_x] > 0
                replace = valid & (column_weights[target_x] > weights[:, target_x])
                canvas[replace, target_x] = mapped_image[replace, target_x]
                weights[replace, target_x] = column_weights[target_x]
        coverage = np.where(weights > 0, 255, 0).astype(np.uint8)
        seam = least_covered_seam(coverage)
        # Preserve chronological orientation: x=0 corresponds to the first
        # observation.  Moving the seam is a presentation choice and must not
        # silently reorder the surface sequence.
        fit.model.seam_angle_degrees = seam / config.output_width * angle_span / (2 * np.pi) * 360.0
        return canvas, coverage, fit.model, {
            "coverage_fraction": coverage_fraction(coverage),
            "surface_coverage_fraction": min(1.0, angle_span / (2.0 * np.pi)),
            "observed_angle_degrees": angle_span / (2.0 * np.pi) * 360.0,
            "match_response": responses,
            "angular_steps": steps,
            "pose_residual_radians": pose_residual,
            "mapping": "surface_angle_height",
        }

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

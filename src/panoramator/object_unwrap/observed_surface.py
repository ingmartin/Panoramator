from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from . import analyzer as analyzer_module
from .analyzer import Analysis, AnalyzedFrame
from .coverage import coverage_fraction
from .cylinder.mapper import central_band, horizontal_shift, normalized_wall
from .models import SurfaceKind, SurfaceModel, UnwrapConfig

if TYPE_CHECKING:
    from .service import _SurfaceBuild


@dataclass(slots=True)
class ObservedSurfaceBuild:
    image: np.ndarray
    coverage: np.ndarray
    model: SurfaceModel
    measurements: dict[str, float | int | str | list[float] | list[int]]
    artifacts: dict[str, object]


@dataclass(slots=True)
class _PreparedFrame:
    item: AnalyzedFrame
    mask: np.ndarray
    angle_to_vertical: float


class ObservedSurfaceBuilder:
    def build(
        self,
        analysis: Analysis,
        config: UnwrapConfig,
        baseline_build: _SurfaceBuild,
    ) -> ObservedSurfaceBuild:
        measurements = _baseline_measurements(baseline_build.measurements)
        artifacts = dict(baseline_build.artifacts)
        artifacts["baseline_publish_image"] = baseline_build.image
        artifacts["baseline_publish_coverage"] = baseline_build.coverage
        artifacts["baseline_build_measurements"] = baseline_build.measurements

        branch_rejected: list[dict[str, float | int | str]] = []
        artifacts["observed_branch_rejected_frames"] = branch_rejected
        input_count = len(analysis.frames)
        measurements["observed_branch_input_frame_count"] = input_count
        if input_count < 6:
            return self._fallback(
                baseline_build,
                measurements,
                artifacts,
                branch_rejected,
                abort_reason="insufficient_analyzed_frames",
                axis_valid_count=0,
                selected_count=0,
                redundant_count=0,
                mask_fallback_count=0,
            )

        prepared, mask_fallback_count = _prepare_frames(analysis.frames, branch_rejected)
        axis_valid = [item for item in prepared if item.angle_to_vertical <= 18.0]
        artifacts["observed_branch_axis_valid_frames"] = _frame_payload(axis_valid)
        artifacts["observed_branch_selected_frames"] = []
        for item in prepared:
            if item.angle_to_vertical > 18.0:
                branch_rejected.append(
                    {
                        "frame_index": item.item.frame.index,
                        "timestamp_seconds": item.item.frame.timestamp_seconds,
                        "reason": "observed_branch_axis_instability",
                        "angle_to_vertical_degrees": round(float(item.angle_to_vertical), 6),
                    }
                )
        if not axis_valid:
            return self._fallback(
                baseline_build,
                measurements,
                artifacts,
                branch_rejected,
                abort_reason="branch_build_failed",
                axis_valid_count=0,
                selected_count=0,
                redundant_count=0,
                mask_fallback_count=mask_fallback_count,
            )

        selected, redundant_rejected = _select_frames(axis_valid)
        branch_rejected.extend(redundant_rejected)
        if len(axis_valid) >= 12 and len(selected) < 12:
            selected = _force_minimum_selection(axis_valid, 12)
        if not selected:
            return self._fallback(
                baseline_build,
                measurements,
                artifacts,
                branch_rejected,
                abort_reason="branch_build_failed",
                axis_valid_count=len(axis_valid),
                selected_count=0,
                redundant_count=len(redundant_rejected),
                mask_fallback_count=mask_fallback_count,
            )

        canvas_height = config.output_height
        baseline_width = int(baseline_build.image.shape[1])
        existing_atlas_width = _safe_int(baseline_build.measurements.get("atlas_width"), baseline_width)
        canvas_width = max(existing_atlas_width, baseline_width)
        fragments = [_normalized_fragment(item.item, item.mask, canvas_height) for item in selected]
        offsets, used_steps, used_phase_correlation, used_bbox_fallback = self._offsets(
            selected,
            fragments,
            config,
            baseline_build,
        )
        offset_scale = _offset_scale(analysis, baseline_build.measurements)
        if offset_scale != 1.0:
            offsets = [offsets[0], *[step * offset_scale for step in offsets[1:]]]
        mosaic, coverage, source, overlap_fraction, mean_gradient_gain = _compose_mosaic(
            fragments,
            offsets,
            canvas_height,
            canvas_width,
        )
        connected_fraction = _largest_component_fraction(coverage)
        branch_gradient_energy = _gradient_energy(mosaic, coverage)
        baseline_gradient_energy = _gradient_energy(baseline_build.image, baseline_build.coverage)
        gradient_energy_ratio = (
            float(branch_gradient_energy / baseline_gradient_energy)
            if baseline_gradient_energy > 1e-6
            else 0.0
        )
        if np.count_nonzero(coverage) == 0 or connected_fraction < 0.82:
            return self._fallback(
                baseline_build,
                measurements,
                artifacts,
                branch_rejected,
                abort_reason="branch_build_failed",
                axis_valid_count=len(axis_valid),
                selected_count=len(selected),
                redundant_count=len(redundant_rejected),
                mask_fallback_count=mask_fallback_count,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                used_steps=used_steps,
                used_phase_correlation=used_phase_correlation,
                used_bbox_fallback=used_bbox_fallback,
                connected_fraction=connected_fraction,
                branch_gradient_energy=branch_gradient_energy,
                baseline_gradient_energy=baseline_gradient_energy,
                gradient_energy_ratio=gradient_energy_ratio,
            )
        visual_quality_passed = not (
            baseline_gradient_energy > 1e-6
            and gradient_energy_ratio > 1.75
        )
        if not visual_quality_passed:
            return self._fallback(
                baseline_build,
                measurements,
                artifacts,
                branch_rejected,
                abort_reason="branch_visual_quality_rejected",
                axis_valid_count=len(axis_valid),
                selected_count=len(selected),
                redundant_count=len(redundant_rejected),
                mask_fallback_count=mask_fallback_count,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                used_steps=used_steps,
                used_phase_correlation=used_phase_correlation,
                used_bbox_fallback=used_bbox_fallback,
                connected_fraction=connected_fraction,
                branch_gradient_energy=branch_gradient_energy,
                baseline_gradient_energy=baseline_gradient_energy,
                gradient_energy_ratio=gradient_energy_ratio,
                visual_quality_passed=False,
            )

        artifacts["observed_branch_axis_valid_frames"] = _frame_payload(axis_valid)
        artifacts["observed_branch_selected_frames"] = _frame_payload(selected)
        artifacts["observed_branch_mosaic"] = mosaic
        artifacts["observed_branch_coverage"] = coverage
        artifacts["observed_branch_source"] = source

        observed_fraction = coverage_fraction(coverage)
        measurements["coverage_fraction"] = observed_fraction
        measurements["observed_coverage_fraction"] = observed_fraction
        measurements["publishable_surface_coverage_fraction"] = observed_fraction
        measurements["surface_coverage_fraction"] = observed_fraction
        measurements["observed_branch_applied"] = 1
        measurements["observed_branch_input_frame_count"] = input_count
        measurements["observed_branch_axis_valid_frame_count"] = len(axis_valid)
        measurements["observed_branch_selected_frame_count"] = len(selected)
        measurements["observed_branch_redundant_frame_count"] = len(redundant_rejected)
        measurements["observed_branch_mask_fallback_count"] = mask_fallback_count
        measurements["observed_branch_canvas_width"] = canvas_width
        measurements["observed_branch_canvas_height"] = canvas_height
        measurements["observed_branch_used_angular_steps"] = int(used_steps)
        measurements["observed_branch_used_phase_correlation"] = int(used_phase_correlation)
        measurements["observed_branch_used_bbox_fallback"] = int(used_bbox_fallback)
        measurements["observed_branch_coverage_fraction"] = observed_fraction
        measurements["observed_branch_overlap_conflict_fraction"] = overlap_fraction
        measurements["observed_branch_mean_gradient_gain"] = mean_gradient_gain
        measurements["observed_branch_largest_component_fraction"] = connected_fraction
        measurements["observed_branch_offset_scale"] = offset_scale
        measurements["observed_branch_visual_quality_passed"] = 1
        measurements["observed_branch_gradient_energy"] = branch_gradient_energy
        measurements["observed_branch_baseline_gradient_energy"] = baseline_gradient_energy
        measurements["observed_branch_gradient_energy_ratio"] = gradient_energy_ratio
        return ObservedSurfaceBuild(mosaic, coverage, baseline_build.model, measurements, artifacts)

    def _fallback(
        self,
        baseline_build: _SurfaceBuild,
        measurements: dict[str, float | int | str | list[float] | list[int]],
        artifacts: dict[str, object],
        branch_rejected: list[dict[str, float | int | str]],
        *,
        abort_reason: str,
        axis_valid_count: int,
        selected_count: int,
        redundant_count: int,
        mask_fallback_count: int,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
        used_steps: bool = False,
        used_phase_correlation: bool = False,
        used_bbox_fallback: bool = False,
        connected_fraction: float = 0.0,
        branch_gradient_energy: float = 0.0,
        baseline_gradient_energy: float = 0.0,
        gradient_energy_ratio: float = 0.0,
        visual_quality_passed: bool = True,
    ) -> ObservedSurfaceBuild:
        measurements["observed_branch_applied"] = 0
        measurements["observed_branch_abort_reason"] = abort_reason
        measurements["observed_branch_axis_valid_frame_count"] = axis_valid_count
        measurements["observed_branch_selected_frame_count"] = selected_count
        measurements["observed_branch_redundant_frame_count"] = redundant_count
        measurements["observed_branch_mask_fallback_count"] = mask_fallback_count
        measurements["observed_branch_canvas_width"] = canvas_width or int(baseline_build.image.shape[1])
        measurements["observed_branch_canvas_height"] = canvas_height or int(baseline_build.image.shape[0])
        measurements["observed_branch_used_angular_steps"] = int(used_steps)
        measurements["observed_branch_used_phase_correlation"] = int(used_phase_correlation)
        measurements["observed_branch_used_bbox_fallback"] = int(used_bbox_fallback)
        measurements["observed_branch_coverage_fraction"] = float(
            baseline_build.measurements.get("observed_coverage_fraction", coverage_fraction(baseline_build.coverage))
        )
        measurements["observed_branch_overlap_conflict_fraction"] = 0.0
        measurements["observed_branch_mean_gradient_gain"] = 0.0
        measurements["observed_branch_largest_component_fraction"] = connected_fraction
        measurements["observed_branch_offset_scale"] = 1.0
        measurements["observed_branch_visual_quality_passed"] = int(visual_quality_passed)
        measurements["observed_branch_gradient_energy"] = branch_gradient_energy
        measurements["observed_branch_baseline_gradient_energy"] = baseline_gradient_energy
        measurements["observed_branch_gradient_energy_ratio"] = gradient_energy_ratio
        artifacts.setdefault("observed_branch_axis_valid_frames", [])
        artifacts.setdefault("observed_branch_selected_frames", [])
        artifacts["observed_branch_rejected_frames"] = branch_rejected
        return ObservedSurfaceBuild(
            baseline_build.image,
            baseline_build.coverage,
            baseline_build.model,
            measurements,
            artifacts,
        )

    def _offsets(
        self,
        selected: list[_PreparedFrame],
        fragments: list[tuple[np.ndarray, np.ndarray]],
        config: UnwrapConfig,
        baseline_build: _SurfaceBuild,
    ) -> tuple[list[float], bool, bool, bool]:
        if len(selected) <= 1:
            return [0.0], False, False, False
        raw_steps = _angular_pixel_steps(selected, baseline_build.measurements)
        used_steps = len(raw_steps) == len(selected) - 1
        used_phase_correlation = False
        used_bbox_fallback = False
        if not used_steps:
            raw_steps = []
            for left, right, left_fragment, right_fragment in zip(
                selected[:-1],
                selected[1:],
                fragments[:-1],
                fragments[1:],
                strict=True,
            ):
                step = _phase_correlation_step(left_fragment, right_fragment)
                if step is not None:
                    raw_steps.append(step)
                    used_phase_correlation = True
                    continue
                raw_steps.append(_bbox_fallback_step(left.item.bbox, right.item.bbox, config.output_height))
                used_bbox_fallback = True
            used_phase_correlation = used_phase_correlation and not used_steps
        oriented = _orient_steps(raw_steps)
        offsets = [0.0]
        for step in oriented:
            offsets.append(offsets[-1] + step)
        return offsets, used_steps, used_phase_correlation, used_bbox_fallback


def _baseline_measurements(
    baseline: dict[str, float | int | str | list[float] | list[int]],
) -> dict[str, float | int | str | list[float] | list[int]]:
    measurements = dict(baseline)
    for key, value in baseline.items():
        measurements.setdefault(f"baseline_{key}", value)
    return measurements


def _prepare_frames(
    frames: list[AnalyzedFrame],
    branch_rejected: list[dict[str, float | int | str]],
) -> tuple[list[_PreparedFrame], int]:
    prepared: list[_PreparedFrame] = []
    mask_fallback_count = 0
    for item in frames:
        mask, used_mask_fallback = _publish_mask(item)
        if used_mask_fallback:
            mask_fallback_count += 1
        angle = _angle_to_vertical(mask)
        if angle is None:
            branch_rejected.append(
                {
                    "frame_index": item.frame.index,
                    "timestamp_seconds": item.frame.timestamp_seconds,
                    "reason": "observed_branch_axis_instability",
                }
            )
            continue
        prepared.append(_PreparedFrame(item, mask, angle))
    return prepared, mask_fallback_count


def _publish_mask(item: AnalyzedFrame) -> tuple[np.ndarray, bool]:
    publish_mask = item.publish_mask.copy()
    if not np.any(publish_mask) and item.core_mask is not None:
        publish_mask = item.core_mask.copy()
    original_publish = item.publish_mask.copy()
    if item.nuisance_mask is not None:
        reduced = cv2.bitwise_and(publish_mask, cv2.bitwise_not(item.nuisance_mask))
        original_area = int(np.count_nonzero(original_publish))
        if original_area > 0 and np.count_nonzero(reduced) < 0.5 * original_area:
            return original_publish, True
        publish_mask = reduced
    return publish_mask, False


def _angle_to_vertical(mask: np.ndarray) -> float | None:
    largest = _largest_component(mask)
    if largest is None:
        return None
    points = np.column_stack(np.nonzero(largest > 0)).astype(np.float32)
    if points.shape[0] < 5:
        return None
    points = points[:, ::-1].reshape(-1, 1, 2)
    box = cv2.boxPoints(cv2.minAreaRect(points))
    edges = (box[1] - box[0], box[2] - box[1])
    deviations: list[float] = []
    for edge in edges:
        angle = abs(float(np.degrees(np.arctan2(edge[1], edge[0]))))
        if angle > 90.0:
            angle = 180.0 - angle
        deviations.append(abs(90.0 - angle))
    return min(deviations, default=90.0)


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    component_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(component_mask):
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask)
    if count <= 1:
        return component_mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas) + 1)
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def _select_frames(
    frames: list[_PreparedFrame],
) -> tuple[list[_PreparedFrame], list[dict[str, float | int | str]]]:
    if not frames:
        return [], []
    selected = [frames[0]]
    rejected: list[dict[str, float | int | str]] = []
    observed_mask = _thumbnail_mask(frames[0])
    previous_thumbnail, previous_mask = analyzer_module._band_thumbnail(frames[0].item)
    for candidate in frames[1:]:
        thumbnail, mask = analyzer_module._band_thumbnail(candidate.item)
        new_mask_fraction = analyzer_module._new_mask_fraction(observed_mask, mask)
        band_difference = analyzer_module._band_difference(previous_thumbnail, previous_mask, thumbnail, mask)
        bbox_shift_ratio = analyzer_module._bbox_shift_ratio(selected[-1].item.bbox, candidate.item.bbox)
        keep = (
            new_mask_fraction >= 0.01
            or band_difference >= 0.03
            or bbox_shift_ratio >= 0.015
        )
        if keep:
            selected.append(candidate)
            observed_mask |= mask
            previous_thumbnail, previous_mask = thumbnail, mask
            continue
        rejected.append(
            {
                "frame_index": candidate.item.frame.index,
                "timestamp_seconds": candidate.item.frame.timestamp_seconds,
                "reason": "observed_branch_redundant_frame",
                "new_mask_fraction": round(float(new_mask_fraction), 6),
                "band_difference": round(float(band_difference), 6),
                "bbox_shift_ratio": round(float(bbox_shift_ratio), 6),
            }
        )
    return selected, rejected


def _thumbnail_mask(frame: _PreparedFrame) -> np.ndarray:
    _, thumbnail_mask = analyzer_module._band_thumbnail(frame.item)
    return thumbnail_mask.copy()


def _force_minimum_selection(frames: list[_PreparedFrame], minimum: int) -> list[_PreparedFrame]:
    if len(frames) <= minimum:
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, minimum)
    selected_indices: list[int] = []
    for index in indices:
        rounded = int(round(float(index)))
        if rounded not in selected_indices:
            selected_indices.append(rounded)
    if len(selected_indices) < minimum:
        for index in range(len(frames)):
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) == minimum:
                break
    selected_indices.sort()
    return [frames[index] for index in selected_indices]


def _normalized_fragment(item: AnalyzedFrame, mask: np.ndarray, height: int) -> tuple[np.ndarray, np.ndarray]:
    return normalized_wall(item.frame.image, mask, item.bbox, height)


def _angular_pixel_steps(
    selected: list[_PreparedFrame],
    measurements: dict[str, float | int | str | list[float] | list[int]],
) -> list[float]:
    angular_steps = measurements.get("angular_steps")
    pixels_per_radian = measurements.get("pixels_per_radian")
    if (
        not isinstance(angular_steps, list)
        or len(angular_steps) != len(selected) - 1
        or not isinstance(pixels_per_radian, (int, float))
        or not np.isfinite(pixels_per_radian)
        or float(pixels_per_radian) <= 0
    ):
        return []
    return [float(step) * float(pixels_per_radian) for step in angular_steps if isinstance(step, (int, float))]


def _phase_correlation_step(
    left_fragment: tuple[np.ndarray, np.ndarray],
    right_fragment: tuple[np.ndarray, np.ndarray],
) -> float | None:
    left_image, left_mask = central_band(*left_fragment, 0.55)
    right_image, right_mask = central_band(*right_fragment, 0.55)
    if not np.any(left_mask) or not np.any(right_mask):
        return None
    left = left_image.copy()
    right = right_image.copy()
    left[left_mask == 0] = 0
    right[right_mask == 0] = 0
    shift, response = horizontal_shift(left, right)
    if not np.isfinite(shift) or not np.isfinite(response):
        return None
    return float(-shift / 256.0 * max(left.shape[1], 1))


def _bbox_fallback_step(
    left_bbox: tuple[int, int, int, int],
    right_bbox: tuple[int, int, int, int],
    output_height: int,
) -> float:
    left_center = left_bbox[0] + left_bbox[2] * 0.5
    right_center = right_bbox[0] + right_bbox[2] * 0.5
    mean_height = max((left_bbox[3] + right_bbox[3]) * 0.5, 1.0)
    height_scale = output_height / mean_height
    return float(abs(right_center - left_center) * height_scale)


def _orient_steps(steps: list[float]) -> list[float]:
    if not steps:
        return []
    direction = 1.0 if sum(steps) >= 0 else -1.0
    return [max(float(step) * direction, 0.0) for step in steps]


def _offset_scale(
    analysis: Analysis,
    measurements: dict[str, float | int | str | list[float] | list[int]],
) -> float:
    quality_gate_passed = measurements.get("quality_gate_passed") == 1
    if analysis.kind is SurfaceKind.CYLINDRICAL and quality_gate_passed:
        return 1.0
    return 0.45


def _compose_mosaic(
    fragments: list[tuple[np.ndarray, np.ndarray]],
    offsets: list[float],
    canvas_height: int,
    canvas_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    raw_positions = [float(offset) for offset in offsets]
    widths = [fragment[0].shape[1] for fragment in fragments]
    minimum = min(raw_positions)
    span = max(position + width for position, width in zip(raw_positions, widths, strict=True)) - minimum
    scale = 1.0
    if span > canvas_width and span > 0:
        scale = max((canvas_width - 1) / span, 0.05)
    positions = [max(int(round((position - minimum) * scale)), 0) for position in raw_positions]
    canvas = np.zeros((canvas_height, canvas_width, 3), np.uint8)
    coverage = np.zeros((canvas_height, canvas_width), np.uint8)
    owner = np.zeros((canvas_height, canvas_width), np.uint16)
    overlap_map = np.zeros((canvas_height, canvas_width), dtype=bool)
    seam_scores: list[float] = []
    for frame_id, ((image, mask), left) in enumerate(zip(fragments, positions, strict=True), start=1):
        patch_image = image
        patch_mask = mask
        if scale != 1.0:
            target_width = max(8, int(round(image.shape[1] * scale)))
            patch_image = cv2.resize(image, (target_width, canvas_height), interpolation=cv2.INTER_AREA)
            patch_mask = cv2.resize(mask, (target_width, canvas_height), interpolation=cv2.INTER_NEAREST)
        if left >= canvas_width:
            continue
        right = min(left + patch_image.shape[1], canvas_width)
        source_right = right - left
        if source_right <= 0:
            continue
        patch_image = patch_image[:, :source_right]
        patch_mask = patch_mask[:, :source_right]
        region_mask = patch_mask > 0
        occupied = coverage[:, left:right] > 0
        overlap = region_mask & occupied
        overlap_map[:, left:right] |= overlap
        empty = region_mask & ~occupied
        replace = empty.copy()
        if np.any(overlap):
            seam = _best_vertical_seam(canvas[:, left:right], patch_image, overlap)
            if seam is not None:
                seam_column, seam_score = seam
                seam_scores.append(seam_score)
                replace |= region_mask & (np.arange(patch_image.shape[1])[None, :] >= seam_column)
        if np.any(empty):
            coverage[:, left:right][empty] = 255
        if np.any(replace):
            canvas[:, left:right][replace] = patch_image[replace]
            owner[:, left:right][replace] = frame_id
            coverage[:, left:right][region_mask] = 255
    occupied_pixels = max(int(np.count_nonzero(coverage)), 1)
    overlap_fraction = float(np.count_nonzero(overlap_map) / occupied_pixels)
    mean_seam_score = float(np.mean(seam_scores) / 255.0) if seam_scores else 0.0
    return canvas, coverage, owner, overlap_fraction, mean_seam_score


def _gradient_map(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    gradient[mask == 0] = 0.0
    return gradient


def _owner_score_map(mask: np.ndarray) -> np.ndarray:
    width = mask.shape[1]
    if width <= 0:
        return np.zeros_like(mask, dtype=np.float32)
    horizontal = np.minimum(np.arange(width) + 1, width - np.arange(width)).astype(np.float32)
    feather = np.clip(horizontal / max(width * 0.22, 1.0), 0.05, 1.0)
    score = np.repeat(feather[None, :], mask.shape[0], axis=0)
    score[mask == 0] = 0.0
    return score


def _best_vertical_seam(
    current: np.ndarray,
    candidate: np.ndarray,
    overlap: np.ndarray,
) -> tuple[int, float] | None:
    supported = np.count_nonzero(overlap, axis=0) > current.shape[0] * 0.25
    columns = np.flatnonzero(supported)
    if not columns.size:
        return None
    scores: list[float] = []
    for column in columns:
        mask = overlap[:, column]
        difference = np.mean(
            np.abs(current[:, column][mask].astype(np.float32) - candidate[:, column][mask].astype(np.float32))
        )
        scores.append(float(difference))
    best = int(np.argmin(scores))
    # Move the seam slightly into the candidate so the new fragment contributes
    # a coherent central region instead of only a thin tail.
    column = min(int(columns[best]) + 16, candidate.shape[1] - 1)
    return column, float(scores[best])


def _gradient_energy(image: np.ndarray, coverage: np.ndarray) -> float:
    mask = coverage > 0
    if not np.any(mask):
        return 0.0
    gradient = _gradient_map(image, np.where(mask, 255, 0).astype(np.uint8))
    return float(np.mean(gradient[mask]))


def _largest_component_fraction(coverage: np.ndarray) -> float:
    largest = _largest_component(coverage)
    total = int(np.count_nonzero(coverage))
    if largest is None or total == 0:
        return 0.0
    return float(np.count_nonzero(largest) / total)


def _frame_payload(frames: list[_PreparedFrame]) -> list[dict[str, float | int]]:
    return [
        {
            "frame_index": item.item.frame.index,
            "timestamp_seconds": item.item.frame.timestamp_seconds,
        }
        for item in frames
    ]


def _safe_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and np.isfinite(value):
        return max(int(value), 1)
    return default

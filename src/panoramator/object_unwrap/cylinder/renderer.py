from __future__ import annotations

import itertools
from collections.abc import Sequence

import cv2
import numpy as np

from ..analyzer import AnalyzedFrame
from ..gap_fill import interpolate_surface_gaps

# This is a source-selection confidence threshold, not a segmentation
# threshold.  Cylindrical artwork often contains narrow but important details
# (ears, handles, lettering) that occupy only a small part of a source column.
# Geometry validation remains strict; publication must not discard such
# details merely because the column is not mostly foreground.
_MIN_MASK_COLUMN_FRACTION = 0.30
_MAX_VERTICAL_INTERPOLATION_GAP = 24
_MAX_RELATIVE_COLUMN = 0.999
_OWNER_SMOOTHING_WINDOW = 31


def render_inverse_cylindrical_atlas(
    frames: Sequence[AnalyzedFrame],
    angles: Sequence[float],
    output_height: int,
    output_width: int,
    vertical_offsets: Sequence[float] | None = None,
    interpolate_gaps: bool = False,
    max_interpolation_gap_px: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Render one cylindrical atlas by inverse sampling from the best view.

    The renderer owns one source frame for every atlas column.  This keeps a
    surface detail in one temporal observation instead of averaging overlapping
    views, while the existing pose and publication layers remain responsible for
    deciding whether the result is trustworthy.
    """
    if not frames or len(frames) != len(angles):
        raise ValueError("frames and angles must be non-empty and have equal length")
    if output_height < 1 or output_width < 1:
        raise ValueError("output dimensions must be positive")

    input_frame_count = len(frames)
    frames, angles, vertical_offsets = _select_single_cycle(frames, angles, vertical_offsets)
    prepared, reference_radius, half_view = _prepare_frames(frames, angles, output_height, vertical_offsets)
    angle_array = np.asarray(angles, dtype=np.float64)
    sweep = float(np.ptp(angle_array))
    full_cycle = sweep >= 0.995 * 2.0 * np.pi
    if full_cycle:
        world_start = float(angle_array[0])
        world_span = 2.0 * np.pi
    else:
        world_start = float(np.min(angle_array) - half_view)
        world_span = sweep + 2.0 * half_view

    target_angles = world_start + (
        (np.arange(output_width, dtype=np.float64) + 0.5) / output_width
    ) * world_span
    best_score = np.zeros(output_width, dtype=np.float32)
    best_frame = np.full(output_width, -1, dtype=np.int16)

    for frame_index, (crop, crop_mask, angle, sharpness) in enumerate(prepared):
        local = angle - target_angles
        if full_cycle:
            local = (local + np.pi) % (2.0 * np.pi) - np.pi
        visible = np.abs(local) <= half_view
        source_x = crop.shape[1] * 0.5 + reference_radius * np.sin(local)
        valid_source = (source_x >= 0.0) & (source_x < crop.shape[1] - 1)
        valid = visible & valid_source
        if not np.any(valid):
            continue

        left = np.floor(source_x).astype(np.int32)
        fraction = (source_x - left).astype(np.float32)
        left = np.clip(left, 0, crop.shape[1] - 1)
        right = np.clip(left + 1, 0, crop.shape[1] - 1)
        mask_column = (
            crop_mask[:, left] * (1.0 - fraction[None, :])
            + crop_mask[:, right] * fraction[None, :]
        )
        mask_fraction = np.mean(mask_column, axis=0)
        relative = np.clip(
            (source_x - crop.shape[1] * 0.5) / max(reference_radius, 1.0),
            -_MAX_RELATIVE_COLUMN,
            _MAX_RELATIVE_COLUMN,
        )
        score = (
            mask_fraction
            * (1.0 - np.abs(relative) * 0.35)
            * np.cos(np.minimum(np.abs(local), np.pi * 0.5))
            * sharpness
        ).astype(np.float32)
        score[~valid | (mask_fraction < _MIN_MASK_COLUMN_FRACTION)] = 0.0
        replace = score > best_score
        best_score[replace] = score[replace]
        best_frame[replace] = frame_index

    best_frame = _smooth_owners(best_frame)
    result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    coverage = np.zeros((output_height, output_width), dtype=np.uint8)
    source_map = np.zeros((output_height, output_width), dtype=np.uint16)
    valid_columns = best_frame >= 0
    for target_x in np.flatnonzero(valid_columns):
        frame_index = int(best_frame[target_x])
        if frame_index < 0 or frame_index >= len(prepared):
            continue
        crop, crop_mask, angle, _ = prepared[frame_index]
        local_angle = float(angle - target_angles[target_x])
        if full_cycle:
            local_angle = (local_angle + np.pi) % (2.0 * np.pi) - np.pi
        source_x = crop.shape[1] * 0.5 + reference_radius * np.sin(local_angle)
        if not 0.0 <= source_x < crop.shape[1] - 1:
            continue
        left = int(np.floor(source_x))
        fraction = source_x - left
        right = min(left + 1, crop.shape[1] - 1)
        result[:, target_x] = (
            crop[:, left].astype(np.float32) * (1.0 - fraction)
            + crop[:, right].astype(np.float32) * fraction
        ).astype(np.uint8)
        # Once a source column has passed the geometric confidence test, keep
        # the complete image column.  A cylindrical atlas publishes a wall
        # band, not a per-pixel foreground cutout: ears, lettering, highlights
        # and visible background inside that band are texture and must survive.
        # The mask still controls which columns can own the atlas, so this
        # does not weaken trajectory validation or admit arbitrary frames.
        valid_pixels = np.ones(crop.shape[0], dtype=bool)
        coverage[valid_pixels, target_x] = 255
        source_map[valid_pixels, target_x] = np.uint16(frame_index + 1)

    result[coverage == 0] = 0
    vertical_gap_fill_pixels = _fill_internal_vertical_gaps(result, coverage, source_map)
    extended_gap_fill_pixels = 0
    if interpolate_gaps:
        extended_gap_fill_pixels = interpolate_surface_gaps(
            result,
            coverage,
            source_map,
            max_gap=max_interpolation_gap_px,
        )
    occupied_columns = np.flatnonzero(np.any(coverage > 0, axis=0))
    if len(occupied_columns) >= 2:
        ordered = np.sort(occupied_columns)
        for left, right in itertools.pairwise(ordered):
            gap = int(right - left)
            if gap <= 1:
                continue
            alpha = np.arange(1, gap, dtype=np.float32) / gap
            result[:, left + 1 : right] = (
                result[:, left, None, :].astype(np.float32) * (1.0 - alpha[None, :, None])
                + result[:, right, None, :].astype(np.float32) * alpha[None, :, None]
            ).astype(np.uint8)
            coverage[:, left + 1 : right] = 255
            source_map[:, left + 1 : right] = source_map[:, left, None]
    if len(prepared) >= 8 and np.any(coverage):
        smoothed = cv2.GaussianBlur(result, (9, 1), 0)
        result[coverage > 0] = smoothed[coverage > 0]
        result[coverage == 0] = 0
    # The temporal trajectory is solved in capture order; publication uses the
    # opposite horizontal convention so the first visible side is on the right.
    result = np.asarray(cv2.flip(result, 1))
    coverage = np.asarray(cv2.flip(coverage, 1))
    source_map = np.asarray(cv2.flip(source_map, 1))
    if full_cycle and np.any(coverage) and output_width > 16:
        result, coverage, source_map = _move_seam(result, coverage, source_map)

    occupied = coverage > 0
    boundary = occupied[:, 1:] & occupied[:, :-1] & (
        source_map[:, 1:] != source_map[:, :-1]
    )
    error = np.zeros_like(coverage, dtype=np.uint8)
    if np.any(boundary):
        difference = np.mean(
            np.abs(result[:, 1:].astype(np.int16) - result[:, :-1].astype(np.int16)),
            axis=2,
        ).astype(np.uint8)
        error[:, 1:][boundary] = difference[boundary]

    measurements: dict[str, float | int] = {
        "inverse_cylindrical_cycle_input_frames": input_frame_count,
        "inverse_cylindrical_cycle_selected_frames": len(frames),
        "inverse_cylindrical_full_cycle": int(full_cycle),
        "inverse_cylindrical_valid_columns": int(np.count_nonzero(np.any(occupied, axis=0))),
        "inverse_cylindrical_owner_transitions": int(np.count_nonzero(boundary)),
        "inverse_cylindrical_coverage_fraction": float(np.mean(occupied)),
        "inverse_cylindrical_world_span_radians": world_span,
        "inverse_cylindrical_vertical_gap_fill_pixels": vertical_gap_fill_pixels,
        "inverse_cylindrical_extended_gap_fill_pixels": extended_gap_fill_pixels,
        "inverse_cylindrical_interpolate_gaps": int(interpolate_gaps),
        "inverse_cylindrical_full_column_publication": 1,
    }
    return result, coverage, source_map, error, measurements


def render_adaptive_slit_cylindrical_atlas(
    frames: Sequence[AnalyzedFrame],
    angles: Sequence[float],
    output_height: int,
    output_width: int,
    vertical_offsets: Sequence[float] | None = None,
    interpolate_gaps: bool = False,
    max_interpolation_gap_px: int = 96,
    blend_width_factor: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Render a cylindrical unwrap with one adaptive slit per camera step.

    A broad-frame renderer lets several neighbouring views compete for the
    same surface region.  That is attractive for coverage, but it can repeat
    a character or create a full-height vertical band when the pose estimate
    is only slightly wrong.  The donor implementation assigns each target
    angle to the interval halfway to its neighbours, so a frame contributes
    only its own narrow slit and at most a small boundary blend.
    """
    if not frames or len(frames) != len(angles):
        raise ValueError("frames and angles must be non-empty and have equal length")
    if output_height < 1 or output_width < 1:
        raise ValueError("output dimensions must be positive")
    if blend_width_factor <= 0:
        raise ValueError("blend_width_factor must be > 0")

    input_frame_count = len(frames)
    frames, angles, vertical_offsets = _select_single_cycle(frames, angles, vertical_offsets)
    prepared, reference_radius, _ = _prepare_frames(frames, angles, output_height, vertical_offsets)
    angle_array = np.asarray(angles, dtype=np.float64)
    full_cycle = float(np.ptp(angle_array)) >= 0.995 * 2.0 * np.pi
    if len(angle_array) > 1:
        steps = np.abs(np.diff(angle_array))
        steps = steps[steps > 1e-4]
        median_step = float(np.median(steps)) if steps.size else 2.0 * np.pi / len(angle_array)
    else:
        median_step = 2.0 * np.pi
    direction = float(np.sign(np.median(np.diff(angle_array)))) if len(angle_array) > 1 else 1.0
    if direction == 0.0:
        direction = 1.0
    previous_gaps, following_gaps = _slit_neighbour_gaps(angle_array, full_cycle, median_step)
    if full_cycle:
        world_start = float(np.min(angle_array))
        world_span = 2.0 * np.pi
    else:
        world_start = float(np.min(angle_array) - median_step * 0.5)
        world_span = float(min(2.0 * np.pi, np.ptp(angle_array) + median_step))

    # Normalize only the low-frequency exposure drift.  This preserves the
    # printed artwork while preventing neighbouring slits from becoming
    # obvious vertical brightness bands.
    luminances: list[float] = []
    for crop, mask, _angle, _sharpness in prepared:
        valid = crop[mask > 0.30]
        if valid.size:
            luminances.append(float(np.median(cv2.cvtColor(valid[None, ...], cv2.COLOR_BGR2GRAY))))
        else:
            luminances.append(0.0)
    target_luminance = float(np.median([value for value in luminances if value > 0])) if any(luminances) else 0.0
    normalized: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    for (crop, mask, angle, sharpness), luminance in zip(prepared, luminances, strict=True):
        if target_luminance > 0 and luminance > 0:
            gain = float(np.clip(target_luminance / luminance, 0.78, 1.28))
            if abs(gain - 1.0) > 0.01:
                crop = np.clip(crop.astype(np.float32) * gain, 0.0, 255.0).astype(np.uint8)
        normalized.append((crop, mask, angle, sharpness))
    prepared = normalized
    prepared_angles = np.asarray([item[2] for item in prepared], dtype=np.float64)

    result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    coverage = np.zeros((output_height, output_width), dtype=np.uint8)
    source_map = np.zeros((output_height, output_width), dtype=np.uint16)
    valid_columns = np.zeros(output_width, dtype=bool)
    owner_columns = np.full(output_width, -1, dtype=np.int16)
    for target_x in range(output_width):
        target_angle = world_start + (target_x + 0.5) / output_width * world_span
        if full_cycle:
            distance = np.abs((prepared_angles - target_angle + np.pi) % (2.0 * np.pi) - np.pi)
        else:
            distance = np.abs(prepared_angles - target_angle)
        candidates: list[int] = []
        for index in np.argsort(distance):
            frame_angle = float(prepared_angles[int(index)])
            if _slit_owns_target(
                target_angle,
                frame_angle,
                float(previous_gaps[int(index)]),
                float(following_gaps[int(index)]),
                direction,
                full_cycle,
            ):
                candidates.append(int(index))

        samples: list[tuple[np.ndarray, float, int]] = []
        for frame_index in candidates:
            crop, mask, angle, _sharpness = prepared[frame_index]
            if full_cycle:
                surface_angle = (angle - target_angle + np.pi) % (2.0 * np.pi) - np.pi
            else:
                surface_angle = angle - target_angle
            source_x = crop.shape[1] * 0.5 + reference_radius * np.sin(surface_angle)
            if not 0.0 <= source_x < crop.shape[1] - 1:
                continue
            left = int(np.floor(source_x))
            fraction = float(source_x - left)
            right = min(left + 1, crop.shape[1] - 1)
            weight = mask[:, left] * (1.0 - fraction) + mask[:, right] * fraction
            if float(np.mean(weight)) < _MIN_MASK_COLUMN_FRACTION:
                continue
            sample = crop[:, left, :].astype(np.float32) * (1.0 - fraction) + crop[:, right, :].astype(np.float32) * fraction
            samples.append((sample, float(distance[frame_index]), frame_index))
            if len(samples) >= 2:
                break
        if not samples:
            continue
        if len(samples) == 1:
            result[:, target_x, :] = samples[0][0].astype(np.uint8)
            owner_columns[target_x] = samples[0][2]
        else:
            blend_width = max(median_step * blend_width_factor, 1e-3)
            weights = np.asarray(
                [max(0.0, 1.0 - distance_value / blend_width) for _, distance_value, _ in samples],
                dtype=np.float32,
            )
            if float(weights.sum()) <= 0.0:
                result[:, target_x, :] = samples[0][0].astype(np.uint8)
                owner_columns[target_x] = samples[0][2]
            else:
                weights /= weights.sum()
                result[:, target_x, :] = np.sum(
                    np.stack([sample * float(weight) for (sample, _, _), weight in zip(samples, weights, strict=True)]),
                    axis=0,
                ).astype(np.uint8)
                owner_columns[target_x] = samples[0][2]
        valid_columns[target_x] = True
        coverage[:, target_x] = 255
        source_map[:, target_x] = np.uint16(owner_columns[target_x] + 1)

    vertical_gap_fill_pixels = _fill_internal_vertical_gaps(result, coverage, source_map)
    extended_gap_fill_pixels = 0
    if interpolate_gaps:
        extended_gap_fill_pixels = interpolate_surface_gaps(
            result, coverage, source_map, max_gap=max_interpolation_gap_px
        )
    result[coverage == 0] = 0
    if np.any(valid_columns):
        result = np.asarray(cv2.GaussianBlur(result, (9, 1), 0))
        result[coverage == 0] = 0
    result = np.asarray(cv2.flip(result, 1))
    coverage = np.asarray(cv2.flip(coverage, 1))
    source_map = np.asarray(cv2.flip(source_map, 1))
    if full_cycle and np.any(coverage) and output_width > 16:
        result, coverage, source_map = _move_seam(result, coverage, source_map)

    occupied = coverage > 0
    boundary = occupied[:, 1:] & occupied[:, :-1] & (source_map[:, 1:] != source_map[:, :-1])
    error = np.zeros_like(coverage, dtype=np.uint8)
    if np.any(boundary):
        difference = np.mean(np.abs(result[:, 1:].astype(np.int16) - result[:, :-1].astype(np.int16)), axis=2).astype(np.uint8)
        error[:, 1:][boundary] = difference[boundary]
    measurements: dict[str, float | int] = {
        "inverse_cylindrical_cycle_input_frames": input_frame_count,
        "inverse_cylindrical_cycle_selected_frames": len(frames),
        "inverse_cylindrical_full_cycle": int(full_cycle),
        "inverse_cylindrical_valid_columns": int(np.count_nonzero(np.any(occupied, axis=0))),
        "inverse_cylindrical_owner_transitions": int(np.count_nonzero(boundary)),
        "inverse_cylindrical_coverage_fraction": float(np.mean(occupied)),
        "inverse_cylindrical_world_span_radians": world_span,
        "inverse_cylindrical_vertical_gap_fill_pixels": vertical_gap_fill_pixels,
        "inverse_cylindrical_extended_gap_fill_pixels": extended_gap_fill_pixels,
        "inverse_cylindrical_interpolate_gaps": int(interpolate_gaps),
        "inverse_cylindrical_adaptive_slit": 1,
        "inverse_cylindrical_median_step_radians": median_step,
        "inverse_cylindrical_blend_width_factor": blend_width_factor,
    }
    return result, coverage, source_map, error, measurements


def _select_single_cycle(
    frames: Sequence[AnalyzedFrame],
    angles: Sequence[float],
    vertical_offsets: Sequence[float] | None,
) -> tuple[list[AnalyzedFrame], list[float], list[float] | None]:
    """Keep at most one chronological revolution before circular rendering.

    A hand-held orbit can overshoot 360 degrees by only a few degrees.  Treating
    that tail as another valid full-cycle observation is enough to put the last
    character back at the beginning when the renderer uses modulo arithmetic.
    The donor adaptive-slit renderer removes this tail before mapping columns;
    the inverse cylindrical path must apply the same invariant.
    """
    if len(angles) < 2 or float(np.ptp(np.asarray(angles, dtype=np.float64))) <= 2.0 * np.pi:
        return list(frames), list(angles), list(vertical_offsets) if vertical_offsets is not None else None

    values = np.asarray(angles, dtype=np.float64)
    direction = float(np.median(np.diff(values)))
    if direction < 0.0:
        upper = float(values[0])
        lower = upper - 2.0 * np.pi
        selected = [index for index, angle in enumerate(values) if lower < angle <= upper]
    else:
        lower = float(values[0])
        upper = lower + 2.0 * np.pi
        selected = [index for index, angle in enumerate(values) if lower <= angle < upper]

    if len(selected) < 3:
        selected = list(range(max(0, len(values) - 3), len(values)))
    selected_frames = [frames[index] for index in selected]
    selected_angles = [float(values[index]) for index in selected]
    selected_offsets = (
        [float(vertical_offsets[index]) for index in selected]
        if vertical_offsets is not None and len(vertical_offsets) == len(frames)
        else list(vertical_offsets) if vertical_offsets is not None else None
    )
    return selected_frames, selected_angles, selected_offsets


def _slit_neighbour_gaps(
    angles: np.ndarray, full_cycle: bool, median_step: float
) -> tuple[np.ndarray, np.ndarray]:
    count = len(angles)
    previous = np.full(count, median_step, dtype=np.float64)
    following = np.full(count, median_step, dtype=np.float64)
    if count < 2:
        return previous, following
    steps = np.abs(np.diff(angles))
    previous[1:] = np.maximum(steps, 1e-4)
    following[:-1] = np.maximum(steps, 1e-4)
    if full_cycle:
        wrap_gap = max(2.0 * np.pi - float(np.ptp(angles)), 1e-4)
        previous[0] = wrap_gap
        following[-1] = wrap_gap
    return previous, following


def _slit_owns_target(
    target_angle: float,
    frame_angle: float,
    previous_gap: float,
    following_gap: float,
    direction: float,
    full_cycle: bool,
) -> bool:
    signed_distance = (target_angle - frame_angle) * direction
    if full_cycle:
        forward_distance = signed_distance % (2.0 * np.pi)
        return bool(
            forward_distance <= following_gap * 0.5 + 1e-6
            or forward_distance >= 2.0 * np.pi - previous_gap * 0.5 - 1e-6
        )
    return bool(
        -previous_gap * 0.5 - 1e-6
        <= signed_distance
        <= following_gap * 0.5 + 1e-6
    )


def _prepare_frames(
    frames: Sequence[AnalyzedFrame],
    angles: Sequence[float],
    output_height: int,
    vertical_offsets: Sequence[float] | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float, float]], float, float]:
    widths = np.asarray([item.bbox[2] for item in frames], dtype=np.float32)
    heights = np.asarray([item.bbox[3] for item in frames], dtype=np.float32)
    fixed_height = max(2, int(np.ceil(np.percentile(heights, 95) + 4.0)))
    fixed_width = max(16, round(float(np.median(widths)) * 0.84))
    center_x = float(np.median([item.bbox[0] + item.bbox[2] * 0.5 for item in frames]))
    center_y = float(np.median([item.bbox[1] + item.bbox[3] * 0.5 for item in frames]))
    # The radius is estimated from the detected cylinder width.  The crop is
    # intentionally narrower than that estimate, so deriving the radius from
    # the crop would make the crop appear wider than a half-cylinder and would
    # over-expand the angular view of every frame.
    reference_radius = max(float(np.median(widths)) * 0.48, 2.0)
    half_view = float(np.arcsin(np.clip((fixed_width * 0.5) / reference_radius, -0.999, 0.999)))
    prepared: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    sharpness_values = np.asarray([max(item.sharpness, 1.0) for item in frames], dtype=np.float32)
    sharpness_scale = max(float(np.median(sharpness_values)), 1.0)
    for frame_index, (item, angle, sharpness_value) in enumerate(zip(frames, angles, sharpness_values, strict=True)):
        frame_height, frame_width = item.frame.image.shape[:2]
        x = round(center_x - fixed_width * 0.5)
        y = round(center_y - fixed_height * 0.5)
        x = max(0, min(x, max(frame_width - 2, 0)))
        y = max(0, min(y, max(frame_height - 2, 0)))
        width = min(fixed_width, frame_width - x)
        height = min(fixed_height, frame_height - y)
        crop = item.frame.image[y : y + height, x : x + width]
        # ``publish_mask`` is intentionally conservative because it is also
        # used by geometry/quality diagnostics.  It is not a safe pixel
        # deletion mask: narrow printed details can be outside its stable
        # vertical run.  The cylindrical renderer uses the union with the
        # already detected geometry, still clipped to the object bbox, so it
        # preserves observed artwork without admitting arbitrary frame
        # background.
        mask = np.maximum(
            item.publish_mask[y : y + height, x : x + width],
            item.geometry_mask[y : y + height, x : x + width],
        )
        if crop.size == 0 or mask.size == 0:
            continue
        if vertical_offsets is not None and frame_index < len(vertical_offsets):
            offset = float(vertical_offsets[frame_index])
            if abs(offset) > 0.25:
                transform = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, -offset]], dtype=np.float32)
                crop = np.asarray(
                    cv2.warpAffine(crop, transform, (int(crop.shape[1]), int(crop.shape[0])), borderMode=cv2.BORDER_REPLICATE)
                )
                mask = np.asarray(
                    cv2.warpAffine(
                        mask,
                        transform,
                        (int(mask.shape[1]), int(mask.shape[0])),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                )
        crop = np.asarray(cv2.resize(crop, (fixed_width, output_height), interpolation=cv2.INTER_AREA))
        mask = np.asarray(cv2.resize(mask, (fixed_width, output_height), interpolation=cv2.INTER_NEAREST))
        sharpness = float(np.clip(0.65 + np.sqrt(float(sharpness_value) / sharpness_scale) * 0.25, 0.65, 1.25))
        prepared.append((crop, mask.astype(np.float32) / 255.0, float(angle), sharpness))
    if not prepared:
        raise ValueError("No usable cylindrical frame crops were produced")
    return prepared, reference_radius, half_view


def _fill_internal_vertical_gaps(
    image: np.ndarray,
    coverage: np.ndarray,
    source_map: np.ndarray,
    max_gap: int = _MAX_VERTICAL_INTERPOLATION_GAP,
) -> int:
    """Interpolate short holes between real samples in the same atlas column.

    Large unobserved margins remain transparent.  Only gaps bracketed by two
    observed pixels in one column are filled, which makes this a conservative
    image-completion step rather than a way to invent an unobserved surface.
    """
    filled = 0
    for column in range(image.shape[1]):
        rows = np.flatnonzero(coverage[:, column] > 0)
        if rows.size < 2:
            continue
        for top, bottom in itertools.pairwise(rows):
            gap = int(bottom - top - 1)
            if gap <= 0 or gap > max_gap:
                continue
            alpha = np.arange(1, gap + 1, dtype=np.float32) / (gap + 1)
            image[top + 1 : bottom, column] = (
                image[top, column].astype(np.float32) * (1.0 - alpha[:, None])
                + image[bottom, column].astype(np.float32) * alpha[:, None]
            ).astype(np.uint8)
            coverage[top + 1 : bottom, column] = 255
            source_map[top + 1 : bottom, column] = source_map[top, column]
            filled += gap
    return filled


def _smooth_owners(owners: np.ndarray) -> np.ndarray:
    valid = owners >= 0
    if np.count_nonzero(valid) < 5:
        return owners
    values = owners.copy()
    values[~valid] = 0
    window = min(101, len(values) if len(values) % 2 else len(values) - 1)
    if window < 3:
        return owners
    smoothed = cv2.medianBlur(values.astype(np.uint8)[None, :], window)[0].astype(np.int16)
    result = owners.copy()
    result[valid] = smoothed[valid]
    allowed = {int(value) for value in owners[valid]}
    invalid_smoothed = valid & np.asarray(
        [int(value) not in allowed for value in result], dtype=bool
    )
    result[invalid_smoothed] = owners[invalid_smoothed]
    return result


def _move_seam(
    image: np.ndarray, coverage: np.ndarray, source_map: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    energy = np.mean(np.abs(gray - np.roll(gray, -1, axis=1)), axis=0)
    occupied = np.any(coverage > 0, axis=0)
    both_valid = occupied & np.roll(occupied, -1)
    if np.any(both_valid):
        energy[~both_valid] = np.inf
    shift = int(np.argmin(energy)) + 1
    return (
        np.roll(image, -shift, axis=1),
        np.roll(coverage, -shift, axis=1),
        np.roll(source_map, -shift, axis=1),
    )

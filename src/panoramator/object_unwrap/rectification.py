from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class QualityGateResult:
    passed: bool
    boundary_fraction: float
    mean_boundary_error: float
    severe_boundary_fraction: float
    severe_boundary_footprint: float
    weighted_seam_risk: float
    weighted_seam_footprint: float
    seam_risk_map: np.ndarray

    @property
    def measurements(self) -> dict[str, float | int]:
        return {
            "quality_gate_passed": int(self.passed),
            "quality_gate_boundary_fraction": self.boundary_fraction,
            "quality_gate_mean_boundary_error": self.mean_boundary_error,
            "quality_gate_severe_boundary_fraction": self.severe_boundary_fraction,
            "quality_gate_severe_boundary_footprint": self.severe_boundary_footprint,
            "quality_gate_weighted_seam_risk": self.weighted_seam_risk,
            "quality_gate_weighted_seam_footprint": self.weighted_seam_footprint,
        }


@dataclass(slots=True)
class StripEstimate:
    top: np.ndarray
    axis: np.ndarray
    bottom: np.ndarray
    valid_columns: np.ndarray
    column_fraction: float
    median_band_height: float
    max_axis_step: float
    max_top_step: float
    max_bottom_step: float

    @property
    def measurements(self) -> dict[str, float | int]:
        return {
            "rectification_column_fraction": self.column_fraction,
            "rectification_median_band_height": self.median_band_height,
            "rectification_max_axis_step": self.max_axis_step,
            "rectification_max_top_step": self.max_top_step,
            "rectification_max_bottom_step": self.max_bottom_step,
        }


def evaluate_mosaic_quality(
    image: np.ndarray,
    coverage: np.ndarray,
    source_map: np.ndarray,
    error_map: np.ndarray,
    max_mean_boundary_error: float,
    max_severe_boundary_fraction: float,
    severe_error_threshold: float,
    max_severe_boundary_footprint: float,
) -> QualityGateResult:
    occupied = coverage > 0
    vertical_boundaries = occupied[:, 1:] & occupied[:, :-1] & (source_map[:, 1:] != source_map[:, :-1])
    horizontal_boundaries = occupied[1:, :] & occupied[:-1, :] & (source_map[1:, :] != source_map[:-1, :])
    boundary_mask = np.zeros_like(occupied, dtype=bool)
    boundary_mask[:, 1:] |= vertical_boundaries
    boundary_mask[:, :-1] |= vertical_boundaries
    boundary_mask[1:, :] |= horizontal_boundaries
    boundary_mask[:-1, :] |= horizontal_boundaries
    boundary_pixels = int(boundary_mask.sum())
    occupied_pixels = int(occupied.sum())
    if boundary_pixels == 0 or occupied_pixels == 0:
        return QualityGateResult(True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.zeros_like(coverage, dtype=np.uint8))
    errors = error_map[boundary_mask].astype(np.float32)
    mean_boundary_error = float(np.mean(errors))
    severe_boundary_fraction = float(np.count_nonzero(errors >= severe_error_threshold) / max(boundary_pixels, 1))
    boundary_fraction = float(boundary_pixels / occupied_pixels)
    severe_boundary_footprint = boundary_fraction * severe_boundary_fraction
    seam_risk_map, weighted_seam_risk = _weighted_seam_risk_map(
        image,
        occupied,
        boundary_mask,
        error_map,
        severe_error_threshold,
    )
    weighted_seam_footprint = boundary_fraction * weighted_seam_risk
    return QualityGateResult(
        passed=(
            mean_boundary_error <= max_mean_boundary_error
            and severe_boundary_footprint <= max_severe_boundary_footprint
            and weighted_seam_footprint <= max_severe_boundary_footprint
            and (
                severe_boundary_fraction <= max_severe_boundary_fraction
                or weighted_seam_footprint <= max_severe_boundary_footprint * 0.75
            )
        ),
        boundary_fraction=boundary_fraction,
        mean_boundary_error=mean_boundary_error,
        severe_boundary_fraction=severe_boundary_fraction,
        severe_boundary_footprint=severe_boundary_footprint,
        weighted_seam_risk=weighted_seam_risk,
        weighted_seam_footprint=weighted_seam_footprint,
        seam_risk_map=seam_risk_map,
    )


def estimate_strip(
    coverage: np.ndarray,
    min_column_fraction: float,
    smoothing_window: int,
    max_axis_step: float,
) -> StripEstimate | None:
    occupied = coverage > 0
    rows, columns = coverage.shape
    top = np.full(columns, np.nan, dtype=np.float32)
    bottom = np.full(columns, np.nan, dtype=np.float32)
    axis = np.full(columns, np.nan, dtype=np.float32)
    valid_columns = np.zeros(columns, dtype=bool)
    for column in range(columns):
        ys = np.flatnonzero(occupied[:, column])
        if ys.size < max(8, rows // 12):
            continue
        top[column] = float(ys[0])
        bottom[column] = float(ys[-1])
        axis[column] = float(np.mean(ys))
        valid_columns[column] = True
    valid_columns = _filter_support_columns(top, bottom, valid_columns, rows)
    column_fraction = float(np.count_nonzero(valid_columns) / max(columns, 1))
    if column_fraction < min_column_fraction:
        return None
    top = _smooth_profile(top, valid_columns, smoothing_window)
    bottom = _smooth_profile(bottom, valid_columns, smoothing_window)
    axis = _smooth_profile(axis, valid_columns, smoothing_window)
    axis = np.clip(axis, top + 1.0, bottom - 1.0)
    band_height = bottom - top
    if not np.all(np.isfinite(band_height[valid_columns])) or float(np.median(band_height[valid_columns])) < max(12.0, rows * 0.1):
        return None
    smoothed_axis_step = float(np.max(np.abs(np.diff(axis[valid_columns])))) if np.count_nonzero(valid_columns) > 1 else 0.0
    if smoothed_axis_step > max_axis_step:
        return None
    top_step = float(np.max(np.abs(np.diff(top[valid_columns])))) if np.count_nonzero(valid_columns) > 1 else 0.0
    bottom_step = float(np.max(np.abs(np.diff(bottom[valid_columns])))) if np.count_nonzero(valid_columns) > 1 else 0.0
    return StripEstimate(
        top=top,
        axis=axis,
        bottom=bottom,
        valid_columns=valid_columns,
        column_fraction=column_fraction,
        median_band_height=float(np.median(band_height[valid_columns])),
        max_axis_step=smoothed_axis_step,
        max_top_step=top_step,
        max_bottom_step=bottom_step,
    )


def rectify_mosaic(
    image: np.ndarray,
    coverage: np.ndarray,
    source_map: np.ndarray,
    error_map: np.ndarray,
    strip: StripEstimate,
    output_height: int,
    output_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = image.shape[1]
    scale = output_height / max(strip.median_band_height, 1.0)
    target_width = max(1, round(width * scale))
    if output_width is not None:
        target_width = min(target_width, output_width)
    target_axis = (output_height - 1) * 0.5
    columns = np.linspace(0, width - 1, target_width, dtype=np.float32)
    map_x = np.repeat(columns[None, :], output_height, axis=0)
    map_y = np.empty((output_height, target_width), dtype=np.float32)
    upper = max(target_axis, 1.0)
    lower = max((output_height - 1) - target_axis, 1.0)
    top = strip.top.astype(np.float32)
    axis = strip.axis.astype(np.float32)
    bottom = strip.bottom.astype(np.float32)
    for row in range(output_height):
        if row <= target_axis:
            alpha = row / upper
            map_y[row] = np.interp(columns, np.arange(width, dtype=np.float32), top + (axis - top) * alpha)
        else:
            alpha = (row - target_axis) / lower
            map_y[row] = np.interp(columns, np.arange(width, dtype=np.float32), axis + (bottom - axis) * alpha)
    rectified = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rectified_coverage = cv2.remap(coverage, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    rectified_source = cv2.remap(source_map, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    rectified_error = cv2.remap(error_map, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rectified_coverage = np.where(rectified_coverage > 0, 255, 0).astype(np.uint8)
    rectified_source = rectified_source.astype(source_map.dtype, copy=False)
    rectified_error = rectified_error.astype(error_map.dtype, copy=False)
    rectified[rectified_coverage == 0] = 0
    rectified_source[rectified_coverage == 0] = 0
    rectified_error[rectified_coverage == 0] = 0
    return rectified, rectified_coverage, rectified_source, rectified_error


def _smooth_profile(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    result = values.astype(np.float32, copy=True)
    if not np.any(valid):
        return result
    indices = np.arange(len(values), dtype=np.float32)
    result[~valid] = np.interp(indices[~valid], indices[valid], result[valid])
    kernel = max(3, window | 1)
    result = cv2.GaussianBlur(result[None, :], (kernel, 1), 0).reshape(-1)
    return result.astype(np.float32, copy=False)


def _filter_support_columns(top: np.ndarray, bottom: np.ndarray, valid: np.ndarray, rows: int) -> np.ndarray:
    filtered = valid.copy()
    if np.count_nonzero(filtered) < 3:
        return filtered
    heights = bottom - top + 1.0
    median_height = float(np.median(heights[filtered]))
    if median_height <= 0:
        return np.zeros_like(filtered)
    smooth_top = _smooth_profile(top, filtered, 9)
    smooth_bottom = _smooth_profile(bottom, filtered, 9)
    top_residual = np.abs(top - smooth_top)
    bottom_residual = np.abs(bottom - smooth_bottom)
    top_step = np.zeros_like(top, dtype=np.float32)
    bottom_step = np.zeros_like(bottom, dtype=np.float32)
    top_delta = np.abs(np.diff(smooth_top))
    bottom_delta = np.abs(np.diff(smooth_bottom))
    top_step[1:] = np.maximum(top_step[1:], top_delta)
    top_step[:-1] = np.maximum(top_step[:-1], top_delta)
    bottom_step[1:] = np.maximum(bottom_step[1:], bottom_delta)
    bottom_step[:-1] = np.maximum(bottom_step[:-1], bottom_delta)
    min_height = max(8.0, median_height * 0.4)
    residual_tolerance = max(3.0, rows * 0.06, median_height * 0.18)
    step_tolerance = max(3.0, median_height * 0.16)
    filtered &= heights >= min_height
    filtered &= top_residual <= residual_tolerance
    filtered &= bottom_residual <= residual_tolerance
    filtered &= top_step <= step_tolerance
    filtered &= bottom_step <= step_tolerance
    return filtered


def _weighted_seam_risk_map(
    image: np.ndarray,
    occupied: np.ndarray,
    boundary_mask: np.ndarray,
    error_map: np.ndarray,
    severe_error_threshold: float,
) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    detail = np.zeros_like(gradient, dtype=np.float32)
    occupied_gradient = gradient[occupied]
    if occupied_gradient.size:
        scale = max(float(np.percentile(occupied_gradient, 95)), 1.0)
        detail[occupied] = np.clip(occupied_gradient / scale, 0.0, 1.0)
    boundary_float = boundary_mask.astype(np.float32)
    severe_map = ((error_map.astype(np.float32) >= severe_error_threshold) & boundary_mask).astype(np.float32)
    local_boundary_density = cv2.GaussianBlur(boundary_float, (0, 0), sigmaX=6.0, sigmaY=6.0)
    local_severe_density = cv2.GaussianBlur(severe_map, (0, 0), sigmaX=10.0, sigmaY=10.0)
    error_weight = np.clip(error_map.astype(np.float32) / max(float(severe_error_threshold), 1.0), 0.0, 2.0)
    risk = error_weight * (0.35 + 0.65 * detail) * (0.4 + 0.6 * np.clip(local_boundary_density, 0.0, 1.0))
    risk *= 0.5 + 0.5 * np.clip(local_severe_density * 4.0, 0.0, 1.0)
    risk *= boundary_float
    boundary_values = risk[boundary_mask]
    weighted_seam_risk = float(np.mean(boundary_values)) if boundary_values.size else 0.0
    scale = max(float(np.max(boundary_values)) if boundary_values.size else 0.0, 1e-6)
    risk_map = np.clip(risk / scale * 255.0, 0, 255).astype(np.uint8)
    return risk_map, weighted_seam_risk

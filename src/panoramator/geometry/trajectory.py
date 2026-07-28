from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panoramator.config.models import PanoramaConfig


@dataclass(frozen=True, slots=True)
class StabilizedTrajectory:
    homographies: list[np.ndarray]
    diagnostics: dict[str, list[float]]


def stabilize_rotation_trajectory(
    pairwise_homographies: list[np.ndarray], config: PanoramaConfig
) -> StabilizedTrajectory:
    """Smooth only vertical drift, roll and tiny scale noise in a rotation chain.

    Horizontal displacement is deliberately retained: it represents the extent of
    the panorama rather than camera shake.
    """
    if not pairwise_homographies:
        return StabilizedTrajectory([np.eye(3)], {})
    raw = [_similarity_parameters(matrix) for matrix in pairwise_homographies]
    cumulative = np.cumsum(np.asarray(raw), axis=0)
    smoothed = cumulative.copy()
    window = config.trajectory_smoothing_window
    for column in (1, 2, 3):  # vertical shift, roll, log-scale
        smoothed[:, column] = _moving_average(cumulative[:, column], window)
    # A global similarity correction is intentionally weak; absolute scale changes
    # larger than the configured allowance are evidence, not stabilisation.
    max_log = float(np.log1p(config.max_rotation_scale_correction))
    smoothed[:, 3] = np.clip(smoothed[:, 3], cumulative[:, 3] - max_log, cumulative[:, 3] + max_log)
    globals_ = [np.eye(3, dtype=np.float64)]
    for tx, ty, angle, log_scale in smoothed:
        globals_.append(_similarity_matrix(tx, ty, angle, log_scale))
    return StabilizedTrajectory(
        globals_,
        {
            "raw_vertical_shift": cumulative[:, 1].tolist(),
            "smoothed_vertical_shift": smoothed[:, 1].tolist(),
            "raw_roll_degrees": np.degrees(cumulative[:, 2]).tolist(),
            "smoothed_roll_degrees": np.degrees(smoothed[:, 2]).tolist(),
            "raw_scale": np.exp(cumulative[:, 3]).tolist(),
            "smoothed_scale": np.exp(smoothed[:, 3]).tolist(),
        },
    )


def _similarity_parameters(matrix: np.ndarray) -> tuple[float, float, float, float]:
    linear = matrix[:2, :2]
    scale = max(1e-8, float(np.sqrt(abs(np.linalg.det(linear)))))
    angle = float(np.arctan2(linear[1, 0], linear[0, 0]))
    return float(matrix[0, 2]), float(matrix[1, 2]), angle, float(np.log(scale))


def _similarity_matrix(tx: float, ty: float, angle: float, log_scale: float) -> np.ndarray:
    scale = float(np.exp(log_scale))
    cos, sin = np.cos(angle) * scale, np.sin(angle) * scale
    return np.array([[cos, -sin, tx], [sin, cos, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 2:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(2 * radius + 1) / (2 * radius + 1), mode="valid")

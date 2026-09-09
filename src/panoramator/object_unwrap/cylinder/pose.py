from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CylinderTrajectory:
    angles: list[float]
    steps: list[float]
    accepted_pairs: int
    residual_radians: float
    sweep_radians: float
    repeated_observation: bool
    accepted: list[bool]
    rejection_reasons: list[str]


def select_cylindrical_keyframes(
    frames: Sequence[object], motions: Sequence[float], inliers: Sequence[int]
) -> list[int]:
    """Collapse locally stationary runs while retaining the sharpest view.

    The orbit is first measured densely, then pauses are reduced before the
    final pairwise solve.  This keeps a pause from contributing descriptor
    jitter to the total angular sweep while preserving a real slow orbit.
    """
    count = len(frames)
    if count <= 2 or len(motions) != count - 1:
        return list(range(count))
    steps = np.asarray(motions, dtype=np.float32)
    pause_edges = np.abs(steps) <= 0.03
    window_size = min(5, len(steps))
    for start in range(max(0, len(steps) - window_size + 1)):
        window = steps[start : start + window_size]
        support = np.asarray(inliers[start : start + window_size], dtype=np.float32)
        if (
            abs(float(window.sum())) <= 0.05
            and float(np.abs(window).sum()) <= 0.16
            and float(np.median(np.abs(window))) <= 0.03
            and (not len(support) or float(np.median(support)) >= 4.0)
        ):
            pause_edges[start : start + window_size] = True
    retained: list[int] = []
    start = 0
    while start < count:
        end = start
        while end < count - 1 and pause_edges[end]:
            end += 1
        retained.append(
            max(
                range(start, end + 1),
                key=lambda index: float(getattr(frames[index], "sharpness", 0.0)),
            )
        )
        start = end + 1
    return retained


def stabilize_motion_steps(motions: list[float]) -> list[float]:
    """Remove isolated angular outliers without flattening a slow orbit."""
    if len(motions) < 5:
        return [float(value) for value in motions]
    values = np.asarray(motions, dtype=np.float32)
    radius = 2
    local_median = np.asarray(
        [
            np.median(values[max(0, index - radius) : min(len(values), index + radius + 1)])
            for index in range(len(values))
        ],
        dtype=np.float32,
    )
    residual = np.abs(values - local_median)
    threshold = max(0.075, float(np.median(residual)) * 4.0)
    return np.where(residual > threshold, local_median, values).astype(float).tolist()


def smooth_cylindrical_steps(motions: Sequence[float]) -> list[float]:
    """Apply the short robust filter used for the final ordered orbit."""
    if len(motions) < 5:
        return [float(value) for value in motions]
    values = np.asarray(motions, dtype=np.float32)
    padded = np.pad(values, (2, 2), mode="edge")
    smoothed = np.median(
        np.stack([padded[offset : offset + len(values)] for offset in range(5)]),
        axis=0,
    )
    direction = float(np.sign(np.median(smoothed)))
    if direction < 0:
        smoothed = np.minimum(smoothed, 0.0)
    elif direction > 0:
        smoothed = np.maximum(smoothed, 0.0)
    return smoothed.astype(float).tolist()


def accumulate_vertical_offsets(vertical_deltas: list[float]) -> list[float]:
    """Accumulate and smooth small vertical camera shake between views."""
    if not vertical_deltas:
        return [0.0]
    increments = np.clip(np.asarray(vertical_deltas, dtype=np.float32), -12.0, 12.0)
    offsets = np.concatenate((np.zeros(1, dtype=np.float32), np.cumsum(increments)))
    if len(offsets) >= 7:
        padded = np.pad(offsets, (3, 3), mode="edge")
        offsets = np.median(np.stack([padded[index : index + len(offsets)] for index in range(7)]), axis=0)
        offsets -= offsets[0]
    return offsets.astype(float).tolist()


def solve_monotonic_trajectory(observations: list[tuple[float, float]]) -> CylinderTrajectory:
    """Build a robust temporal azimuth trajectory from adjacent observations.

    ``observations`` contains ``(delta_angle, confidence)`` in video order.
    It deliberately rejects a descriptor match that reverses the dominant motion
    instead of allowing it to reorder the reconstructed surface.
    """
    if not observations:
        return CylinderTrajectory([0.0], [], 0, float("inf"), 0.0, False, [], [])
    reliable = [(delta, confidence) for delta, confidence in observations if confidence >= 0.35 and abs(delta) > 1e-4]
    if not reliable:
        return CylinderTrajectory(
            [0.0] * (len(observations) + 1), [0.0] * len(observations), 0, float("inf"), 0.0, False,
            [False] * len(observations), ["low_texture"] * len(observations),
        )
    direction = 1.0 if float(np.median([delta for delta, _ in reliable])) >= 0 else -1.0
    forward = np.array([direction * delta for delta, _ in reliable if direction * delta > 0], dtype=float)
    nominal = float(np.median(forward)) if forward.size else 0.0
    if nominal <= 1e-4:
        return CylinderTrajectory(
            [0.0] * (len(observations) + 1), [0.0] * len(observations), 0, float("inf"), 0.0, False,
            [False] * len(observations), ["unstable_direction"] * len(observations),
        )
    steps: list[float] = []
    residuals: list[float] = []
    accepted_flags: list[bool] = []
    rejection_reasons: list[str] = []
    accepted = 0
    for delta, confidence in observations:
        candidate = direction * delta
        reason = ""
        if confidence < 0.35:
            reason = "low_texture"
        elif candidate <= 0:
            reason = "reversed_motion"
        if reason:
            # Do not invent a positive motion for an invalid observation.  It
            # remains visible in diagnostics and cannot create fake coverage.
            steps.append(0.0)
            accepted_flags.append(False)
            rejection_reasons.append(reason)
            continue
        accepted += 1
        residuals.append(candidate - nominal)
        steps.append(direction * candidate)
        accepted_flags.append(True)
        rejection_reasons.append("")
    angles = [0.0]
    for step in steps:
        angles.append(angles[-1] + step)
    residual = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else float("inf")
    sweep = abs(angles[-1] - angles[0])
    return CylinderTrajectory(
        angles, steps, accepted, residual, sweep, sweep > 2.0 * np.pi * 1.05, accepted_flags, rejection_reasons
    )

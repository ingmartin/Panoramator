from __future__ import annotations

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
    lower, upper = nominal * 0.35, nominal * 2.5
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
        elif candidate < lower or candidate > upper:
            reason = "scale_jump"
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

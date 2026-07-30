from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PoseEstimate:
    angles_degrees: list[float]
    confidence: float


def optimize_rotation_angles(increments_px: list[float], facade_width_px: float) -> PoseEstimate:
    """Integrate relative object rotations into deterministic global angles.

    This intentionally exposes a backend-neutral narrow interface: feature or
    optical-flow matchers can supply relative horizontal displacements without
    coupling cylinder reconstruction to ORB/SIFT.
    """
    if facade_width_px <= 0:
        return PoseEstimate([0.0], 0.0)
    angles = [0.0]
    for increment in increments_px:
        angles.append(angles[-1] + float(increment) / facade_width_px * 180.0)
    confidence = min(1.0, len(increments_px) / 8.0)
    return PoseEstimate(angles, confidence)

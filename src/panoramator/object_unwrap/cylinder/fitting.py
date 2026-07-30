from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analyzer import AnalyzedFrame
from ..models import SurfaceKind, SurfaceModel


@dataclass(slots=True)
class CylinderFit:
    model: SurfaceModel
    boxes: list[tuple[int, int, int, int]]


def fit_cylinder(frames: list[AnalyzedFrame]) -> CylinderFit:
    boxes = [item.bbox for item in frames]
    widths = np.array([box[2] for box in boxes], dtype=float)
    heights = np.array([box[3] for box in boxes], dtype=float)
    model = SurfaceModel(
        kind=SurfaceKind.CYLINDRICAL,
        radius_px=float(np.median(widths) / 2),
        top_y=float(np.median([box[1] for box in boxes])),
        bottom_y=float(np.median([box[1] + box[3] for box in boxes])),
        confidence=float(min(1.0, np.median(heights / np.maximum(widths, 1)) / 1.25)),
    )
    return CylinderFit(model, boxes)

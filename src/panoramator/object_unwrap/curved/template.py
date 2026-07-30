from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CurvedSurfaceTemplate:
    """Low-dimensional prior used only to decide mesh confidence."""

    width_to_height: float = 1.35
    dome_ratio: float = 0.62

    def compatibility(self, silhouettes: list[np.ndarray]) -> float:
        if not silhouettes:
            return 0.0
        ratios = []
        for silhouette in silhouettes:
            ys, xs = np.nonzero(silhouette)
            if xs.size:
                ratios.append((xs.max() - xs.min() + 1) / max(ys.max() - ys.min() + 1, 1))
        if not ratios:
            return 0.0
        deviation = abs(float(np.median(ratios)) - self.width_to_height) / self.width_to_height
        return float(max(0.0, 1.0 - deviation))

from __future__ import annotations

from ..analyzer import AnalyzedFrame
from ..cylinder.builder import CylinderUnwrapBuilder
from ..models import SurfaceKind, UnwrapConfig


class CurvedSurfaceFallbackBuilder:
    """Observed side-band fallback until a confidence-checked mesh is available."""
    def build(self, frames: list[AnalyzedFrame], config: UnwrapConfig):
        image, coverage, model, measurements = CylinderUnwrapBuilder().build(frames, config)
        model.kind = SurfaceKind.CURVED
        model.confidence *= 0.5
        measurements["fallback"] = "dominant_side_band"
        return image, coverage, model, measurements

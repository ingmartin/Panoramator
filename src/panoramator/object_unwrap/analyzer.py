from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panoramator.domain.models import Frame

from .models import SurfaceKind, UnwrapConfig, UnwrapStatus
from .segmentation import masked_sharpness, object_mask, stable_surface_bbox


@dataclass(slots=True)
class AnalyzedFrame:
    frame: Frame
    mask: np.ndarray
    sharpness: float
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class Analysis:
    frames: list[AnalyzedFrame]
    kind: SurfaceKind
    status: UnwrapStatus | None = None
    message: str = ""
    recommendation: str = ""


class VideoAnalyzer:
    def analyze(self, frames: list[Frame], config: UnwrapConfig) -> Analysis:
        candidates: list[AnalyzedFrame] = []
        for frame in frames:
            mask = object_mask(frame.image, config.min_object_area_ratio)
            if mask is None:
                continue
            bbox = stable_surface_bbox(mask)
            if bbox is None:
                continue
            x, y, width, height = bbox
            candidates.append(AnalyzedFrame(frame, mask, masked_sharpness(frame.image, mask), (x, y, width, height)))
        if len(candidates) < 2:
            return Analysis([], config.surface_kind, UnwrapStatus.OBJECT_NOT_DETECTED,
                            "The foreground surface cannot be separated reliably.",
                            "Record the surface larger in frame with stronger background contrast.")
        sharp = [item for item in candidates if item.sharpness >= config.blur_threshold]
        if len(sharp) < 2:
            return Analysis([], config.surface_kind, UnwrapStatus.EXCESSIVE_MOTION_BLUR,
                            "Too few sharp frames are available for a reliable texture.",
                            "Move more slowly and keep focus fixed.")
        kind = config.surface_kind
        if kind is SurfaceKind.AUTO:
            # A stable near-rectangular silhouette selects the developable model;
            # ambiguous footage uses the conservative curved-surface fallback.
            ratios = [item.bbox[2] / max(item.bbox[3], 1) for item in sharp]
            kind = SurfaceKind.CYLINDRICAL if 0.45 <= float(np.median(ratios)) <= 1.65 else SurfaceKind.CURVED
        return Analysis(sharp, kind)

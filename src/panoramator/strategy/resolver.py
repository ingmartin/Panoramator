from __future__ import annotations

from dataclasses import dataclass, field

from panoramator.config.models import PanoramaConfig
from panoramator.motion_analysis.analyzer import MotionAnalysis


@dataclass(frozen=True, slots=True)
class BuildDecision:
    requested_capture_mode: str
    requested_projection: str
    capture_mode: str
    projection: str
    confidence: float
    reason: str
    measurements: dict[str, float] = field(default_factory=dict)


def resolve_strategy(config: PanoramaConfig, analysis: MotionAnalysis | None = None) -> BuildDecision:
    analysis = analysis or MotionAnalysis.fallback()
    capture = config.capture_mode if config.capture_mode != "auto" else analysis.capture_mode
    confidence = 1.0 if config.capture_mode != "auto" else analysis.confidence
    reason = "manual_capture_mode" if config.capture_mode != "auto" else analysis.reason
    if config.projection != "auto":
        projection = config.projection
        reason = f"{reason}; manual_projection"
    elif capture == "rotation":
        projection = "cylindrical"
    else:
        # Auto mode keeps the compatible planar projection unless a clean
        # in-place rotation is detected. Object-orbit footage is handled later
        # as an unwrap-specific case, not by switching the scene-panorama
        # builder to another projection.
        projection = "planar"
        if capture == "orbit":
            reason = f"{reason}; orbital_capture_requires_unwrap"
    return BuildDecision(config.capture_mode, config.projection, capture, projection, confidence, reason, analysis.measurements)

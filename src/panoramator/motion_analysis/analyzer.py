from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class MotionAnalysis:
    capture_mode: str
    confidence: float
    reason: str
    measurements: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fallback(cls) -> MotionAnalysis:
        return cls("linear", 0.0, "insufficient_confidence_fallback_to_linear_planar")


class MotionAnalyzer:
    """Conservative chain-level classifier based on already estimated geometry."""

    def analyze(self, homographies: list[np.ndarray], pair_metrics: list[dict[str, object]]) -> MotionAnalysis:
        valid = [metric for metric in pair_metrics if bool(metric.get("valid"))]
        if not homographies or not valid:
            return MotionAnalysis.fallback()
        if any(matrix.shape != (3, 3) or not np.isfinite(matrix).all() for matrix in homographies):
            return MotionAnalysis.fallback()
        errors: list[float] = []
        for item in valid:
            error = item.get("reprojection_error", 0.0)
            if not isinstance(error, int | float) or not np.isfinite(error):
                return MotionAnalysis.fallback()
            errors.append(float(error))
        rotations = []
        scales = []
        translations = []
        for matrix in homographies:
            affine = matrix[:2, :2]
            rotations.append(float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0]))))
            scales.append(float(np.sqrt(abs(np.linalg.det(affine)))))
            translations.append(float(np.linalg.norm(matrix[:2, 2])))
        mean_error = sum(errors) / len(errors)
        rotation_mean = float(np.mean(np.abs(rotations)))
        scale_std = float(np.std(scales))
        translation_mean = float(np.mean(translations))
        measurements = {"mean_reprojection_error": mean_error, "mean_rotation_degrees": rotation_mean,
                        "scale_std": scale_std, "mean_translation_px": translation_mean, "pair_count": float(len(valid))}
        # Strong scale instability is a useful conservative parallax signal.
        if scale_std > 0.08 or mean_error > 4.0:
            return MotionAnalysis("orbit", min(0.95, 0.55 + scale_std + mean_error / 20.0), "spatially_inconsistent_geometry_or_scale_change", measurements)
        if len(valid) >= 2 and rotation_mean >= 1.0 and translation_mean > 1.0:
            return MotionAnalysis("rotation", min(0.9, 0.5 + rotation_mean / 20.0), "stable_rotation_indicated_by_chain_geometry", measurements)
        return MotionAnalysis("linear", 0.5, "low_parallax_without_confident_rotation", measurements)

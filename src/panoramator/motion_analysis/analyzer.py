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

    def analyze(
        self,
        homographies: list[np.ndarray],
        pair_metrics: list[dict[str, object]],
        cylindrical_preview: tuple[list[np.ndarray], list[dict[str, object]]] | None = None,
    ) -> MotionAnalysis:
        """Classify a whole chain, optionally comparing a cylindrical preview.

        The preview is deliberately a *gate*, not a reason to force a curved
        result: it must explain at least as many pairs with a lower residual.
        """
        summary = self._summarize(homographies, pair_metrics)
        if summary is None:
            return MotionAnalysis.fallback()
        measurements = dict(summary)
        if summary["scale_std"] > 0.08 or summary["mean_reprojection_error"] > 4.0:
            return MotionAnalysis(
                "orbit",
                min(0.95, 0.55 + summary["scale_std"] + summary["mean_reprojection_error"] / 20.0),
                "spatially_inconsistent_geometry_or_scale_change",
                measurements,
            )

        if cylindrical_preview is not None:
            preview = self._summarize(*cylindrical_preview)
            if preview is not None:
                measurements.update({f"cylindrical_{key}": value for key, value in preview.items()})
                residual_gain = 1.0 - preview["mean_reprojection_error"] / max(
                    summary["mean_reprojection_error"], 1e-6
                )
                measurements["cylindrical_residual_gain"] = residual_gain
                # A yaw rotation is largely a horizontal translation on a
                # cylinder, so do not require an affine roll in this branch.
                if (
                    preview["pair_count"] >= 2
                    and preview["mean_inlier_ratio"] >= summary["mean_inlier_ratio"] - 0.05
                    and residual_gain >= 0.10
                    and preview["mean_horizontal_translation_px"] > 1.0
                ):
                    return MotionAnalysis(
                        "rotation",
                        min(0.95, 0.60 + residual_gain + preview["mean_inlier_ratio"] / 4.0),
                        "cylindrical_preview_explains_rotation_better",
                        measurements,
                    )

        if cylindrical_preview is None and summary["pair_count"] >= 2 and summary["mean_rotation_degrees"] >= 1.0 and summary["mean_translation_px"] > 1.0:
            return MotionAnalysis(
                "rotation",
                min(0.9, 0.5 + summary["mean_rotation_degrees"] / 20.0),
                "stable_rotation_indicated_by_chain_geometry",
                measurements,
            )
        return MotionAnalysis("linear", 0.5, "low_parallax_without_confident_rotation", measurements)

    @staticmethod
    def _summarize(homographies: list[np.ndarray], pair_metrics: list[dict[str, object]]) -> dict[str, float] | None:
        valid = [metric for metric in pair_metrics if bool(metric.get("valid"))]
        if not homographies or not valid:
            return None
        if any(matrix.shape != (3, 3) or not np.isfinite(matrix).all() for matrix in homographies):
            return None
        errors: list[float] = []
        inlier_ratios: list[float] = []
        for item in valid:
            error = item.get("reprojection_error", 0.0)
            if not isinstance(error, int | float) or not np.isfinite(error):
                return None
            errors.append(float(error))
            inliers, good = item.get("inliers", 0), item.get("good_matches", 0)
            if isinstance(inliers, int | float) and isinstance(good, int | float) and good > 0:
                inlier_ratios.append(float(inliers) / float(good))
        rotations = []
        scales = []
        translations = []
        horizontal_translations = []
        for matrix in homographies:
            affine = matrix[:2, :2]
            rotations.append(float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0]))))
            scales.append(float(np.sqrt(abs(np.linalg.det(affine)))))
            translations.append(float(np.linalg.norm(matrix[:2, 2])))
            horizontal_translations.append(abs(float(matrix[0, 2])))
        mean_error = sum(errors) / len(errors)
        rotation_mean = float(np.mean(np.abs(rotations)))
        scale_std = float(np.std(scales))
        translation_mean = float(np.mean(translations))
        return {
            "mean_reprojection_error": mean_error,
            "reprojection_error_std": float(np.std(errors)),
            "mean_rotation_degrees": rotation_mean,
            "scale_std": scale_std,
            "mean_translation_px": translation_mean,
            "mean_horizontal_translation_px": float(np.mean(horizontal_translations)),
            "pair_count": float(len(valid)),
            "mean_inlier_ratio": float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
        }

from __future__ import annotations

import cv2
import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import FeatureSet, Frame, MatchSet, PairGeometry


class HomographyEstimator:
    def __init__(self, config: PanoramaConfig) -> None:
        self.config = config

    def estimate(
        self,
        left_frame: Frame,
        right_frame: Frame,
        left_features: FeatureSet,
        right_features: FeatureSet,
        matches: MatchSet,
    ) -> PairGeometry:
        if len(matches.good_matches) < self.config.min_match_count:
            return PairGeometry(None, 0, float("inf"), False, "not_enough_matches")

        src_points = np.asarray(
            [left_features.keypoints[m.queryIdx].pt for m in matches.good_matches], dtype=np.float32
        ).reshape(-1, 1, 2)
        dst_points = np.asarray(
            [right_features.keypoints[m.trainIdx].pt for m in matches.good_matches], dtype=np.float32
        ).reshape(-1, 1, 2)

        homography, mask, reason = self._estimate_transform(dst_points, src_points)
        if homography is None or mask is None:
            return PairGeometry(None, 0, float("inf"), False, reason)

        inliers = int(mask.ravel().sum())
        inlier_ratio = inliers / len(matches.good_matches)
        if inliers < self.config.min_inlier_count:
            return PairGeometry(None, inliers, float("inf"), False, "not_enough_inliers")
        if inlier_ratio < self.config.min_inlier_ratio:
            return PairGeometry(None, inliers, float("inf"), False, "inlier_ratio")
        reprojection_error = self._compute_reprojection_error(dst_points, src_points, homography, mask)
        if reprojection_error > self.config.max_reprojection_error:
            return PairGeometry(None, inliers, reprojection_error, False, "reprojection_error")

        valid, reason = self._validate_transform(homography, right_frame.image.shape[:2])
        return PairGeometry(homography, inliers, reprojection_error, valid, reason)

    def _estimate_transform(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        model = self.config.motion_model
        if model == "translation":
            deltas = (dst_points.reshape(-1, 2) - src_points.reshape(-1, 2)).astype(np.float64)
            dx, dy = np.median(deltas, axis=0)
            transform = np.array(
                [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            residuals = np.linalg.norm(deltas - np.array([dx, dy]), axis=1)
            mask = (residuals <= self.config.ransac_threshold).astype(np.uint8).reshape(-1, 1)
            return transform, mask, "ok"

        if model == "partial_affine":
            affine, mask = cv2.estimateAffinePartial2D(
                src_points,
                dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold,
            )
            if affine is None or mask is None:
                return None, None, "partial_affine_failed"
            return _affine_to_homography(affine), mask, "ok"

        if model == "affine":
            affine, mask = cv2.estimateAffine2D(
                src_points,
                dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold,
            )
            if affine is None or mask is None:
                return None, None, "affine_failed"
            return _affine_to_homography(affine), mask, "ok"

        homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, self.config.ransac_threshold)
        if homography is None or mask is None:
            return None, None, "homography_failed"
        return homography, mask, "ok"

    def _validate_transform(
        self, homography: np.ndarray, source_shape: tuple[int, int] | None = None
    ) -> tuple[bool, str]:
        linear = homography[:2, :2]
        _, singular_values, _ = np.linalg.svd(linear)
        scale_x = float(singular_values[0])
        scale_y = float(singular_values[1])
        scale_deviation = max(abs(scale_x - 1.0), abs(scale_y - 1.0))
        if scale_deviation > self.config.max_scale_deviation:
            return False, "scale_deviation"

        rotation = np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))
        if abs(float(rotation)) > self.config.max_rotation_degrees:
            return False, "rotation_deviation"

        if self.config.motion_model == "homography":
            perspective = max(abs(float(homography[2, 0])), abs(float(homography[2, 1])))
            if perspective > 1e-3:
                return False, "perspective_deviation"
            if source_shape is not None and not self._has_reasonable_projected_size(homography, source_shape):
                return False, "projected_frame_scale"

        return True, "ok"

    def _has_reasonable_projected_size(self, homography: np.ndarray, source_shape: tuple[int, int]) -> bool:
        height, width = source_shape
        corners = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        if not np.isfinite(projected).all():
            return False
        projected_width, projected_height = projected.max(axis=0) - projected.min(axis=0)
        maximum_scale = self.config.max_homography_corner_scale
        return projected_width <= width * maximum_scale and projected_height <= height * maximum_scale

    @staticmethod
    def _compute_reprojection_error(
        src_points: np.ndarray,
        dst_points: np.ndarray,
        homography: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        projected = cv2.perspectiveTransform(src_points, homography)
        inlier_mask = mask.ravel().astype(bool)
        if not np.any(inlier_mask):
            return float("inf")
        deltas = projected[inlier_mask] - dst_points[inlier_mask]
        return float(np.mean(np.linalg.norm(deltas, axis=2)))


def accumulate_global_homographies(pairwise_homographies: list[np.ndarray]) -> list[np.ndarray]:
    global_homographies = [np.eye(3, dtype=np.float64)]
    current = np.eye(3, dtype=np.float64)
    for homography in pairwise_homographies:
        current = current @ homography
        current = current / current[2, 2]
        global_homographies.append(current.copy())
    return global_homographies


def _affine_to_homography(affine: np.ndarray) -> np.ndarray:
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = affine
    return homography

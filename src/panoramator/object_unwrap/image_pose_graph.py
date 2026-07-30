from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .analyzer import AnalyzedFrame


@dataclass(slots=True)
class ImagePoseGraph:
    edges: list[dict[str, float | int | str]]

    @property
    def valid_edges(self) -> int:
        return sum(edge["reason"] == "ok" for edge in self.edges)


def build_image_pose_graph(frames: list[AnalyzedFrame], max_hop: int = 3) -> ImagePoseGraph:
    """Measure image-space constraints without claiming that they are UV pose.

    Features are detected in a mildly dilated object mask.  Each edge records
    how many RANSAC inliers remain in the strict surface mask, preventing a
    background-only registration from being mistaken for surface evidence.
    """
    detector = cv2.ORB_create(nfeatures=1800)  # type: ignore[attr-defined]
    features = []
    for item in frames:
        geometry_mask = cv2.dilate(item.mask, np.ones((15, 15), np.uint8))
        keypoints, descriptors = detector.detectAndCompute(cv2.cvtColor(item.frame.image, cv2.COLOR_BGR2GRAY), geometry_mask)
        features.append((keypoints, descriptors))
    edges: list[dict[str, float | int | str]] = []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    for left_index in range(len(frames) - 1):
        for right_index in range(left_index + 1, min(len(frames), left_index + max_hop + 1)):
            left_points, left_descriptors = features[left_index]
            right_points, right_descriptors = features[right_index]
            edge: dict[str, float | int | str] = {
                "left_frame": frames[left_index].frame.index,
                "right_frame": frames[right_index].frame.index,
                "hop": right_index - left_index,
                "good_matches": 0,
                "inliers": 0,
                "surface_inliers": 0,
                "reprojection_error": float("inf"),
                "reason": "low_texture",
            }
            if left_descriptors is None or right_descriptors is None:
                edges.append(edge)
                continue
            pairs = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
            good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
            edge["good_matches"] = len(good)
            if len(good) < 12:
                edges.append(edge)
                continue
            source = np.asarray([right_points[match.trainIdx].pt for match in good], np.float32)
            target = np.asarray([left_points[match.queryIdx].pt for match in good], np.float32)
            affine, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if affine is None or inliers is None:
                edge["reason"] = "ransac_failed"
                edges.append(edge)
                continue
            accepted = inliers.ravel().astype(bool)
            edge["inliers"] = int(accepted.sum())
            projected = cv2.transform(source[accepted, None, :], affine).reshape(-1, 2)
            edge["reprojection_error"] = float(np.mean(np.linalg.norm(projected - target[accepted], axis=1)))
            strict_left = frames[left_index].mask
            strict_right = frames[right_index].mask
            surface = []
            for match, keep in zip(good, accepted, strict=False):
                if not keep:
                    continue
                x1, y1 = np.round(left_points[match.queryIdx].pt).astype(int)
                x2, y2 = np.round(right_points[match.trainIdx].pt).astype(int)
                if 0 <= y1 < strict_left.shape[0] and 0 <= x1 < strict_left.shape[1] and 0 <= y2 < strict_right.shape[0] and 0 <= x2 < strict_right.shape[1]:
                    surface.append(strict_left[y1, x1] > 0 and strict_right[y2, x2] > 0)
            surface_inliers = int(sum(surface))
            edge["surface_inliers"] = surface_inliers
            edge["reason"] = "ok" if surface_inliers >= 8 else "background_only"
            if edge["reason"] == "ok":
                edge.update({f"a{row}{column}": float(affine[row, column]) for row in range(2) for column in range(3)})
            edges.append(edge)
    return ImagePoseGraph(edges)

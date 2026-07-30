from __future__ import annotations

import cv2
import numpy as np


def normalized_wall(image: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int], height: int) -> tuple[np.ndarray, np.ndarray]:
    x, y, width, source_height = bbox
    wall = image[y : y + source_height, x : x + width]
    wall_mask = mask[y : y + source_height, x : x + width]
    # Exclude a thin silhouette boundary, where the background can leak in.
    wall_mask = cv2.erode(wall_mask, np.ones((3, 3), np.uint8))
    target_width = max(16, int(round(width * height / max(source_height, 1))))
    return (
        cv2.resize(wall, (target_width, height), interpolation=cv2.INTER_AREA),
        cv2.resize(wall_mask, (target_width, height), interpolation=cv2.INTER_NEAREST),
    )


def central_band(image: np.ndarray, mask: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep the low-perspective central part of an observed surface patch."""
    width = image.shape[1]
    band_width = max(8, int(round(width * ratio)))
    start = max(0, (width - band_width) // 2)
    end = min(width, start + band_width)
    return image[:, start:end], mask[:, start:end]


def horizontal_shift(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Estimate texture displacement; response makes bad matches harmless."""
    left_source = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right_source = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float32)
    left_gray = cv2.resize(left_source, (256, 128), interpolation=cv2.INTER_AREA)
    right_gray = cv2.resize(right_source, (256, 128), interpolation=cv2.INTER_AREA)
    shift, response = cv2.phaseCorrelate(left_gray, right_gray)
    return float(shift[0]), float(response)


def feature_shift(left: np.ndarray, left_mask: np.ndarray, right: np.ndarray, right_mask: np.ndarray) -> tuple[float, float]:
    """Return image-space horizontal displacement from masked ORB matches."""
    detector = cv2.ORB_create(nfeatures=1200)  # type: ignore[attr-defined]
    left_keypoints, left_descriptors = detector.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), left_mask)
    right_keypoints, right_descriptors = detector.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), right_mask)
    if left_descriptors is None or right_descriptors is None:
        return 0.0, 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(left_descriptors, right_descriptors, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
    if len(good) < 8:
        return 0.0, 0.0
    source = np.asarray([left_keypoints[match.queryIdx].pt for match in good], dtype=np.float32)
    target = np.asarray([right_keypoints[match.trainIdx].pt for match in good], dtype=np.float32)
    affine, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if affine is None or inliers is None:
        return 0.0, 0.0
    inlier_ratio = float(np.asarray(inliers).mean())
    return float(affine[0, 2]), inlier_ratio


def angular_increment(
    left: np.ndarray,
    left_mask: np.ndarray,
    right: np.ndarray,
    right_mask: np.ndarray,
    central_band_ratio: float,
) -> tuple[float, float]:
    """Estimate relative viewing azimuth from surface-local feature matches."""
    detector = cv2.ORB_create(nfeatures=1600)  # type: ignore[attr-defined]
    left_keypoints, left_descriptors = detector.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), left_mask)
    right_keypoints, right_descriptors = detector.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), right_mask)
    if left_descriptors is None or right_descriptors is None:
        return 0.0, 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(left_descriptors, right_descriptors, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.70 * pair[1].distance]
    if len(good) < 10:
        return 0.0, 0.0
    left_width = max(left.shape[1] - 1, 1)
    right_width = max(right.shape[1] - 1, 1)
    deltas = []
    for match in good:
        left_coordinate = (left_keypoints[match.queryIdx].pt[0] / left_width * 2.0 - 1.0) * central_band_ratio
        right_coordinate = (right_keypoints[match.trainIdx].pt[0] / right_width * 2.0 - 1.0) * central_band_ratio
        deltas.append(float(np.arcsin(np.clip(left_coordinate, -0.98, 0.98)) - np.arcsin(np.clip(right_coordinate, -0.98, 0.98))))
    median = float(np.median(deltas))
    inliers = np.abs(np.asarray(deltas) - median) < 0.075
    if int(inliers.sum()) < 8:
        return 0.0, 0.0
    return float(np.median(np.asarray(deltas)[inliers])), float(inliers.mean())


def flow_angular_increment(
    left: np.ndarray,
    left_mask: np.ndarray,
    right: np.ndarray,
    central_band_ratio: float,
) -> tuple[float, float]:
    """Track local texture between adjacent views before descriptor matching."""
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    if right_gray.shape != left_gray.shape:
        right_gray = cv2.resize(right_gray, (left_gray.shape[1], left_gray.shape[0]), interpolation=cv2.INTER_AREA)
    points = cv2.goodFeaturesToTrack(left_gray, maxCorners=500, qualityLevel=0.008, minDistance=5, mask=left_mask)
    if points is None or len(points) < 12:
        return 0.0, 0.0
    tracked, status, errors = cv2.calcOpticalFlowPyrLK(
        left_gray, right_gray, points, None, None, None, (21, 21), 3
    )  # type: ignore[call-overload]
    if tracked is None or status is None or errors is None:
        return 0.0, 0.0
    if tracked is None:
        return 0.0, 0.0
    backward, backward_status, backward_errors = cv2.calcOpticalFlowPyrLK(
        right_gray, left_gray, tracked, None, None, None, (21, 21), 3
    )  # type: ignore[call-overload]
    if backward is None or backward_status is None or backward_errors is None:
        return 0.0, 0.0
    roundtrip = np.linalg.norm(backward.reshape(-1, 2) - points.reshape(-1, 2), axis=1)
    valid = (status.ravel() == 1) & (errors.ravel() < 18.0) & (backward_status.ravel() == 1) & (roundtrip < 1.25)
    origin = points.reshape(-1, 2)[valid]
    destination = tracked.reshape(-1, 2)[valid]
    if len(origin) < 12:
        return 0.0, 0.0
    width = max(left.shape[1] - 1, 1)
    left_coordinate = (origin[:, 0] / width * 2.0 - 1.0) * central_band_ratio
    right_coordinate = (destination[:, 0] / width * 2.0 - 1.0) * central_band_ratio
    deltas = np.arcsin(np.clip(left_coordinate, -0.98, 0.98)) - np.arcsin(np.clip(right_coordinate, -0.98, 0.98))
    median = float(np.median(deltas))
    inliers = np.abs(deltas - median) < 0.05
    if int(inliers.sum()) < 10:
        return 0.0, 0.0
    return float(np.median(deltas[inliers])), float(inliers.mean())

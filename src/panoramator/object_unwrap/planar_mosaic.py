from __future__ import annotations

import itertools

import cv2
import numpy as np

from .analyzer import AnalyzedFrame
from .models import PublishProfile


def build_planar_mosaic(
    frames: list[AnalyzedFrame],
    edges: list[dict[str, float | int | str]],
    output_height: int,
    publish_profile: PublishProfile = PublishProfile.BALANCED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Warp observed masks with graph transforms into one image-space mosaic."""
    by_pair = {(int(edge["left_frame"]), int(edge["right_frame"])): edge for edge in edges if edge["reason"] == "ok"}
    transforms = [np.eye(3, dtype=np.float64)]
    for left, right in itertools.pairwise(frames):
        edge = by_pair.get((left.frame.index, right.frame.index))
        if edge is None:
            return None
        affine = np.array([[edge[f"a{row}{column}"] for column in range(3)] for row in range(2)], dtype=np.float64)
        local = np.eye(3, dtype=np.float64)
        local[:2] = affine
        transforms.append(transforms[-1] @ local)
    corners = []
    for item, transform in zip(frames, transforms, strict=True):
        height, width = item.frame.image.shape[:2]
        points = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(points, transform).reshape(-1, 2))
    all_corners = np.concatenate(corners)
    minimum, maximum = np.floor(all_corners.min(axis=0)), np.ceil(all_corners.max(axis=0))
    width, height = (maximum - minimum).astype(int)
    if width < 2 or height < 2 or width > 12_000 or height > 12_000 or width * height > 8_000_000:
        return None
    offset = np.eye(3, dtype=np.float64)
    offset[:2, 2] = -minimum
    canvas: np.ndarray = np.zeros((height, width, 3), np.float32)
    owner: np.ndarray = np.zeros((height, width), np.uint16)
    strength: np.ndarray = np.zeros((height, width), np.float32)
    owner_detail: np.ndarray = np.zeros((height, width), np.float32)
    publish_weight: np.ndarray = np.zeros((height, width), np.float32)
    error: np.ndarray = np.zeros((height, width), np.float32)
    for index, (item, transform) in enumerate(zip(frames, transforms, strict=True), start=1):
        warped = cv2.warpPerspective(item.frame.image, offset @ transform, (width, height))
        mask = cv2.warpPerspective(item.publish_mask, offset @ transform, (width, height), flags=cv2.INTER_NEAREST)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        candidate_detail = cv2.warpPerspective(_detail_map(item.frame.image, item.publish_mask), offset @ transform, (width, height))
        candidate_score = _owner_score(distance, candidate_detail, item.sharpness, mask > 0)
        overlap = (mask > 0) & (strength > 0)
        if np.any(overlap):
            error[overlap] = np.maximum(error[overlap], np.mean(np.abs(canvas[overlap] - warped[overlap].astype(np.float32)), axis=1))
        seam_stability = _seam_stability_map(owner, strength > 0)
        soft_publish = _soft_publish_mask(
            owner_detail,
            candidate_detail,
            error,
            overlap,
            seam_stability,
            publish_profile,
        )
        replace = (mask > 0) & (
            candidate_score > _required_owner_score(strength, owner_detail, candidate_detail, error, overlap, seam_stability)
        )
        if np.any(soft_publish):
            _apply_soft_publish(canvas, publish_weight, warped, strength, candidate_score, soft_publish)
        hard_replace = replace & ~soft_publish
        if np.any(hard_replace):
            canvas[hard_replace] = warped[hard_replace].astype(np.float32)
            publish_weight[hard_replace] = np.maximum(candidate_score[hard_replace], 1e-3)
        hard_new = (mask > 0) & (strength == 0)
        if np.any(hard_new):
            canvas[hard_new] = warped[hard_new].astype(np.float32)
            publish_weight[hard_new] = np.maximum(candidate_score[hard_new], 1e-3)
        owner[replace] = index
        strength[replace] = candidate_score[replace]
        owner_detail[replace] = candidate_detail[replace]
    coverage: np.ndarray = np.where(strength > 0, 255, 0).astype(np.uint8)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    if output_height != height:
        scale = output_height / height
        target_width = max(1, round(width * scale))
        canvas = cv2.resize(canvas, (target_width, output_height), interpolation=cv2.INTER_AREA)
        coverage = cv2.resize(coverage, (target_width, output_height), interpolation=cv2.INTER_NEAREST)
        owner = cv2.resize(owner, (target_width, output_height), interpolation=cv2.INTER_NEAREST)
        error = cv2.resize(error, (target_width, output_height), interpolation=cv2.INTER_AREA)
    return canvas, coverage, owner, np.clip(error, 0, 255).astype(np.uint8)


def _detail_map(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    occupied = mask > 0
    detail = np.zeros_like(gradient, dtype=np.float32)
    occupied_gradient = gradient[occupied]
    if occupied_gradient.size:
        scale = max(float(np.percentile(occupied_gradient, 95)), 1.0)
        detail[occupied] = np.clip(occupied_gradient / scale, 0.0, 1.0)
    return detail


def _owner_score(distance: np.ndarray, detail: np.ndarray, sharpness: float, occupied: np.ndarray) -> np.ndarray:
    normalized_distance = np.zeros_like(distance, dtype=np.float32)
    occupied_distance = distance[occupied]
    if occupied_distance.size:
        scale = max(float(np.percentile(occupied_distance, 95)), 1.0)
        normalized_distance[occupied] = np.clip(occupied_distance / scale, 0.0, 1.0)
    centrality_bonus = 0.7 + 0.3 * normalized_distance
    detail_bonus = 0.8 + 0.2 * detail.astype(np.float32)
    return distance.astype(np.float32) * max(float(sharpness), 1.0) * centrality_bonus * detail_bonus


def _required_owner_score(
    current_score: np.ndarray,
    current_detail: np.ndarray,
    candidate_detail: np.ndarray,
    error: np.ndarray,
    overlap: np.ndarray,
    seam_stability: np.ndarray,
) -> np.ndarray:
    required = current_score.copy()
    if not np.any(overlap):
        return required
    conflict_error = np.clip(error.astype(np.float32) / 48.0, 0.0, 1.5)
    detail_conflict = np.maximum(current_detail, candidate_detail).astype(np.float32)
    protected_region = np.clip(seam_stability.astype(np.float32), 0.0, 1.0) * detail_conflict
    margin = 1.0 + detail_conflict * conflict_error * 0.35 + protected_region * (0.12 + 0.18 * conflict_error)
    required[overlap] *= margin[overlap]
    return required


def _seam_stability_map(owner: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    if not np.any(occupied):
        return np.zeros_like(owner, dtype=np.float32)
    boundary = np.zeros_like(owner, dtype=np.uint8)
    boundary[:, 1:] |= (owner[:, 1:] != owner[:, :-1]) & occupied[:, 1:] & occupied[:, :-1]
    boundary[:, :-1] |= (owner[:, 1:] != owner[:, :-1]) & occupied[:, 1:] & occupied[:, :-1]
    boundary[1:, :] |= (owner[1:, :] != owner[:-1, :]) & occupied[1:, :] & occupied[:-1, :]
    boundary[:-1, :] |= (owner[1:, :] != owner[:-1, :]) & occupied[1:, :] & occupied[:-1, :]
    interior = (occupied & (boundary == 0)).astype(np.uint8)
    stability = cv2.distanceTransform(interior, cv2.DIST_L2, 3).astype(np.float32)
    occupied_values = stability[occupied]
    if occupied_values.size:
        scale = max(float(np.percentile(occupied_values, 90)), 1.0)
        stability[occupied] = np.clip(occupied_values / scale, 0.0, 1.0)
    return stability


def _soft_publish_mask(
    current_detail: np.ndarray,
    candidate_detail: np.ndarray,
    error: np.ndarray,
    overlap: np.ndarray,
    seam_stability: np.ndarray,
    publish_profile: PublishProfile,
) -> np.ndarray:
    if not np.any(overlap):
        return np.zeros_like(overlap, dtype=bool)
    settings = _publish_profile_settings(publish_profile)
    detail_conflict = np.maximum(current_detail, candidate_detail).astype(np.float32)
    conflict_error = np.clip(error.astype(np.float32) / 48.0, 0.0, 1.5)
    protected_anchor = detail_conflict * np.clip(seam_stability.astype(np.float32), 0.0, 1.0)
    very_smooth = detail_conflict <= settings["soft_detail_floor"]
    smooth_with_consistent_photometry = (detail_conflict <= settings["soft_detail_ceiling"]) & (
        conflict_error <= settings["soft_error_ceiling"]
    )
    return overlap & (protected_anchor <= settings["soft_anchor_protection"]) & (
        very_smooth | smooth_with_consistent_photometry
    )


def _apply_soft_publish(
    canvas: np.ndarray,
    publish_weight: np.ndarray,
    warped: np.ndarray,
    current_score: np.ndarray,
    candidate_score: np.ndarray,
    soft_publish: np.ndarray,
) -> None:
    current_weight = np.maximum(np.sqrt(np.maximum(current_score[soft_publish], 0.0)), 1e-3)
    candidate_weight = np.maximum(np.sqrt(np.maximum(candidate_score[soft_publish], 0.0)), 1e-3)
    total_weight = current_weight + candidate_weight
    canvas[soft_publish] = (
        canvas[soft_publish] * current_weight[:, None] + warped[soft_publish].astype(np.float32) * candidate_weight[:, None]
    ) / total_weight[:, None]
    publish_weight[soft_publish] = np.maximum(publish_weight[soft_publish] + candidate_weight, total_weight)


def _publish_profile_settings(publish_profile: PublishProfile) -> dict[str, float]:
    if publish_profile is PublishProfile.CONSERVATIVE:
        return {
            "soft_detail_floor": 0.08,
            "soft_detail_ceiling": 0.16,
            "soft_error_ceiling": 0.75,
            "soft_anchor_protection": 0.05,
        }
    if publish_profile is PublishProfile.COVERAGE_FIRST:
        return {
            "soft_detail_floor": 0.16,
            "soft_detail_ceiling": 0.30,
            "soft_error_ceiling": 1.25,
            "soft_anchor_protection": 0.12,
        }
    return {
        "soft_detail_floor": 0.12,
        "soft_detail_ceiling": 0.22,
        "soft_error_ceiling": 0.9,
        "soft_anchor_protection": 0.08,
    }

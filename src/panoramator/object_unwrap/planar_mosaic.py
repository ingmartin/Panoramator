from __future__ import annotations

import cv2
import numpy as np

from .analyzer import AnalyzedFrame


def build_planar_mosaic(
    frames: list[AnalyzedFrame], edges: list[dict[str, float | int | str]], output_height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Warp observed masks with graph transforms into one image-space mosaic."""
    by_pair = {(int(edge["left_frame"]), int(edge["right_frame"])): edge for edge in edges if edge["reason"] == "ok"}
    transforms = [np.eye(3, dtype=np.float64)]
    for left, right in zip(frames, frames[1:], strict=False):
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
    canvas: np.ndarray = np.zeros((height, width, 3), np.uint8)
    owner: np.ndarray = np.zeros((height, width), np.uint16)
    strength: np.ndarray = np.zeros((height, width), np.float32)
    error: np.ndarray = np.zeros((height, width), np.float32)
    for index, (item, transform) in enumerate(zip(frames, transforms, strict=True), start=1):
        warped = cv2.warpPerspective(item.frame.image, offset @ transform, (width, height))
        x, y, box_width, box_height = item.bbox
        surface_mask = np.zeros_like(item.mask)
        surface_mask[y : y + box_height, x : x + box_width] = item.mask[y : y + box_height, x : x + box_width]
        mask = cv2.warpPerspective(surface_mask, offset @ transform, (width, height), flags=cv2.INTER_NEAREST)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3) * max(item.sharpness, 1.0)
        overlap = (mask > 0) & (strength > 0)
        if np.any(overlap):
            error[overlap] = np.maximum(error[overlap], np.mean(np.abs(canvas[overlap].astype(np.float32) - warped[overlap]), axis=1))
        replace = (mask > 0) & (distance > strength)
        canvas[replace] = warped[replace]
        owner[replace] = index
        strength[replace] = distance[replace]
    coverage: np.ndarray = np.where(strength > 0, 255, 0).astype(np.uint8)
    if output_height != height:
        scale = output_height / height
        target_width = max(1, int(round(width * scale)))
        canvas = cv2.resize(canvas, (target_width, output_height), interpolation=cv2.INTER_AREA)
        coverage = cv2.resize(coverage, (target_width, output_height), interpolation=cv2.INTER_NEAREST)
        owner = cv2.resize(owner, (target_width, output_height), interpolation=cv2.INTER_NEAREST)
        error = cv2.resize(error, (target_width, output_height), interpolation=cv2.INTER_AREA)
    return canvas, coverage, owner, np.clip(error, 0, 255).astype(np.uint8)

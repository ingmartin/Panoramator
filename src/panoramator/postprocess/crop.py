from __future__ import annotations

import cv2
import numpy as np


def crop_black_borders(image: np.ndarray, visible_mask: np.ndarray | None = None) -> np.ndarray:
    mask = _resolve_visible_mask(image, visible_mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    return image[y : y + height, x : x + width]


def crop_to_visible_area(image: np.ndarray, visible_mask: np.ndarray | None = None) -> np.ndarray:
    mask = _resolve_visible_mask(image, visible_mask)
    visible = mask > 0
    top, left, height, width = _largest_visible_rectangle(visible)
    if height == 0 or width == 0:
        return image
    return image[top : top + height, left : left + width]


def _resolve_visible_mask(image: np.ndarray, visible_mask: np.ndarray | None) -> np.ndarray:
    if visible_mask is not None:
        return (visible_mask > 0).astype(np.uint8) * 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return threshold


def _largest_visible_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    if mask.size == 0:
        return 0, 0, 0, 0

    heights = np.zeros(mask.shape[1], dtype=np.int32)
    best_area = 0
    best_rect = (0, 0, 0, 0)

    for row in range(mask.shape[0]):
        heights = np.where(mask[row], heights + 1, 0)
        stack: list[int] = []

        for col in range(mask.shape[1] + 1):
            current_height = int(heights[col]) if col < mask.shape[1] else 0
            while stack and current_height < int(heights[stack[-1]]):
                height = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                width = col - left
                area = height * width
                if area > best_area:
                    top = row - height + 1
                    best_area = area
                    best_rect = (top, left, height, width)
            stack.append(col)

    return best_rect

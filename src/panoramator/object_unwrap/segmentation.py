from __future__ import annotations

import cv2
import numpy as np


def object_mask(image: np.ndarray, min_area_ratio: float = 0.025) -> np.ndarray | None:
    """Return the largest plausible foreground component without an ML model.

    GrabCut is deliberately seeded with an inset rectangle: it is deterministic
    and works for the expected single-object videos while never inventing pixels
    outside the observed mask.
    """
    height, width = image.shape[:2]
    if height < 8 or width < 8:
        return None
    labels = np.zeros((height, width), np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    inset_x, inset_y = max(1, width // 12), max(1, height // 12)
    try:
        cv2.grabCut(image, labels, (inset_x, inset_y, width - 2 * inset_x, height - 2 * inset_y), background, foreground, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    mask: np.ndarray = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if stats[best, cv2.CC_STAT_AREA] < height * width * min_area_ratio:
        return None
    result = np.where(components == best, 255, 0).astype(np.uint8)
    # Components touching almost every border are normally the background.
    x, y, w, h, _ = stats[best]
    if x == 0 and y == 0 and x + w == width and y + h == height:
        return None
    return result


def masked_sharpness(image: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = cv2.Laplacian(gray, cv2.CV_64F)[mask > 0]
    return float(values.var()) if values.size else 0.0


def stable_surface_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the widest stable horizontal band and discard transient attachments."""
    rows = np.count_nonzero(mask, axis=1)
    if not rows.size or rows.max() == 0:
        return None
    valid = rows >= rows.max() * 0.72
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, present in enumerate(valid):
        if present and start is None:
            start = index
        elif not present and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(valid)))
    if not runs:
        return None
    top, bottom = max(runs, key=lambda run: run[1] - run[0])
    if bottom - top < max(12, mask.shape[0] // 8):
        return None
    columns = np.count_nonzero(mask[top:bottom], axis=0)
    x_positions = np.flatnonzero(columns >= max(1, (bottom - top) * 0.5))
    if not x_positions.size:
        return None
    return int(x_positions[0]), int(top), int(x_positions[-1] - x_positions[0] + 1), int(bottom - top)

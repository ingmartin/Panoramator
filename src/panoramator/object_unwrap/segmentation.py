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


def publish_surface_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Keep only the main observed side band inside the stable bbox.

    The matching geometry may legitimately include extra support above the
    object band, but publication must stay conservative: for each column we
    keep only the vertical foreground run intersecting the stable band centre.
    This removes fingers or other occluders sticking up above the rim while
    preserving the contiguous visible surface below.
    """
    x, y, width, height = bbox
    result = np.zeros_like(mask)
    axis_row = y + max(height // 2, 0)
    tops = np.full(width, np.nan, dtype=np.float32)
    bottoms = np.full(width, np.nan, dtype=np.float32)
    for column in range(x, x + width):
        ys = np.flatnonzero(mask[:, column] > 0)
        if ys.size == 0:
            continue
        splits = np.flatnonzero(np.diff(ys) > 1) + 1
        runs = np.split(ys, splits)
        chosen = None
        for run in runs:
            if run[0] <= axis_row <= run[-1]:
                chosen = run
                break
        if chosen is None:
            chosen = min(runs, key=lambda run: min(abs(int(run[0]) - axis_row), abs(int(run[-1]) - axis_row)))
        local = column - x
        tops[local] = float(chosen[0])
        bottoms[local] = float(chosen[-1])
    valid = np.isfinite(tops) & np.isfinite(bottoms)
    if not np.any(valid):
        return result
    indices = np.arange(width, dtype=np.float32)
    tops[~valid] = np.interp(indices[~valid], indices[valid], tops[valid])
    bottoms[~valid] = np.interp(indices[~valid], indices[valid], bottoms[valid])
    band_heights = bottoms - tops + 1.0
    median_height = float(np.median(band_heights[valid]))
    kernel = max(9, (width // 12) | 1)
    smooth_top = cv2.GaussianBlur(tops[None, :], (kernel, 1), 0).reshape(-1)
    smooth_bottom = cv2.GaussianBlur(bottoms[None, :], (kernel, 1), 0).reshape(-1)
    margin = max(2, height // 30)
    spike_tolerance = max(2.0, median_height * 0.12)
    min_support_height = max(6.0, median_height * 0.35)
    for column in range(x, x + width):
        local = column - x
        trimmed_top = max(tops[local], smooth_top[local] - spike_tolerance)
        trimmed_bottom = min(bottoms[local], smooth_bottom[local] + spike_tolerance)
        top = int(round(max(y, trimmed_top - margin)))
        bottom = int(round(min(y + height - 1, trimmed_bottom + margin)))
        if bottom < top:
            continue
        present = mask[top : bottom + 1, column] > 0
        if int(np.count_nonzero(present)) < min_support_height:
            continue
        result[top : bottom + 1, column][present] = 255
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return result

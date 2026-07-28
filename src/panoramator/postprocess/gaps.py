from __future__ import annotations

import numpy as np


def fill_narrow_mask_gaps(
    image: np.ndarray, visible_mask: np.ndarray, max_width: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Interpolate only narrow horizontal mask gaps enclosed by valid pixels.

    This repairs thin vertical black strips introduced by rasterisation gaps.  It
    deliberately does not touch image borders or wider missing regions, where
    inventing image content would be misleading.
    """
    result = image.copy()
    mask = (visible_mask > 0).astype(np.uint8) * 255
    filled_runs = 0
    filled_pixels = 0
    for row in range(mask.shape[0]):
        valid = mask[row] > 0
        column = 0
        while column < len(valid):
            if valid[column]:
                column += 1
                continue
            start = column
            while column < len(valid) and not valid[column]:
                column += 1
            end = column
            width = end - start
            if start == 0 or end == len(valid) or width > max_width:
                continue
            left = result[row, start - 1].astype(np.float32)
            right = result[row, end].astype(np.float32)
            for offset in range(width):
                alpha = (offset + 1) / (width + 1)
                result[row, start + offset] = np.clip(left * (1.0 - alpha) + right * alpha, 0, 255)
            mask[row, start:end] = 255
            filled_runs += 1
            filled_pixels += width
    return result, mask, {"filled_runs": float(filled_runs), "filled_pixels": float(filled_pixels)}

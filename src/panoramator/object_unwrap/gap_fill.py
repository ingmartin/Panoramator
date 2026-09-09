from __future__ import annotations

import itertools

import numpy as np


def interpolate_surface_gaps(
    image: np.ndarray,
    coverage: np.ndarray,
    source_map: np.ndarray,
    max_gap: int,
) -> int:
    """Fill bounded holes between observed surface pixels.

    The operation is deliberately limited to gaps bracketed by real samples
    in the same row or column.  It never fills an outer margin or an area with
    no observations on one side, so it cannot turn missing geometry into a
    claimed surface.
    """
    if max_gap < 1:
        raise ValueError("max_gap must be >= 1")
    filled = 0
    for row in range(image.shape[0]):
        columns = np.flatnonzero(coverage[row] > 0)
        for left, right in itertools.pairwise(columns):
            filled += _interpolate_horizontal(image, coverage, source_map, row, int(left), int(right), max_gap)
    for column in range(image.shape[1]):
        rows = np.flatnonzero(coverage[:, column] > 0)
        for top, bottom in itertools.pairwise(rows):
            filled += _interpolate_vertical(image, coverage, source_map, column, int(top), int(bottom), max_gap)
    return filled


def _interpolate_horizontal(
    image: np.ndarray,
    coverage: np.ndarray,
    source_map: np.ndarray,
    row: int,
    left: int,
    right: int,
    max_gap: int,
) -> int:
    gap = right - left - 1
    if gap <= 0 or gap > max_gap:
        return 0
    alpha = np.arange(1, gap + 1, dtype=np.float32) / (gap + 1)
    image[row, left + 1 : right] = (
        image[row, left].astype(np.float32) * (1.0 - alpha[:, None])
        + image[row, right].astype(np.float32) * alpha[:, None]
    ).astype(np.uint8)
    coverage[row, left + 1 : right] = 255
    source_map[row, left + 1 : right] = source_map[row, left]
    return gap


def _interpolate_vertical(
    image: np.ndarray,
    coverage: np.ndarray,
    source_map: np.ndarray,
    column: int,
    top: int,
    bottom: int,
    max_gap: int,
) -> int:
    gap = bottom - top - 1
    if gap <= 0 or gap > max_gap:
        return 0
    alpha = np.arange(1, gap + 1, dtype=np.float32) / (gap + 1)
    image[top + 1 : bottom, column] = (
        image[top, column].astype(np.float32) * (1.0 - alpha[:, None])
        + image[bottom, column].astype(np.float32) * alpha[:, None]
    ).astype(np.uint8)
    coverage[top + 1 : bottom, column] = 255
    source_map[top + 1 : bottom, column] = source_map[top, column]
    return gap

from __future__ import annotations

import cv2
import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import CanvasModel


class PanoramaCanvasBuilder:
    def __init__(self, config: PanoramaConfig) -> None:
        self.config = config

    def build(self, frame_shapes: list[tuple[int, int]], homographies: list[np.ndarray]) -> CanvasModel:
        all_corners = []
        for (height, width), homography in zip(frame_shapes, homographies, strict=True):
            corners = np.float32(
                [[0, 0], [width, 0], [width, height], [0, height]]
            ).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners, homography)
            all_corners.append(warped_corners.reshape(-1, 2))

        stacked = np.vstack(all_corners)
        min_x, min_y = np.floor(stacked.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(stacked.max(axis=0)).astype(int)
        width = int(max_x - min_x)
        height = int(max_y - min_y)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid canvas size computed: {width}x{height}")
        if width > self.config.max_canvas_width or height > self.config.max_canvas_height:
            raise RuntimeError(
                f"Canvas size {width}x{height} exceeds limits "
                f"{self.config.max_canvas_width}x{self.config.max_canvas_height}"
            )
        offset = np.array(
            [[1.0, 0.0, -float(min_x)], [0.0, 1.0, -float(min_y)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return CanvasModel(width=width, height=height, offset_matrix=offset, global_homographies=homographies)

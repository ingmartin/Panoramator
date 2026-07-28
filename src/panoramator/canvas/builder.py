from __future__ import annotations

import cv2
import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import CanvasModel
from panoramator.projection.models import PlanarProjection, Projection


class PanoramaCanvasBuilder:
    def __init__(self, config: PanoramaConfig) -> None:
        self.config = config

    def build(
        self,
        frame_shapes: list[tuple[int, int]],
        homographies: list[np.ndarray],
        projection: Projection | None = None,
    ) -> CanvasModel:
        projection = projection or PlanarProjection()
        all_corners = []
        for (height, width), homography in zip(frame_shapes, homographies, strict=True):
            corners = self._frame_contour(height, width, curved=projection.name != "planar")
            if projection.name == "planar":
                warped_corners = cv2.perspectiveTransform(corners, homography)
                all_corners.append(warped_corners.reshape(-1, 2))
            else:
                # Geometry for a curved panorama is estimated in local surface
                # coordinates.  The global transform therefore follows the local
                # projection: H(P(x)), not P(H(x)).
                surface_corners = projection.project_points(corners.reshape(-1, 2))
                warped_corners = cv2.perspectiveTransform(
                    surface_corners.astype(np.float32).reshape(-1, 1, 2), homography
                )
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
        return CanvasModel(width=width, height=height, offset_matrix=offset, global_homographies=homographies, projection=projection)

    def _frame_contour(self, height: int, width: int, curved: bool) -> np.ndarray:
        if not curved:
            points = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
        else:
            count = self.config.projection_contour_samples
            top = np.column_stack((np.linspace(0, width, count), np.zeros(count)))
            right = np.column_stack((np.full(count, width), np.linspace(0, height, count)))
            bottom = np.column_stack((np.linspace(width, 0, count), np.full(count, height)))
            left = np.column_stack((np.zeros(count), np.linspace(height, 0, count)))
            points = np.vstack((top, right, bottom, left))
        return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

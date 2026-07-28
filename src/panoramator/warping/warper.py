from __future__ import annotations

import cv2
import numpy as np

from panoramator.domain.models import CanvasModel, Frame
from panoramator.projection.models import PlanarProjection


class FrameWarper:
    def warp(self, frame: Frame, homography: np.ndarray, canvas: CanvasModel) -> tuple[np.ndarray, np.ndarray]:
        projection = canvas.projection or PlanarProjection()
        if projection.name != "planar":
            return self._warp_curved(frame, homography, canvas)
        transform = canvas.offset_matrix @ homography
        warped = cv2.warpPerspective(frame.image, transform, (canvas.width, canvas.height))
        mask = np.ones(frame.image.shape[:2], dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(
            mask,
            transform,
            (canvas.width, canvas.height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        return warped, warped_mask

    @staticmethod
    def _warp_curved(frame: Frame, homography: np.ndarray, canvas: CanvasModel) -> tuple[np.ndarray, np.ndarray]:
        projection = canvas.projection
        assert projection is not None
        inverse = np.linalg.inv(homography)
        warped = np.zeros((canvas.height, canvas.width, *frame.image.shape[2:]), dtype=frame.image.dtype)
        mask = np.full(frame.image.shape[:2], 255, dtype=np.uint8)
        warped_mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
        row_block = 256
        for top in range(0, canvas.height, row_block):
            bottom = min(top + row_block, canvas.height)
            yy, xx = np.indices((bottom - top, canvas.width), dtype=np.float64)
            surface = np.stack(
                (xx - canvas.offset_matrix[0, 2], yy + top - canvas.offset_matrix[1, 2]), axis=-1
            )
            homogeneous = np.concatenate(
                (surface.reshape(-1, 2), np.ones((surface.size // 2, 1))), axis=1
            )
            local_homogeneous = homogeneous @ inverse.T
            denominator = local_homogeneous[:, 2]
            valid = np.isfinite(local_homogeneous).all(axis=1) & (np.abs(denominator) > 1e-8)
            local_surface = np.full((len(homogeneous), 2), np.nan, dtype=np.float64)
            local_surface[valid] = local_homogeneous[valid, :2] / denominator[valid, None]
            source = np.full((len(homogeneous), 2), -1.0, dtype=np.float64)
            valid_indices = np.flatnonzero(valid)
            if len(valid_indices):
                surface_valid = projection.valid_surface_points(local_surface[valid_indices])
                valid[valid_indices[~surface_valid]] = False
            source[valid] = projection.unproject_points(local_surface[valid])
            valid &= np.isfinite(source).all(axis=1)
            map_x = source[:, 0].reshape(bottom - top, canvas.width).astype(np.float32)
            map_y = source[:, 1].reshape(bottom - top, canvas.width).astype(np.float32)
            warped[top:bottom] = cv2.remap(
                frame.image, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT
            )
            block_mask = cv2.remap(
                mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
            )
            block_mask.reshape(-1)[~valid] = 0
            warped_mask[top:bottom] = block_mask
        return warped, warped_mask

from __future__ import annotations

import cv2
import numpy as np

from panoramator.domain.models import CanvasModel, Frame


class FrameWarper:
    def warp(self, frame: Frame, homography: np.ndarray, canvas: CanvasModel) -> tuple[np.ndarray, np.ndarray]:
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

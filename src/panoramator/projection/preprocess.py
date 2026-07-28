from __future__ import annotations

import cv2
import numpy as np

from panoramator.domain.models import Frame
from panoramator.projection.models import Projection


def project_frame_for_geometry(frame: Frame, projection: Projection) -> Frame:
    """Return a feature-only frame expressed in the selected projection's coordinates.

    The original frame remains untouched for the final one-pass render.  Keeping the
    projected image at the original size also makes feature coordinates directly
    usable as local coordinates of the projected surface.
    """
    if projection.name == "planar":
        return frame
    image = _remap_to_projection(frame.image, projection)
    feature_image = None
    if frame.feature_image is not None:
        feature_image = cv2.resize(
            image,
            (frame.feature_image.shape[1], frame.feature_image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return Frame(frame.index, frame.timestamp_seconds, image, feature_image)


def _remap_to_projection(image: np.ndarray, projection: Projection) -> np.ndarray:
    height, width = image.shape[:2]
    y, x = np.indices((height, width), dtype=np.float64)
    surface = np.stack((x, y), axis=-1)
    source = projection.unproject_points(surface)
    map_x = source[..., 0].astype(np.float32)
    map_y = source[..., 1].astype(np.float32)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
    )

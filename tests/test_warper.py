from __future__ import annotations

import numpy as np

from panoramator.domain.models import CanvasModel, Frame
from panoramator.warping.warper import FrameWarper


def test_warp_projects_image_and_mask_to_canvas() -> None:
    frame = Frame(index=0, timestamp_seconds=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8))
    frame.image[1:3, 1:3] = 255
    canvas = CanvasModel(
        width=8,
        height=8,
        offset_matrix=np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        global_homographies=[],
    )

    warped, mask = FrameWarper().warp(frame, np.eye(3, dtype=np.float64), canvas)

    assert warped.shape == (8, 8, 3)
    assert mask.shape == (8, 8)
    assert mask.max() == 255
    assert warped[2:4, 3:5].mean() > 0


def test_warp_uses_nearest_neighbor_interpolation_for_mask() -> None:
    frame = Frame(index=0, timestamp_seconds=0.0, image=np.zeros((2, 2, 3), dtype=np.uint8))
    canvas = CanvasModel(
        width=4,
        height=4,
        offset_matrix=np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float64),
        global_homographies=[],
    )

    _, mask = FrameWarper().warp(frame, np.eye(3, dtype=np.float64), canvas)

    assert set(np.unique(mask)).issubset({0, 255})

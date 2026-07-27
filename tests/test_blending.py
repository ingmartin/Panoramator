import numpy as np
import pytest

from panoramator.blending.overlay import AverageBlender
from panoramator.config.models import PanoramaConfig


def test_blender_returns_image_with_same_shape() -> None:
    config = PanoramaConfig(feather_blend_kernel=9, seam_blur_kernel=5, seam_band_width=5)
    blender = AverageBlender(config)

    left = np.zeros((20, 20, 3), dtype=np.uint8)
    left[:, :12] = 255
    right = np.zeros((20, 20, 3), dtype=np.uint8)
    right[:, 8:] = 200

    left_mask = np.zeros((20, 20), dtype=np.uint8)
    left_mask[:, :12] = 255
    right_mask = np.zeros((20, 20), dtype=np.uint8)
    right_mask[:, 8:] = 255

    blended = blender.blend([left, right], [left_mask, right_mask])

    assert blended.shape == left.shape
    assert blended.dtype == np.uint8
    assert np.any(blended[:, 9:11] > 0)


def test_blender_rejects_empty_frame_sequence() -> None:
    with pytest.raises(RuntimeError, match="No warped frames"):
        AverageBlender(PanoramaConfig()).blend([], [])


def test_weight_map_preserves_interior_weight() -> None:
    config = PanoramaConfig(feather_blend_kernel=9, seam_blur_kernel=1, seam_band_width=3)
    blender = AverageBlender(config)
    mask = np.zeros((25, 25), dtype=np.uint8)
    mask[3:22, 3:22] = 255

    weight = blender._weight_map(mask)

    assert weight[12, 12] > 0.95
    assert weight[3, 3] < weight[12, 12]


def test_blender_can_bias_overlap_toward_sharper_frame() -> None:
    config = PanoramaConfig(feather_blend_kernel=9, seam_blur_kernel=1, seam_band_width=3, overlap_sharpness_weight=0.5)
    blender = AverageBlender(config)

    sharp = np.zeros((24, 24, 3), dtype=np.uint8)
    sharp[:, :16] = 255
    soft = sharp.copy()
    soft = np.clip((soft.astype(np.float32) * 0.65), 0, 255).astype(np.uint8)

    left_mask = np.zeros((24, 24), dtype=np.uint8)
    left_mask[:, :16] = 255
    right_mask = np.zeros((24, 24), dtype=np.uint8)
    right_mask[:, 8:] = 255

    blended = blender.blend([sharp, soft], [left_mask, right_mask], [400.0, 100.0])

    assert blended[12, 10, 0] > blended[12, 14, 0]

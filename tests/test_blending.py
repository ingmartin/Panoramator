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


def test_blender_can_keep_a_single_sharp_source_in_overlap() -> None:
    config = PanoramaConfig(feather_blend_kernel=1, seam_blur_kernel=1, overlap_sharpness_weight=1.0)
    blender = AverageBlender(config)
    first = np.full((8, 8, 3), 40, dtype=np.uint8)
    second = np.full((8, 8, 3), 200, dtype=np.uint8)
    mask = np.full((8, 8), 255, dtype=np.uint8)

    blended = blender.blend([first, second], [mask, mask], [100.0, 200.0], prefer_sharp_source=True)

    assert np.all(blended == 200)


def test_seam_blender_respects_disabled_photometric_normalization() -> None:
    blender = AverageBlender(
        PanoramaConfig(enable_photometric_normalization=False, rotation_min_new_coverage_ratio=0.0)
    )
    first = np.full((8, 8, 3), 40, dtype=np.uint8)
    second = np.full((8, 8, 3), 180, dtype=np.uint8)
    first_mask = np.zeros((8, 8), dtype=np.uint8)
    first_mask[:, :6] = 255
    second_mask = np.zeros((8, 8), dtype=np.uint8)
    second_mask[:, 2:] = 255

    blender.blend([first, second], [first_mask, second_mask], prefer_sharp_source=True)

    assert blender.last_photometric_metrics == []


def test_seam_blender_anchors_global_photometric_offsets_to_first_frame() -> None:
    blender = AverageBlender(
        PanoramaConfig(
            photometric_offset_limit=40.0,
            feather_blend_kernel=1,
            enable_global_photometric_normalization=True,
        )
    )
    first = np.full((10, 10, 3), 100, dtype=np.uint8)
    second = np.full((10, 10, 3), 130, dtype=np.uint8)
    mask = np.full((10, 10), 255, dtype=np.uint8)

    blender.blend([first, second], [mask, mask], prefer_sharp_source=True)

    assert len(blender.last_global_photometric_metrics) == 2
    assert blender.last_global_photometric_metrics[0]["offset_b"] == 0.0
    assert blender.last_global_photometric_metrics[1]["offset_b"] == pytest.approx(-30.0)

from __future__ import annotations

import numpy as np

from panoramator.blending.overlay import AverageBlender
from panoramator.config.models import PanoramaConfig
from panoramator.geometry.trajectory import stabilize_rotation_trajectory
from panoramator.postprocess.crop import crop_with_policy


def test_curved_auto_crop_uses_bounding_policy() -> None:
    image = np.full((10, 20, 3), 200, dtype=np.uint8)
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[1:9, 1:19] = 255
    mask[1:4, 1:5] = 0

    cropped, policy, loss = crop_with_policy(
        image, mask, "inscribed_rectangle", max_inscribed_loss=0.1, max_inscribed_width_loss=0.1
    )

    assert policy == "bounding_fallback_excessive_inscribed_loss"
    assert cropped.shape[:2] == (8, 18)
    assert loss > 0.1


def test_preserve_alpha_crop_keeps_visible_mask() -> None:
    image = np.full((5, 5, 3), 200, dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 255

    cropped, policy, _ = crop_with_policy(image, mask, "preserve_alpha", max_inscribed_loss=0.5, max_inscribed_width_loss=0.5)

    assert policy == "preserve_alpha"
    assert cropped.shape == (3, 3, 4)
    assert np.all(cropped[..., 3] == 255)


def test_rotation_stabilization_preserves_horizontal_panorama_extent() -> None:
    pairs = [
        np.array([[1.0, -0.03, 20.0], [0.03, 1.0, 2.0], [0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.02, 20.0], [-0.02, 1.0, -2.0], [0.0, 0.0, 1.0]]),
    ]

    stabilized = stabilize_rotation_trajectory(pairs, PanoramaConfig(trajectory_smoothing_window=3))

    assert len(stabilized.homographies) == 3
    assert stabilized.homographies[-1][0, 2] == 40.0
    assert "smoothed_vertical_shift" in stabilized.diagnostics


def test_seam_blender_chooses_single_source_in_overlap() -> None:
    blender = AverageBlender(PanoramaConfig(feather_blend_kernel=1))
    first = np.full((6, 6, 3), 10, dtype=np.uint8)
    second = np.full((6, 6, 3), 200, dtype=np.uint8)
    first_mask = np.zeros((6, 6), dtype=np.uint8)
    first_mask[:, :5] = 255
    second_mask = np.zeros((6, 6), dtype=np.uint8)
    second_mask[:, 1:] = 255

    result = blender.blend([first, second], [first_mask, second_mask], [10.0, 100.0], prefer_sharp_source=True)

    assert np.all(result[:, -1] == 200)
    assert blender.last_seam_metrics[0]["overlap_pixels"] == 24.0

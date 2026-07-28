from __future__ import annotations

import numpy as np

from panoramator.blending.overlay import AverageBlender
from panoramator.application.use_cases import PanoramaBuilder, _ChainBuildResult
from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import Frame, FrameQuality, SelectedFrame
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


def test_rotation_chain_is_decimated_by_geometric_baseline() -> None:
    frames = [
        SelectedFrame(Frame(index, 0.0, np.zeros((4, 4, 3), dtype=np.uint8)), FrameQuality(1.0, 1.0, True, "ok"))
        for index in range(4)
    ]
    pairs = [
        np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ]
    chain = _ChainBuildResult("orb", 1, ["orb"], [1], frames, [], frames, pairs, [])

    decimated, metrics = PanoramaBuilder(PanoramaConfig(rotation_min_baseline_px=15.0))._decimate_rotation_chain(chain)

    assert [item.frame.index for item in decimated.filtered_frames] == [0, 3]
    assert len(decimated.pairwise_homographies) == 1
    assert metrics[1]["decision"] == "rejected_insufficient_baseline"


def test_rotation_baseline_default_is_conservative_for_handheld_capture() -> None:
    assert PanoramaConfig().rotation_min_baseline_px == 12.0


def test_rotation_blender_skips_frame_with_negligible_new_coverage() -> None:
    blender = AverageBlender(PanoramaConfig(rotation_min_new_coverage_ratio=0.2, feather_blend_kernel=1))
    first = np.full((10, 10, 3), 20, dtype=np.uint8)
    second = np.full((10, 10, 3), 200, dtype=np.uint8)
    first_mask = np.full((10, 10), 255, dtype=np.uint8)
    second_mask = first_mask.copy()
    second_mask[0, 0] = 0

    result = blender.blend([first, second], [first_mask, second_mask], prefer_sharp_source=True)

    assert np.all(result == 20)
    assert blender.last_seam_metrics[0]["decision"] == -1.0


def test_rotation_blender_reduces_overlap_colour_error() -> None:
    blender = AverageBlender(PanoramaConfig(rotation_min_new_coverage_ratio=0.0, feather_blend_kernel=1))
    first = np.full((10, 10, 3), 100, dtype=np.uint8)
    second = np.full((10, 10, 3), 130, dtype=np.uint8)
    first_mask = np.zeros((10, 10), dtype=np.uint8)
    first_mask[:, :8] = 255
    second_mask = np.zeros((10, 10), dtype=np.uint8)
    second_mask[:, 2:] = 255

    blender.blend([first, second], [first_mask, second_mask], prefer_sharp_source=True)

    metric = blender.last_photometric_metrics[0]
    assert metric["applied"] == 1.0
    assert metric["error_after"] < metric["error_before"]

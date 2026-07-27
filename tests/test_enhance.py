import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import Frame, FrameQuality, SelectedFrame
from panoramator.postprocess.enhance import (
    apply_final_sharpening,
    normalize_selected_frames,
)


def _selected_frame(index: int, image: np.ndarray) -> SelectedFrame:
    return SelectedFrame(
        frame=Frame(index=index, timestamp_seconds=float(index), image=image),
        quality=FrameQuality(
            sharpness=10.0 + index,
            difference_score=float(index),
            accepted=True,
            reason="selected",
        ),
    )


def test_apply_final_sharpening_returns_original_image_when_disabled() -> None:
    image = np.full((5, 5, 3), 120, dtype=np.uint8)
    config = PanoramaConfig(enable_final_sharpening=False)

    result = apply_final_sharpening(image, config)

    assert result is image


def test_apply_final_sharpening_emphasizes_edges_when_enabled() -> None:
    image = np.full((9, 9, 3), 80, dtype=np.uint8)
    image[3:6, 3:6] = 200
    config = PanoramaConfig(enable_final_sharpening=True, final_sharpen_strength=0.8, final_sharpen_sigma=1.0)

    result = apply_final_sharpening(image, config)

    assert result.dtype == np.uint8
    assert result.shape == image.shape
    assert not np.array_equal(result, image)
    assert int(result[4, 4, 0]) >= int(image[4, 4, 0])


def test_normalize_selected_frames_returns_original_list_when_disabled() -> None:
    frames = [
        _selected_frame(0, np.full((4, 4, 3), 50, dtype=np.uint8)),
        _selected_frame(1, np.full((4, 4, 3), 180, dtype=np.uint8)),
    ]

    result = normalize_selected_frames(frames, PanoramaConfig(enable_photometric_normalization=False))

    assert result is frames


def test_normalize_selected_frames_uses_offset_for_low_variance_images() -> None:
    frames = [
        _selected_frame(0, np.full((4, 4, 3), 30, dtype=np.uint8)),
        _selected_frame(1, np.full((4, 4, 3), 200, dtype=np.uint8)),
    ]

    normalized = normalize_selected_frames(frames, PanoramaConfig(photometric_smoothing=1.0))

    assert float(normalized[1].frame.image.mean()) == 176.0
    assert normalized[1].frame.image.dtype == np.uint8


def test_normalize_selected_frames_updates_reference_progressively() -> None:
    frames = [
        _selected_frame(0, np.full((4, 4, 3), 40, dtype=np.uint8)),
        _selected_frame(1, np.full((4, 4, 3), 200, dtype=np.uint8)),
        _selected_frame(2, np.full((4, 4, 3), 240, dtype=np.uint8)),
    ]

    normalized = normalize_selected_frames(frames, PanoramaConfig(photometric_smoothing=1.0))

    assert float(normalized[1].frame.image.mean()) == 176.0
    assert float(normalized[2].frame.image.mean()) == 216.0
    assert normalized[0] is frames[0]


def test_normalize_selected_frames_clamps_gain_and_offset_for_textured_images() -> None:
    reference = np.array(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[0, 0, 0], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    source = np.array(
        [
            [[120, 120, 120], [121, 121, 121]],
            [[120, 120, 120], [121, 121, 121]],
        ],
        dtype=np.uint8,
    )
    frames = [_selected_frame(0, reference), _selected_frame(1, source)]

    normalized = normalize_selected_frames(frames, PanoramaConfig(photometric_smoothing=1.0))

    assert normalized[1].frame.image.min() == 127
    assert normalized[1].frame.image.max() == 128
    assert normalized[1].frame.image.dtype == np.uint8


def test_normalize_selected_frames_reduces_luminance_gap_for_textured_images() -> None:
    reference = np.array(
        [
            [[20, 20, 20], [80, 80, 80]],
            [[20, 20, 20], [80, 80, 80]],
        ],
        dtype=np.uint8,
    )
    source = np.array(
        [
            [[100, 100, 100], [140, 140, 140]],
            [[100, 100, 100], [140, 140, 140]],
        ],
        dtype=np.uint8,
    )

    normalized = normalize_selected_frames(
        [_selected_frame(0, reference), _selected_frame(1, source)],
        PanoramaConfig(photometric_smoothing=1.0),
    )

    adjusted = normalized[1].frame.image
    assert abs(float(adjusted.mean()) - float(reference.mean())) < abs(float(source.mean()) - float(reference.mean()))
    assert float(adjusted.std()) > float(source.std())

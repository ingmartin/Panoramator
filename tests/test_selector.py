import cv2
import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import Frame
from panoramator.selection.selector import FrameSelector, _sharpness_score


def test_fixed_blur_threshold_can_reject_all_frames() -> None:
    config = PanoramaConfig(blur_threshold=1000.0, adaptive_blur_threshold=False)
    frames = [
        Frame(index=0, timestamp_seconds=0.0, image=np.full((20, 20, 3), 127, dtype=np.uint8)),
        Frame(index=1, timestamp_seconds=1.0, image=np.full((20, 20, 3), 127, dtype=np.uint8)),
    ]

    selected, rejected = FrameSelector(config).select(frames)

    assert len(selected) == 0
    assert len(rejected) == 2


def test_adaptive_blur_threshold_can_relax_fixed_threshold() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = 255
    shifted = np.zeros((32, 32, 3), dtype=np.uint8)
    shifted[6:22, 10:26] = 255

    config = PanoramaConfig(
        blur_threshold=1000.0,
        adaptive_blur_threshold=True,
        min_difference=0.0,
    )
    frames = [
        Frame(index=0, timestamp_seconds=0.0, image=image),
        Frame(index=1, timestamp_seconds=1.0, image=shifted),
    ]

    selected, rejected = FrameSelector(config).select(frames)

    assert len(selected) == 2
    assert len(rejected) == 0


def test_blur_rescue_sharpening_can_save_slightly_soft_frame() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[18:46, 18:46] = 255
    softened = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2, sigmaY=1.2)

    original_sharpness = _sharpness_score(softened)
    rescued_sharpness = _sharpness_score(
        cv2.addWeighted(
            softened,
            1.2,
            cv2.GaussianBlur(softened, (0, 0), sigmaX=1.0, sigmaY=1.0),
            -0.2,
            0,
        )
    )
    threshold = (original_sharpness + rescued_sharpness) / 2.0

    config = PanoramaConfig(
        blur_threshold=threshold,
        adaptive_blur_threshold=False,
        min_difference=0.0,
        blur_rescue_sharpen_strength=0.2,
        blur_rescue_sharpen_sigma=1.0,
    )
    frames = [Frame(index=0, timestamp_seconds=0.0, image=softened)]

    selected, rejected = FrameSelector(config).select(frames)

    assert len(selected) == 1
    assert len(rejected) == 0
    assert selected[0].quality.reason == "selected_sharpened"
    assert selected[0].quality.sharpness > original_sharpness


def test_blur_rescue_sharpening_preserves_feature_image() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[18:46, 18:46] = 255
    softened = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2, sigmaY=1.2)
    feature_image = cv2.resize(softened, (32, 32), interpolation=cv2.INTER_AREA)

    original_sharpness = _sharpness_score(softened)
    rescued_sharpness = _sharpness_score(
        cv2.addWeighted(
            softened,
            1.2,
            cv2.GaussianBlur(softened, (0, 0), sigmaX=1.0, sigmaY=1.0),
            -0.2,
            0,
        )
    )
    threshold = (original_sharpness + rescued_sharpness) / 2.0

    config = PanoramaConfig(blur_threshold=threshold, adaptive_blur_threshold=False, min_difference=0.0)
    frames = [Frame(index=0, timestamp_seconds=0.0, image=softened, feature_image=feature_image)]

    selected, _ = FrameSelector(config).select(frames)

    assert selected[0].frame.feature_image is not None
    assert selected[0].frame.feature_image.shape == feature_image.shape


def test_blur_rescue_sharpening_can_be_disabled() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[18:46, 18:46] = 255
    softened = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2, sigmaY=1.2)

    original_sharpness = _sharpness_score(softened)
    rescued_sharpness = _sharpness_score(
        cv2.addWeighted(
            softened,
            1.2,
            cv2.GaussianBlur(softened, (0, 0), sigmaX=1.0, sigmaY=1.0),
            -0.2,
            0,
        )
    )
    threshold = (original_sharpness + rescued_sharpness) / 2.0

    config = PanoramaConfig(
        blur_threshold=threshold,
        adaptive_blur_threshold=False,
        min_difference=0.0,
        enable_blur_rescue_sharpening=False,
    )
    frames = [Frame(index=0, timestamp_seconds=0.0, image=softened)]

    selected, rejected = FrameSelector(config).select(frames)

    assert len(selected) == 0
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "blur"


def test_frame_selection_window_prefers_sharpest_candidate() -> None:
    sharp = np.zeros((32, 32, 3), dtype=np.uint8)
    sharp[8:24, 8:24] = 255
    soft = cv2.GaussianBlur(sharp, (0, 0), sigmaX=2.0, sigmaY=2.0)

    config = PanoramaConfig(
        blur_threshold=1.0,
        adaptive_blur_threshold=False,
        min_difference=0.0,
        frame_selection_window_size=2,
    )
    frames = [
        Frame(index=0, timestamp_seconds=0.0, image=soft),
        Frame(index=1, timestamp_seconds=1.0, image=sharp),
    ]

    selected, rejected = FrameSelector(config).select(frames)

    assert [item.frame.index for item in selected] == [1]
    assert selected[0].quality.reason == "selected_windowed"
    assert any(item["reason"] == "not_sharpest_in_window" for item in rejected)
    assert len(selected[0].alternates) == 1

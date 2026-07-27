import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.io.video import OpenCVVideoSource


def test_target_frame_indices_cover_video_tail_when_limited() -> None:
    source = OpenCVVideoSource("dummy.mp4", PanoramaConfig(sampling_step=15, max_frames=5))
    indices = source._target_frame_indices(total_frame_count=548, step=15)

    assert len(indices) == 5
    assert indices[0] == 0
    assert indices[-1] == 540
    assert indices == sorted(indices)


def test_target_frame_indices_return_all_candidates_when_under_limit() -> None:
    source = OpenCVVideoSource("dummy.mp4", PanoramaConfig(sampling_step=20, max_frames=100))
    indices = source._target_frame_indices(total_frame_count=90, step=20)

    assert indices == [0, 20, 40, 60, 80]


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], fps: float = 30.0) -> None:
        self._frames = list(frames)
        self._fps = fps
        self._index = 0

    def get(self, prop_id: int) -> float:
        if prop_id == 5:
            return self._fps
        if prop_id == 7:
            return 0.0
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame


def test_iter_frames_falls_back_when_frame_count_is_unavailable() -> None:
    config = PanoramaConfig(sampling_step=2, max_frames=2)
    source = OpenCVVideoSource("dummy.mp4", config)
    source.capture = _FakeCapture([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(6)])

    frames = source.iter_frames()

    assert [frame.index for frame in frames] == [0, 2]
    assert len(frames) == 2


def test_prepare_images_can_keep_full_resolution_and_downscale_features() -> None:
    config = PanoramaConfig(downscale=1.0, feature_downscale=0.5)
    source = OpenCVVideoSource("dummy.mp4", config)
    image = np.zeros((10, 8, 3), dtype=np.uint8)

    base_image, feature_image = source._prepare_images(image)

    assert base_image.shape == (10, 8, 3)
    assert feature_image is not None
    assert feature_image.shape == (5, 4, 3)

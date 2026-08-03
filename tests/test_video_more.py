from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from panoramator.config.models import PanoramaConfig
from panoramator.io import video as video_module
from panoramator.io.video import OpenCVVideoSource


class _FakeCapture:
    def __init__(self, opened: bool = True, frames: list[np.ndarray] | None = None) -> None:
        self._opened = opened
        self._frames = list(frames or [])
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop_id: int) -> float:
        values = {
            video_module.cv2.CAP_PROP_FPS: 24.0,
            video_module.cv2.CAP_PROP_FRAME_COUNT: len(self._frames),
            video_module.cv2.CAP_PROP_FRAME_WIDTH: 8.0,
            video_module.cv2.CAP_PROP_FRAME_HEIGHT: 6.0,
        }
        return values.get(prop_id, 0.0)

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def test_open_reads_metadata_and_stores_capture(monkeypatch) -> None:
    fake_capture = _FakeCapture(frames=[np.zeros((6, 8, 3), dtype=np.uint8)])
    monkeypatch.setattr(video_module.cv2, "VideoCapture", lambda path: fake_capture)

    source = OpenCVVideoSource("video.mp4", PanoramaConfig())
    metadata = source.open()

    assert metadata.fps == 24.0
    assert metadata.frame_count == 1
    assert metadata.width == 8
    assert metadata.height == 6
    assert source.capture is fake_capture


def test_open_raises_for_unavailable_video(monkeypatch) -> None:
    capture = _FakeCapture(opened=False)
    monkeypatch.setattr(video_module.cv2, "VideoCapture", lambda path: capture)

    with pytest.raises(RuntimeError, match="Cannot open video"):
        OpenCVVideoSource("missing.mp4", PanoramaConfig()).open()

    assert capture.released is True


def test_open_closes_an_existing_capture(monkeypatch) -> None:
    previous_capture = _FakeCapture()
    new_capture = _FakeCapture()
    monkeypatch.setattr(video_module.cv2, "VideoCapture", lambda path: new_capture)
    source = OpenCVVideoSource("video.mp4", PanoramaConfig())
    cast(Any, source).capture = previous_capture

    source.open()

    assert previous_capture.released is True
    assert source.capture is new_capture


def test_iter_frames_requires_open_capture() -> None:
    with pytest.raises(RuntimeError, match="not open"):
        OpenCVVideoSource("video.mp4", PanoramaConfig()).iter_frames()


def test_iter_frames_returns_targeted_frames_when_count_is_known() -> None:
    config = PanoramaConfig(sampling_step=2, max_frames=2)
    source = OpenCVVideoSource("video.mp4", config)
    cast(Any, source).capture = _FakeCapture(frames=[np.full((4, 4, 3), fill_value=index, dtype=np.uint8) for index in range(5)])

    frames = source.iter_frames()

    assert [frame.index for frame in frames] == [0, 4]
    assert [frame.timestamp_seconds for frame in frames] == [0.0, 4 / 24]


def test_prepare_images_downscales_base_image() -> None:
    source = OpenCVVideoSource("video.mp4", PanoramaConfig(downscale=0.5, feature_downscale=1.0))
    image = np.zeros((10, 8, 3), dtype=np.uint8)

    base_image, feature_image = source._prepare_images(image)

    assert base_image.shape == (5, 4, 3)
    assert feature_image is None


def test_target_frame_indices_return_empty_for_unknown_count() -> None:
    source = OpenCVVideoSource("video.mp4", PanoramaConfig())

    assert source._target_frame_indices(0, 5) == []


def test_close_releases_existing_capture() -> None:
    source = OpenCVVideoSource("video.mp4", PanoramaConfig())
    capture = _FakeCapture()
    cast(Any, source).capture = capture

    source.close()

    assert capture.released is True
    assert source.capture is None

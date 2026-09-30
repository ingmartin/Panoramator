from pathlib import Path

import cv2
import numpy as np
import pytest

from panoramator.io.frames import (
    DirectoryFrameSource,
    FrameDecodeError,
    decode_frame_input,
)


def test_directory_frame_source_is_sorted_and_filters_non_images(tmp_path: Path) -> None:
    for name in ("10.PNG", "2.jpg", "1.jpeg", "ignore.txt"):
        (tmp_path / name).write_bytes(b"placeholder")

    source = DirectoryFrameSource(tmp_path)

    assert [Path(item).name for item in source] == ["1.jpeg", "10.PNG", "2.jpg"]
    assert len(source) == 3


def test_directory_frame_source_does_not_decode_until_iteration(tmp_path: Path, monkeypatch) -> None:
    image = np.full((3, 4, 3), 80, dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "frame.png"), image)
    calls: list[str] = []
    original = cv2.imread

    def _read(path, flags):
        calls.append(path)
        return original(path, flags)

    monkeypatch.setattr("panoramator.io.frames.cv2.imread", _read)
    source = DirectoryFrameSource(tmp_path)

    assert calls == []
    assert next(iter(source)) == tmp_path / "frame.png"
    assert calls == []


def test_directory_frame_source_reports_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        DirectoryFrameSource(tmp_path / "missing")


def test_decode_frame_input_preserves_bgr_array() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    assert decode_frame_input(image) is image


@pytest.mark.parametrize(
    "image",
    [np.zeros((2, 3), dtype=np.uint8), np.zeros((2, 3, 4), dtype=np.uint8)],
)
def test_decode_frame_input_rejects_non_bgr_arrays(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="H x W x 3"):
        decode_frame_input(image)


def test_decode_frame_input_rejects_broken_image_path(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")

    with pytest.raises(FrameDecodeError, match="Cannot decode"):
        decode_frame_input(broken)

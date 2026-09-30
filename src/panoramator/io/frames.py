"""Frame inputs and lazy frame sources for the panorama builder."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar, Protocol, TypeAlias

import cv2
import numpy as np

FrameInput: TypeAlias = str | Path | np.ndarray


class FrameSource(Protocol):
    """A restart-free, iterable source of BGR frame inputs."""

    def __iter__(self) -> Iterator[FrameInput]: ...

    def __len__(self) -> int: ...


class FrameDecodeError(ValueError):
    """Raised when a path cannot be decoded as an image."""


class DirectoryFrameSource:
    """Lazily enumerate JPEG and PNG files in a directory."""

    _EXTENSIONS: ClassVar[set[str]] = {".jpg", ".jpeg", ".png"}

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Frame directory does not exist: {self.directory}")
        if not self.directory.is_dir():
            raise NotADirectoryError(f"Frame source is not a directory: {self.directory}")
        self._paths = tuple(
            sorted(
                (
                    path
                    for path in self.directory.iterdir()
                    if path.is_file() and path.suffix.lower() in self._EXTENSIONS
                ),
                key=lambda path: path.name.casefold(),
            )
        )

    def __iter__(self) -> Iterator[FrameInput]:
        yield from self._paths

    def __len__(self) -> int:
        return len(self._paths)


def decode_frame_input(value: FrameInput) -> np.ndarray:
    """Return a validated BGR image without copying NumPy inputs."""
    if isinstance(value, np.ndarray):
        _validate_frame_array(value)
        return value
    if isinstance(value, (str, Path)):
        image = cv2.imread(str(value), cv2.IMREAD_COLOR)
        if image is None:
            raise FrameDecodeError(f"Cannot decode frame image: {value}")
        _validate_frame_array(image)
        return image
    raise TypeError("Frame input must be a path or a H x W x 3 NumPy array")


def _validate_frame_array(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Frame array must have shape H x W x 3 in BGR channel order")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("Frame array must not be empty")

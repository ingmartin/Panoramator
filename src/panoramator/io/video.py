from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import Frame, VideoMetadata


class OpenCVVideoSource:
    def __init__(self, path: str | Path, config: PanoramaConfig) -> None:
        self.path = Path(path)
        self.config = config
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> VideoMetadata:
        self.close()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open video: {self.path}")
        self.capture = capture
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return VideoMetadata(self.path, fps, frame_count, width, height)

    def iter_frames(self) -> list[Frame]:
        if self.capture is None:
            raise RuntimeError("Video source is not open")
        step = max(1, self.config.sampling_step)
        fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 1.0)
        total_frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_indices = self._target_frame_indices(total_frame_count, step)
        if not target_indices:
            return self._iter_frames_without_frame_count(step, fps)
        frames: list[Frame] = []
        target_set = set(target_indices)
        frame_index = 0
        while True:
            ok, image = self.capture.read()
            if not ok:
                break
            if frame_index not in target_set:
                frame_index += 1
                continue
            frames.append(Frame(frame_index, frame_index / max(fps, 1e-6), *self._prepare_images(image)))
            frame_index += 1
            if len(frames) >= len(target_indices):
                break
        return frames

    def _iter_frames_without_frame_count(self, step: int, fps: float) -> list[Frame]:
        if self.capture is None:
            raise RuntimeError("Video source is not open")
        frames: list[Frame] = []
        frame_index = 0
        max_frames = max(1, self.config.max_frames)
        while True:
            ok, image = self.capture.read()
            if not ok:
                break
            if frame_index % step != 0:
                frame_index += 1
                continue
            frames.append(Frame(frame_index, frame_index / max(fps, 1e-6), *self._prepare_images(image)))
            frame_index += 1
            if len(frames) >= max_frames:
                break
        return frames

    def _prepare_images(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        base_image = image
        if self.config.downscale != 1.0:
            base_image = cv2.resize(
                image,
                None,
                fx=self.config.downscale,
                fy=self.config.downscale,
                interpolation=cv2.INTER_AREA,
            )
        feature_image = None
        if self.config.feature_downscale != 1.0:
            feature_image = cv2.resize(
                base_image,
                None,
                fx=self.config.feature_downscale,
                fy=self.config.feature_downscale,
                interpolation=cv2.INTER_AREA,
            )
        return base_image, feature_image

    def _target_frame_indices(self, total_frame_count: int, step: int) -> list[int]:
        if total_frame_count <= 0:
            return []

        candidate_indices = list(range(0, total_frame_count, step))
        if not candidate_indices:
            return [0]

        max_frames = max(1, self.config.max_frames)
        if len(candidate_indices) <= max_frames:
            return candidate_indices

        positions = np.linspace(0, len(candidate_indices) - 1, num=max_frames)
        sampled_positions = np.rint(positions).astype(int)
        unique_positions = []
        seen = set()
        for position in sampled_positions:
            value = int(position)
            if value in seen:
                continue
            seen.add(value)
            unique_positions.append(value)

        selected_indices = [candidate_indices[position] for position in unique_positions]
        if selected_indices[-1] != candidate_indices[-1]:
            selected_indices[-1] = candidate_indices[-1]
        return selected_indices

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

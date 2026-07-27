from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import SelectedFrame

_STD_EPSILON = 1.0
_MIN_GAIN = 0.85
_MAX_GAIN = 1.15
_MAX_OFFSET = 24.0


def apply_final_sharpening(image: np.ndarray, config: PanoramaConfig) -> np.ndarray:
    if not config.enable_final_sharpening or config.final_sharpen_strength <= 0:
        return image
    return _unsharp_mask(image, config.final_sharpen_strength, config.final_sharpen_sigma)


def normalize_selected_frames(selected_frames: list[SelectedFrame], config: PanoramaConfig) -> list[SelectedFrame]:
    if not config.enable_photometric_normalization or len(selected_frames) < 2:
        return selected_frames

    normalized = [selected_frames[0]]
    reference_image = selected_frames[0].frame.image
    for item in selected_frames[1:]:
        adjusted = _match_luminance(item.frame.image, reference_image, config.photometric_smoothing)
        normalized_item = replace(item, frame=replace(item.frame, image=adjusted))
        normalized.append(normalized_item)
        reference_image = adjusted
    return normalized


def _match_luminance(image: np.ndarray, reference: np.ndarray, smoothing: float) -> np.ndarray:
    source_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)

    source_mean = float(source_gray.mean())
    source_std = float(source_gray.std())
    reference_mean = float(reference_gray.mean())
    reference_std = float(reference_gray.std())

    if source_std < _STD_EPSILON or reference_std < _STD_EPSILON:
        offset = np.clip(reference_mean - source_mean, -_MAX_OFFSET, _MAX_OFFSET)
        adjusted = image.astype(np.float32) + offset * smoothing
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    gain = reference_std / max(source_std, 1e-6)
    offset = reference_mean - source_mean * gain
    gain = float(np.clip(gain, _MIN_GAIN, _MAX_GAIN))
    gain = 1.0 + (gain - 1.0) * smoothing
    offset = float(np.clip(offset * smoothing, -_MAX_OFFSET, _MAX_OFFSET))

    adjusted = image.astype(np.float32) * gain + offset
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _unsharp_mask(image: np.ndarray, strength: float, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

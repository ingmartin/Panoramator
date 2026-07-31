from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from panoramator.domain.models import Frame

from .models import SurfaceKind, UnwrapConfig, UnwrapStatus
from .segmentation import masked_sharpness, object_mask, publish_surface_mask, stable_surface_bbox


@dataclass(slots=True)
class AnalyzedFrame:
    frame: Frame
    geometry_mask: np.ndarray
    publish_mask: np.ndarray
    sharpness: float
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class Analysis:
    frames: list[AnalyzedFrame]
    kind: SurfaceKind
    status: UnwrapStatus | None = None
    message: str = ""
    recommendation: str = ""
    rejected_frames: list[dict[str, float | int | str]] | None = None
    measurements: dict[str, float | int] | None = None


class VideoAnalyzer:
    def analyze(self, frames: list[Frame], config: UnwrapConfig) -> Analysis:
        candidates: list[AnalyzedFrame] = []
        for frame in frames:
            geometry_mask = object_mask(frame.image, config.min_object_area_ratio)
            if geometry_mask is None:
                continue
            bbox = stable_surface_bbox(geometry_mask)
            if bbox is None:
                continue
            publish_mask = publish_surface_mask(geometry_mask, bbox)
            x, y, width, height = bbox
            candidates.append(
                AnalyzedFrame(
                    frame,
                    geometry_mask,
                    publish_mask,
                    masked_sharpness(frame.image, publish_mask),
                    (x, y, width, height),
                )
            )
        if len(candidates) < 2:
            return Analysis([], config.surface_kind, UnwrapStatus.OBJECT_NOT_DETECTED,
                            "The foreground surface cannot be separated reliably.",
                            "Record the surface larger in frame with stronger background contrast.")
        sharp = [item for item in candidates if item.sharpness >= config.blur_threshold]
        if len(sharp) < 2:
            return Analysis([], config.surface_kind, UnwrapStatus.EXCESSIVE_MOTION_BLUR,
                            "Too few sharp frames are available for a reliable texture.",
                            "Move more slowly and keep focus fixed.")
        selected, rejected, decimation_measurements = self._temporal_decimation(sharp, config)
        if len(selected) < 2:
            selected = [sharp[0], sharp[-1]]
        kind = config.surface_kind
        if kind is SurfaceKind.AUTO:
            # A stable near-rectangular silhouette selects the developable model;
            # ambiguous footage uses the conservative curved-surface fallback.
            ratios = [item.bbox[2] / max(item.bbox[3], 1) for item in selected]
            kind = SurfaceKind.CYLINDRICAL if 0.45 <= float(np.median(ratios)) <= 1.65 else SurfaceKind.CURVED
        return Analysis(selected, kind, rejected_frames=rejected, measurements=decimation_measurements)

    def _temporal_decimation(
        self,
        frames: list[AnalyzedFrame],
        config: UnwrapConfig,
    ) -> tuple[list[AnalyzedFrame], list[dict[str, float | int | str]], dict[str, float | int]]:
        if not config.enable_temporal_decimation or len(frames) <= 2:
            return frames, [], {
                "temporal_decimation_applied": int(config.enable_temporal_decimation),
                "temporal_decimation_kept_frames": len(frames),
                "temporal_decimation_rejected_frames": 0,
            }
        selected = [frames[0]]
        rejected: list[dict[str, float | int | str]] = []
        last_thumbnail, last_mask = _band_thumbnail(frames[0])
        observed_mask = last_mask.copy()
        observed_detail = _detail_energy(last_thumbnail, last_mask)
        for item in frames[1:]:
            thumbnail, mask = _band_thumbnail(item)
            mask_iou = _mask_iou(last_mask, mask)
            band_difference = _band_difference(last_thumbnail, last_mask, thumbnail, mask)
            bbox_shift = _bbox_shift_ratio(selected[-1].bbox, item.bbox)
            new_mask_fraction = _new_mask_fraction(observed_mask, mask)
            detail_gain = _detail_gain(thumbnail, mask, observed_mask)
            if (
                mask_iou >= config.temporal_decimation_max_mask_iou
                and band_difference <= config.temporal_decimation_min_band_difference
                and bbox_shift <= config.temporal_decimation_min_bbox_shift
            ):
                rejected.append(
                    {
                        "frame_index": item.frame.index,
                        "timestamp_seconds": item.frame.timestamp_seconds,
                        "reason": "temporal_decimation_near_duplicate",
                        "mask_iou": mask_iou,
                        "band_difference": band_difference,
                        "bbox_shift_ratio": bbox_shift,
                        "new_mask_fraction": new_mask_fraction,
                        "detail_gain": detail_gain,
                    }
                )
                continue
            if (
                new_mask_fraction < config.temporal_decimation_min_new_mask_fraction
                and detail_gain < config.temporal_decimation_min_detail_gain
            ):
                rejected.append(
                    {
                        "frame_index": item.frame.index,
                        "timestamp_seconds": item.frame.timestamp_seconds,
                        "reason": "temporal_decimation_low_surface_contribution",
                        "mask_iou": mask_iou,
                        "band_difference": band_difference,
                        "bbox_shift_ratio": bbox_shift,
                        "new_mask_fraction": new_mask_fraction,
                        "detail_gain": detail_gain,
                    }
                )
                continue
            selected.append(item)
            last_thumbnail, last_mask = thumbnail, mask
            observed_mask |= mask
            observed_detail += detail_gain
        if len(selected) < 2 and len(frames) >= 2:
            selected = [frames[0], frames[-1]]
            rejected = rejected[:-1] if rejected else rejected
        return selected, rejected, {
            "temporal_decimation_applied": 1,
            "temporal_decimation_kept_frames": len(selected),
            "temporal_decimation_rejected_frames": len(rejected),
            "temporal_decimation_observed_detail": round(float(observed_detail), 6),
        }


def _band_thumbnail(item: AnalyzedFrame) -> tuple[np.ndarray, np.ndarray]:
    x, y, width, height = item.bbox
    patch = item.frame.image[y : y + height, x : x + width]
    patch_mask = item.publish_mask[y : y + height, x : x + width]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (96, 64), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(patch_mask, (96, 64), interpolation=cv2.INTER_NEAREST)
    resized[resized_mask == 0] = 0
    return resized.astype(np.float32), resized_mask > 0


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = left | right
    if not np.any(union):
        return 0.0
    return float(np.count_nonzero(left & right) / np.count_nonzero(union))


def _band_difference(left: np.ndarray, left_mask: np.ndarray, right: np.ndarray, right_mask: np.ndarray) -> float:
    overlap = left_mask & right_mask
    if not np.any(overlap):
        return 1.0
    return float(np.mean(np.abs(left[overlap] - right[overlap])) / 255.0)


def _bbox_shift_ratio(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    left_center = left[0] + left[2] * 0.5
    right_center = right[0] + right[2] * 0.5
    mean_width = max((left[2] + right[2]) * 0.5, 1.0)
    return float(abs(right_center - left_center) / mean_width)


def _new_mask_fraction(observed_mask: np.ndarray, candidate_mask: np.ndarray) -> float:
    candidate_pixels = int(np.count_nonzero(candidate_mask))
    if candidate_pixels == 0:
        return 0.0
    new_pixels = candidate_mask & ~observed_mask
    return float(np.count_nonzero(new_pixels) / candidate_pixels)


def _detail_energy(thumbnail: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    gradient_x = cv2.Sobel(thumbnail, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(thumbnail, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    return float(np.mean(gradient[mask]) / 255.0)


def _detail_gain(thumbnail: np.ndarray, candidate_mask: np.ndarray, observed_mask: np.ndarray) -> float:
    new_pixels = candidate_mask & ~observed_mask
    if not np.any(new_pixels):
        return 0.0
    gradient_x = cv2.Sobel(thumbnail, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(thumbnail, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    return float(np.mean(gradient[new_pixels]) / 255.0)

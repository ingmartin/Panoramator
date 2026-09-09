from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from panoramator.domain.models import Frame

from .models import SurfaceKind, UnwrapConfig, UnwrapStatus
from .segmentation import (
    cylindrical_object_mask,
    masked_sharpness,
    object_mask,
    publish_surface_mask,
    stable_surface_bbox,
)


@dataclass(slots=True)
class AnalyzedFrame:
    frame: Frame
    geometry_mask: np.ndarray
    publish_mask: np.ndarray
    sharpness: float
    bbox: tuple[int, int, int, int]
    core_mask: np.ndarray | None = None
    nuisance_mask: np.ndarray | None = None


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
            geometry_mask = (
                cylindrical_object_mask(frame.image, config.min_object_area_ratio)
                if config.surface_kind is SurfaceKind.CYLINDRICAL
                else object_mask(frame.image, config.min_object_area_ratio)
            )
            if geometry_mask is None:
                continue
            if config.surface_kind is SurfaceKind.CYLINDRICAL:
                core_mask, nuisance_mask = geometry_mask.copy(), np.zeros_like(geometry_mask)
                points = cv2.findNonZero(core_mask)
                bbox = cv2.boundingRect(points) if points is not None else None
            else:
                core_mask, nuisance_mask = _core_body_masks(geometry_mask)
                bbox = stable_surface_bbox(core_mask)
            if bbox is None:
                continue
            publish_mask = publish_surface_mask(core_mask, bbox)
            x, y, width, height = bbox
            candidates.append(
                AnalyzedFrame(
                    frame,
                    geometry_mask,
                    publish_mask,
                    masked_sharpness(frame.image, publish_mask),
                    (x, y, width, height),
                    core_mask=core_mask,
                    nuisance_mask=nuisance_mask,
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
        # Cylindrical pose solving needs the ordered orbit samples.  The generic
        # surface decimator is intentionally conservative about new visible
        # pixels, but that criterion can discard useful texture from a rotating
        # cylinder before its angular motion has been estimated.  Curved/auto
        # analysis keeps the existing decimator and behavior.
        if config.surface_kind is SurfaceKind.CYLINDRICAL:
            selected, rejected, decimation_measurements = sharp, [], {
                "temporal_decimation_applied": 0,
                "temporal_decimation_kept_frames": len(sharp),
                "temporal_decimation_rejected_frames": 0,
            }
        else:
            selected, rejected, decimation_measurements = self._temporal_decimation(sharp, config)
        if len(selected) < 2:
            selected = [sharp[0], sharp[-1]]
        kind = config.surface_kind
        measurements = {**(decimation_measurements or {})}
        if kind is SurfaceKind.AUTO:
            family = _select_surface_family(selected)
            kind = SurfaceKind(str(family["surface_family_validated"]))
            measurements.update(family)
        else:
            measurements["surface_family_candidate"] = kind.value
            measurements["surface_family_confidence"] = 1.0
            measurements["surface_family_validated"] = kind.value
            measurements["surface_family_reason"] = "forced_by_config"
        return Analysis(selected, kind, rejected_frames=rejected, measurements=measurements)

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


def _core_body_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bbox = stable_surface_bbox(mask)
    if bbox is None:
        return mask.copy(), np.zeros_like(mask)
    _x, y, width, height = bbox
    rows = range(y, y + height)
    lefts: list[int] = []
    rights: list[int] = []
    widths: list[int] = []
    for row in rows:
        xs = np.flatnonzero(mask[row, :] > 0)
        if not xs.size:
            continue
        lefts.append(int(xs[0]))
        rights.append(int(xs[-1]))
        widths.append(int(xs[-1] - xs[0] + 1))
    if not widths:
        return mask.copy(), np.zeros_like(mask)
    target_width = max(int(np.percentile(widths, 55)), max(6, width // 2))
    centre = round((np.percentile(lefts, 60) + np.percentile(rights, 40)) * 0.5)
    half_width = max(target_width // 2, 3)
    core_left = max(0, centre - half_width)
    core_right = min(mask.shape[1] - 1, centre + half_width)
    core = np.zeros_like(mask)
    core[y : y + height, core_left : core_right + 1] = mask[y : y + height, core_left : core_right + 1]
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    nuisance = cv2.bitwise_and(mask, cv2.bitwise_not(core))
    return core, nuisance


def _select_surface_family(frames: list[AnalyzedFrame]) -> dict[str, float | str]:
    ratios = np.array([item.bbox[2] / max(item.bbox[3], 1) for item in frames], dtype=np.float32)
    widths = np.array([item.bbox[2] for item in frames], dtype=np.float32)
    axes = np.array([item.bbox[0] + item.bbox[2] * 0.5 for item in frames], dtype=np.float32)
    protrusions = np.array(
        [
            0.0
            if item.nuisance_mask is None or np.count_nonzero(item.geometry_mask) == 0
            else float(np.count_nonzero(item.nuisance_mask) / np.count_nonzero(item.geometry_mask))
            for item in frames
        ],
        dtype=np.float32,
    )
    sidewall_jitter = np.array([_sidewall_jitter(item.core_mask if item.core_mask is not None else item.publish_mask) for item in frames], dtype=np.float32)
    width_stability = np.array([_width_stability(item.core_mask if item.core_mask is not None else item.publish_mask, item.bbox) for item in frames], dtype=np.float32)

    aspect_median = float(np.median(ratios)) if ratios.size else 0.0
    aspect_score = _bounded_score(abs(aspect_median - 0.9), 0.8)
    width_score = _bounded_score(float(np.mean(width_stability)), 0.28)
    sidewall_score = _bounded_score(float(np.mean(sidewall_jitter)), 0.12)
    protrusion_score = _bounded_score(float(np.mean(protrusions)), 0.34)
    axis_score = _bounded_score(float(np.std(axes) / max(np.median(widths), 1.0)), 0.22)

    candidate = SurfaceKind.CYLINDRICAL if 0.3 <= aspect_median <= 1.95 else SurfaceKind.CURVED
    confidence = float(
        0.24 * aspect_score
        + 0.22 * width_score
        + 0.22 * sidewall_score
        + 0.18 * protrusion_score
        + 0.14 * axis_score
    )
    validated = SurfaceKind.CYLINDRICAL if candidate is SurfaceKind.CYLINDRICAL and confidence >= 0.54 else SurfaceKind.CURVED
    if validated is SurfaceKind.CYLINDRICAL:
        reason = "core_body_cylindrical_support"
    elif candidate is SurfaceKind.CYLINDRICAL:
        reason = "cylindrical_candidate_rejected_by_confidence"
    else:
        reason = "core_body_aspect_out_of_range"
    return {
        "surface_family_candidate": candidate.value,
        "surface_family_confidence": round(confidence, 6),
        "surface_family_validated": validated.value,
        "surface_family_reason": reason,
        "core_body_aspect_ratio": round(aspect_median, 6),
        "core_body_width_stability": round(float(np.mean(width_stability)), 6),
        "core_body_sidewall_jitter": round(float(np.mean(sidewall_jitter)), 6),
        "nuisance_region_ratio": round(float(np.mean(protrusions)), 6),
        "core_body_axis_stability": round(float(np.std(axes) / max(np.median(widths), 1.0)), 6),
    }


def _width_stability(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x, y, width, height = bbox
    widths: list[int] = []
    for row in range(y, y + height):
        xs = np.flatnonzero(mask[row, x : x + width] > 0)
        if xs.size:
            widths.append(int(xs[-1] - xs[0] + 1))
    if not widths:
        return 1.0
    return float(np.std(widths) / max(np.mean(widths), 1.0))


def _sidewall_jitter(mask: np.ndarray) -> float:
    lefts: list[int] = []
    rights: list[int] = []
    for row in range(mask.shape[0]):
        xs = np.flatnonzero(mask[row, :] > 0)
        if xs.size:
            lefts.append(int(xs[0]))
            rights.append(int(xs[-1]))
    if len(lefts) < 2:
        return 1.0
    width = max(np.median(np.array(rights) - np.array(lefts) + 1), 1.0)
    return float((np.std(lefts) + np.std(rights)) / (2.0 * width))


def _bounded_score(value: float, scale: float) -> float:
    return float(np.clip(1.0 - value / max(scale, 1e-6), 0.0, 1.0))

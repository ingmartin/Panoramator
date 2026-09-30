from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from .analyzer import Analysis, AnalyzedFrame
from .coverage import coverage_fraction
from .gap_fill import interpolate_surface_gaps
from .models import SurfaceBuild, SurfaceKind, SurfaceModel, UnwrapConfig
from .observed_surface import _largest_component, _publish_mask

_VERTICAL_MIN_SCALE = 0.75
_VERTICAL_MAX_SCALE = 1.25
_PRODUCT_STRIP_RATIO = 0.14
_PRODUCT_STRIP_RATIOS = (0.14, 0.20, 0.26)
_WHITE_DIAGNOSTIC_THRESHOLD = 180
_WHITE_PROTECTION_THRESHOLD = 160


def _numeric_float(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _numeric_int(value: object, default: int = 0) -> int:
    return int(value) if isinstance(value, (int, float)) else default


@dataclass(slots=True)
class ProductSurfaceBuild:
    image: np.ndarray
    coverage: np.ndarray
    model: SurfaceModel
    measurements: dict[str, float | int | str | list[float] | list[int]]
    artifacts: dict[str, object]


@dataclass(slots=True)
class _Strip:
    image: np.ndarray
    mask: np.ndarray
    frame: AnalyzedFrame
    width: int
    source_top: float = 0.0
    source_bottom: float = 0.0
    source_height: float = 0.0
    vertical_scale: float = 1.0
    frame_top_residual_px: float = 0.0
    frame_bottom_residual_px: float = 0.0
    vertical_support_fraction: float = 1.0
    source_white_pixels: int = 0
    source_white_observed_pixels: int = 0
    strip_source_white_pixels: int = 0
    strip_white_pixels: int = 0
    white_mask_rejected: np.ndarray | None = None
    strip_ratio: float = _PRODUCT_STRIP_RATIO
    registration_affine: tuple[float, ...] | None = None


@dataclass(slots=True)
class _VerticalReference:
    top: float
    height: float
    lower_height: float
    upper_height: float
    center: float = 0.0


@dataclass(slots=True)
class _Registration:
    accepted: bool
    dx_source_px: float
    step_atlas_px: float
    inlier_fraction: float
    inlier_count: int
    residual_px: float
    reason: str = ""
    transform: tuple[float, ...] | None = None


def _frame_payload(frames: list[_Strip]) -> list[dict[str, float | int]]:
    return [
        {"frame_index": item.frame.frame.index, "timestamp_seconds": item.frame.frame.timestamp_seconds}
        for item in frames
    ]


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    component = _largest_component(mask)
    if component is None:
        return None
    points = cv2.findNonZero(component)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return int(x), int(y), int(width), int(height)


def _white_pixels(image: np.ndarray, threshold: int = _WHITE_DIAGNOSTIC_THRESHOLD) -> np.ndarray:
    return np.min(image, axis=2) >= threshold


def cylindrical_surface_support_mask(item: AnalyzedFrame) -> tuple[np.ndarray, np.ndarray]:
    """Recover bright observed details inside the analysed cylinder outline.

    GrabCut may classify white print as background.  We only recover original
    bright pixels inside the convex hull of the already detected component;
    no pixel is created and the expansion cannot leave the analysed bbox.
    """

    base = item.geometry_mask.copy()
    x, y, width, height = item.bbox
    roi = base[y : y + height, x : x + width]
    component = _largest_component(roi)
    if component is None:
        return base, np.zeros_like(base)
    points = cv2.findNonZero(component)
    if points is None or len(points) < 3:
        return base, np.zeros_like(base)
    hull = np.zeros_like(roi)
    cv2.fillConvexPoly(hull, cv2.convexHull(points), 255)
    image_roi = item.frame.image[y : y + height, x : x + width]
    bright = _white_pixels(image_roi)
    recovered_roi = (hull > 0) & bright & (roi == 0)
    expanded = base.copy()
    expanded[y : y + height, x : x + width][recovered_roi] = 255
    recovered = np.zeros_like(base)
    recovered[y : y + height, x : x + width][recovered_roi] = 255
    return expanded, recovered


def extract_central_strip(
    item: AnalyzedFrame,
    mask: np.ndarray,
    output_height: int,
    ratio: float = 0.26,
    reference: _VerticalReference | None = None,
) -> _Strip | None:
    """Extract only the stable middle of a publishable surface component."""

    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    x, y, width, height = bbox
    if width < 8 or height < 12:
        return None
    image_crop = item.frame.image[y : y + height, x : x + width]
    mask_crop = mask[y : y + height, x : x + width]
    if image_crop.size == 0:
        return None
    source_white = _white_pixels(image_crop)
    source_white_pixels = int(np.count_nonzero(source_white))
    source_white_observed_pixels = int(np.count_nonzero(source_white & (mask_crop > 0)))
    if reference is None:
        scale = output_height / max(height, 1)
        target_top = 0
        target_height = output_height
    else:
        scale = output_height / max(reference.height, 1.0)
        target_height = max(1, round(height * scale))
        # Every strip gets the same atlas centre.  The source bbox can move
        # vertically as the camera orbits; using its raw top as the canvas
        # origin would turn that motion into a saw-tooth seam.  The source
        # centre is retained as a residual diagnostic instead.
        target_top = round((output_height - target_height) * 0.5)
    normalized_width = max(8, round(width * scale))
    resized_image = cv2.resize(image_crop, (normalized_width, target_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask_crop, (normalized_width, target_height), interpolation=cv2.INTER_NEAREST)
    image_canvas = np.zeros((output_height, normalized_width, 3), np.uint8)
    mask_canvas = np.zeros((output_height, normalized_width), np.uint8)
    source_top = max(0, target_top)
    source_bottom = min(output_height, target_top + target_height)
    if source_bottom <= source_top:
        return None
    crop_top = source_top - target_top
    crop_bottom = crop_top + source_bottom - source_top
    image_canvas[source_top:source_bottom] = resized_image[crop_top:crop_bottom]
    mask_canvas[source_top:source_bottom] = resized_mask[crop_top:crop_bottom]
    image_crop = image_canvas
    mask_crop = mask_canvas
    strip_width = round(normalized_width * float(np.clip(ratio, 0.10, 0.30)))
    strip_width = max(8, min(strip_width, normalized_width))
    left = max(0, (normalized_width - strip_width) // 2)
    right = min(normalized_width, left + strip_width)
    raw_strip = image_crop[:, left:right].copy()
    strip_mask = np.where(mask_crop[:, left:right] > 0, 255, 0).astype(np.uint8)
    strip_source_white_pixels = int(np.count_nonzero(_white_pixels(raw_strip)))
    white_mask_rejected = _white_pixels(raw_strip) & (strip_mask == 0)
    strip = raw_strip.copy()
    valid_fraction = float(np.count_nonzero(strip_mask) / max(strip_mask.size, 1))
    if valid_fraction < 0.35:
        return None
    strip[strip_mask == 0] = 0
    return _Strip(
        strip,
        strip_mask,
        item,
        right - left,
        source_top=float(y),
        source_bottom=float(y + height),
        source_height=float(height),
        vertical_scale=float(height / max(reference.height, 1.0)) if reference is not None else 1.0,
        frame_top_residual_px=(
            float(abs((y + height * 0.5) - reference.center) * scale) if reference is not None else 0.0
        ),
        frame_bottom_residual_px=(
            float(abs((y + height * 0.5) - reference.center) * scale) if reference is not None else 0.0
        ),
        vertical_support_fraction=valid_fraction,
        source_white_pixels=source_white_pixels,
        source_white_observed_pixels=source_white_observed_pixels,
        strip_source_white_pixels=strip_source_white_pixels,
        strip_white_pixels=int(np.count_nonzero(_white_pixels(strip) & (strip_mask > 0))),
        white_mask_rejected=white_mask_rejected.astype(np.uint8) * 255,
        strip_ratio=float(ratio),
    )


def select_adaptive_strip(
    item: AnalyzedFrame,
    mask: np.ndarray,
    output_height: int,
    reference: _VerticalReference,
) -> _Strip | None:
    """Choose the narrowest central band that retains useful bright detail."""

    candidates = [
        strip
        for ratio in _PRODUCT_STRIP_RATIOS
        if (strip := extract_central_strip(item, mask, output_height, ratio=ratio, reference=reference))
        is not None
    ]
    if not candidates:
        return None

    def score(strip: _Strip) -> float:
        observed = max(int(np.count_nonzero(strip.mask)), 1)
        white_recall = float(strip.strip_white_pixels / max(strip.source_white_observed_pixels, 1))
        detail = cv2.cvtColor(strip.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gradient = cv2.magnitude(
            cv2.Sobel(detail, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(detail, cv2.CV_32F, 0, 1, ksize=3),
        )
        detail_score = float(np.mean(gradient[strip.mask > 0]) / 255.0) if observed else 0.0
        return 4.0 * white_recall + 0.25 * detail_score + 0.20 * strip.vertical_support_fraction - 0.45 * strip.strip_ratio

    return max(candidates, key=score)


def _vertical_measurements(
    items: list[AnalyzedFrame],
    masks: list[np.ndarray],
    output_height: int,
) -> tuple[_VerticalReference, list[dict[str, float | int | str]]]:
    measurements: list[dict[str, float | int | str]] = []
    for item, mask in zip(items, masks, strict=True):
        bbox = _mask_bbox(mask)
        if bbox is None:
            continue
        _, top, _, height = bbox
        measurements.append(
            {
                "frame_index": item.frame.index,
                "timestamp_seconds": item.frame.timestamp_seconds,
                "source_top": float(top),
                "source_bottom": float(top + height),
                "source_height": float(height),
            }
        )
    if not measurements:
        return _VerticalReference(0.0, 1.0, 0.75, 1.25, 0.0), []
    heights = np.asarray([float(item["source_height"]) for item in measurements], dtype=np.float32)
    median_height = float(np.median(heights))
    upper_cluster = heights[heights >= median_height]
    reference_height = float(np.median(upper_cluster)) if upper_cluster.size else median_height
    reference_height = max(reference_height, 1.0)
    top_values = np.asarray([float(item["source_top"]) for item in measurements], dtype=np.float32)
    reference_top = float(np.median(top_values))
    centre_values = np.asarray(
        [float(item["source_top"]) + float(item["source_height"]) * 0.5 for item in measurements],
        dtype=np.float32,
    )
    reference_center = float(np.median(centre_values))
    for measurement in measurements:
        measurement_height = float(measurement["source_height"])
        scale = measurement_height / reference_height
        centre_residual = abs(
            (float(measurement["source_top"]) + measurement_height * 0.5) - reference_center
        ) * (output_height / reference_height)
        measurement["vertical_scale"] = scale
        measurement["vertical_height_ratio"] = scale
        measurement["vertical_height_valid"] = int(_VERTICAL_MIN_SCALE <= scale <= _VERTICAL_MAX_SCALE)
        measurement["reference_top"] = reference_top
        measurement["reference_height"] = reference_height
        measurement["estimated_top_residual_px"] = centre_residual
        measurement["estimated_bottom_residual_px"] = centre_residual
        measurement["reference_center"] = reference_center
    return _VerticalReference(
        reference_top,
        reference_height,
        _VERTICAL_MIN_SCALE,
        _VERTICAL_MAX_SCALE,
        reference_center,
    ), measurements


def _vertical_plot(values: list[float], height: int = 160, width: int = 640) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, np.uint8)
    if not values:
        return canvas
    low, high = min(values), max(values)
    span = max(high - low, 1e-6)
    for index, value in enumerate(values):
        x = round(index * (width - 1) / max(len(values) - 1, 1))
        y = round((high - value) / span * (height - 20)) + 10
        cv2.circle(canvas, (x, y), 3, (40, 80, 220), -1)
    return canvas


def _registration_view(item: AnalyzedFrame) -> tuple[np.ndarray, int, int]:
    """Return an expanded object crop used only for frame registration."""

    x, y, width, height = item.bbox
    margin = max(8, round(max(width, height) * 0.10))
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(item.frame.image.shape[1], x + width + margin)
    bottom = min(item.frame.image.shape[0], y + height + margin)
    return item.frame.image[top:bottom, left:right], left, top


def estimate_pairwise_registration(
    left: _Strip,
    right: _Strip,
    output_height: int,
    reference_height: float,
) -> _Registration:
    """Estimate the atlas step between two chronological object views.

    The previous bbox heuristic invented a large step for every frame.  This
    registration uses the full object crop, where there are enough stable
    features even when the narrow publish strip is mostly uniform background.
    """

    left_view, left_x, _ = _registration_view(left.frame)
    right_view, right_x, _ = _registration_view(right.frame)
    left_gray = cv2.cvtColor(left_view, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_view, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=900, contrastThreshold=0.025)
        norm = cv2.NORM_L2
        ratio = 0.75
    else:
        detector = cv2.ORB_create(nfeatures=1200, fastThreshold=8)  # type: ignore[attr-defined]
        norm = cv2.NORM_HAMMING
        ratio = 0.72
    left_keypoints, left_descriptors = detector.detectAndCompute(left_gray, None)
    right_keypoints, right_descriptors = detector.detectAndCompute(right_gray, None)
    if left_descriptors is None or right_descriptors is None:
        return _Registration(False, 0.0, 0.0, 0.0, 0, float("inf"), "registration_no_descriptors")
    matcher = cv2.BFMatcher(norm)
    matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    good = [first for first, second in matches if first.distance < ratio * second.distance]
    if len(good) < 8:
        return _Registration(
            False,
            0.0,
            0.0,
            0.0,
            len(good),
            float("inf"),
            "registration_insufficient_matches",
        )
    left_points = np.asarray([left_keypoints[match.queryIdx].pt for match in good], dtype=np.float32)
    right_points = np.asarray([right_keypoints[match.trainIdx].pt for match in good], dtype=np.float32)
    transform, inlier_mask = cv2.estimateAffinePartial2D(
        right_points,
        left_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    if transform is None or inlier_mask is None:
        return _Registration(False, 0.0, 0.0, 0.0, 0, float("inf"), "registration_ransac_failed")
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_fraction = float(inlier_count / max(len(good), 1))
    projected = np.asarray(cv2.transform(right_points[None, :, :], transform))[0]
    residual = float(np.median(np.linalg.norm(projected[inliers] - left_points[inliers], axis=1)))
    scale_x = float(np.linalg.norm(transform[0, :2]))
    scale_y = float(np.linalg.norm(transform[1, :2]))
    if inlier_count < 12 or inlier_fraction < 0.20:
        reason = "registration_low_inlier_fraction"
    elif residual > 3.5:
        reason = "registration_high_residual"
    elif not (0.90 <= scale_x <= 1.10 and 0.90 <= scale_y <= 1.10):
        reason = "registration_scale_outlier"
    else:
        reason = ""
    dx_source = float(transform[0, 2] + left_x - right_x)
    step_atlas = abs(dx_source) * output_height / max(reference_height, 1.0)
    if not reason and step_atlas < max(1.5, min(left.width, right.width) * 0.12):
        reason = "registration_duplicate_view"
    if not reason and step_atlas > min(left.width, right.width) * 1.25:
        reason = "registration_gap_too_large"
    return _Registration(
        not reason,
        dx_source,
        step_atlas,
        inlier_fraction,
        inlier_count,
        residual,
        reason,
        tuple(float(value) for value in transform.ravel()),
    )


def _warp_strip_with_registration(
    strip: _Strip,
    transform: tuple[float, ...],
    reference_height: float,
    output_height: int,
) -> _Strip:
    """Apply the accepted local affine shape correction to a strip.

    Registration is estimated in source-frame coordinates.  The strip has
    already been normalized to the common atlas height, so convert only the
    linear part into strip coordinates and keep the scalar horizontal atlas
    offset responsible for the orbit position.  This prevents a frame-local
    scale/rotation from moving a full-height band relative to its neighbours.
    """

    matrix = np.asarray(transform, dtype=np.float32).reshape(2, 3)
    bbox_width = max(float(strip.frame.bbox[2]), 1.0)
    source_per_strip_x = bbox_width * float(strip.strip_ratio) / max(strip.width, 1)
    source_per_strip_y = max(float(reference_height), 1.0) / max(output_height, 1)
    source_to_strip = np.diag(
        [1.0 / max(source_per_strip_x, 1e-6), 1.0 / max(source_per_strip_y, 1e-6)]
    ).astype(np.float32)
    strip_to_source = np.diag([source_per_strip_x, source_per_strip_y]).astype(np.float32)
    linear = source_to_strip @ matrix[:, :2] @ strip_to_source
    singular_values = np.linalg.svd(linear, compute_uv=False)
    if not np.all(np.isfinite(singular_values)):
        return strip
    # Feature registration may contain a small amount of perspective-like
    # scale.  Keep it, but reject a pathological local warp rather than
    # producing another stretched strip.
    if float(np.min(singular_values)) < 0.70 or float(np.max(singular_values)) > 1.35:
        return strip
    centre = np.asarray([strip.width * 0.5, output_height * 0.5], dtype=np.float32)
    # Translation is represented by the accumulated atlas offset.  Applying
    # it a second time here would move every strip twice; the affine residual
    # is therefore the local scale/rotation around the strip centre.
    affine = np.column_stack((linear, centre - linear @ centre))
    warped_image = cv2.warpAffine(
        strip.image,
        affine,
        (strip.width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        strip.mask,
        affine,
        (strip.width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_rejected = (
        cv2.warpAffine(
            strip.white_mask_rejected,
            affine,
            (strip.width, output_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if strip.white_mask_rejected is not None
        else None
    )
    return _Strip(
        warped_image,
        warped_mask,
        strip.frame,
        strip.width,
        strip.source_top,
        strip.source_bottom,
        strip.source_height,
        strip.vertical_scale,
        strip.frame_top_residual_px,
        strip.frame_bottom_residual_px,
        float(np.count_nonzero(warped_mask) / max(warped_mask.size, 1)),
        strip.source_white_pixels,
        strip.source_white_observed_pixels,
        strip.strip_source_white_pixels,
        int(np.count_nonzero(_white_pixels(warped_image) & (warped_mask > 0))),
        warped_rejected,
        strip.strip_ratio,
        tuple(float(value) for value in affine.ravel()),
    )


def register_strips(
    strips: list[_Strip],
    output_height: int,
    reference_height: float,
) -> tuple[list[_Strip], list[float], list[dict[str, float | int | str | list[float]]]]:
    """Keep the dense chronological registration chain and its global phase."""

    if not strips:
        return [], [], []
    selected = [strips[0]]
    offsets = [0.0]
    last_observed = strips[0]
    phase = 0.0
    records: list[dict[str, float | int | str | list[float]]] = [
        {
            "left_frame_index": -1,
            "right_frame_index": strips[0].frame.frame.index,
            "accepted": 1,
            "step_atlas_px": 0.0,
            "phase_atlas_px": 0.0,
            "phase_step_atlas_px": 0.0,
            "keyframe_selected": 1,
            "reason": "registration_anchor",
        }
    ]
    for candidate in strips[1:]:
        left_frame_index = last_observed.frame.frame.index
        estimate = estimate_pairwise_registration(last_observed, candidate, output_height, reference_height)
        raw_signed_step = estimate.dx_source_px * output_height / max(reference_height, 1.0)
        phase_step = estimate.step_atlas_px if estimate.accepted else 0.0
        phase_reason = estimate.reason
        phase_outlier = int(estimate.accepted and raw_signed_step <= 0.0)
        phase += phase_step
        keyframe_selected = estimate.accepted
        if keyframe_selected:
            published = _warp_strip_with_registration(
                candidate,
                estimate.transform or (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                reference_height,
                output_height,
            )
            selected.append(published)
            offsets.append(phase)
        last_observed = candidate
        records.append(
            {
                "left_frame_index": left_frame_index,
                "right_frame_index": candidate.frame.frame.index,
                "accepted": int(keyframe_selected),
                "dx_source_px": estimate.dx_source_px,
                "signed_step_atlas_px": raw_signed_step,
                "step_atlas_px": estimate.step_atlas_px,
                "phase_atlas_px": phase,
                "phase_step_atlas_px": phase_step,
                "phase_step_outlier": phase_outlier,
                "keyframe_selected": int(keyframe_selected),
                "inlier_fraction": estimate.inlier_fraction,
                "inlier_count": estimate.inlier_count,
                "residual_px": estimate.residual_px,
                "reason": phase_reason,
                "transform": list(estimate.transform) if estimate.transform is not None else [],
            }
        )
    return selected, offsets, records


def repack_registered_offsets(strips: list[_Strip], offsets: list[float]) -> tuple[list[float], int]:
    """Remove unpublishable holes left by rejected bridge frames.

    This only changes placement of already observed strips; it never paints a
    pixel into the gap.  A large registration jump is compressed into the
    available overlap so the compositor can compare the two real observations
    and keep the seam diagnostics visible.
    """

    if not strips:
        return [], 0
    packed = [0.0]
    clamped = 0
    for left, right, left_offset, right_offset in zip(
        strips[:-1], strips[1:], offsets[:-1], offsets[1:], strict=True
    ):
        raw_step = max(0.0, right_offset - left_offset)
        maximum = max(1.0, 0.88 * min(left.width, right.width))
        step = min(raw_step, maximum)
        if step < raw_step - 1e-6:
            clamped += 1
        packed.append(packed[-1] + step)
    return packed, clamped


def _registration_gap_map(
    strips: list[_Strip],
    offsets: list[float],
    canvas_height: int,
    canvas_width: int,
) -> np.ndarray:
    """Mark horizontal intervals with no observed strip before composition."""

    occupied = np.zeros(canvas_width, np.uint8)
    for strip, offset in zip(strips, offsets, strict=True):
        left = max(0, round(offset))
        right = min(canvas_width, left + strip.width)
        if right > left:
            occupied[left:right] = 255
    gaps = np.zeros(canvas_width, np.uint8)
    observed_columns = np.flatnonzero(occupied)
    if observed_columns.size:
        gaps[observed_columns[0] : observed_columns[-1] + 1] = np.where(occupied[observed_columns[0] : observed_columns[-1] + 1] == 0, 255, 0)
    return np.broadcast_to(gaps[None, :], (canvas_height, canvas_width)).copy()


def _phase_cell_bounds(strips: list[_Strip], offsets: list[float], canvas_width: int) -> list[tuple[int, int]]:
    """Return disjoint atlas cells centered on the monotonic keyframe phases."""

    if not strips:
        return []
    centres = [float(offset + strip.width * 0.5) for strip, offset in zip(strips, offsets, strict=True)]
    left_edges = [0] + [round(np.ceil((left + right) * 0.5)) for left, right in itertools.pairwise(centres)]
    right_edges = [round(np.floor((left + right) * 0.5)) for left, right in itertools.pairwise(centres)] + [canvas_width]
    return [
        (max(0, min(left, canvas_width)), max(0, min(right, canvas_width)))
        for left, right in zip(left_edges, right_edges, strict=True)
    ]


def _white_atlas_artifacts(
    strips: list[_Strip],
    offsets: list[float],
    image: np.ndarray,
    coverage: np.ndarray,
    owner: np.ndarray,
    canvas_width: int,
    phase_cells: list[tuple[int, int]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Trace bright pixels from selected strips into the composed atlas."""

    height = image.shape[0]
    source_map = np.zeros((height, canvas_width), np.uint8)
    rejected_map = np.zeros_like(source_map)
    output_map = np.where(_white_pixels(image), 255, 0).astype(np.uint8)
    loss_map = np.zeros_like(source_map)
    source_white_pixels = sum(strip.source_white_pixels for strip in strips)
    source_white_observed_pixels = sum(strip.source_white_observed_pixels for strip in strips)
    strip_source_white_pixels = 0
    strip_white_pixels = 0
    for frame_id, (strip, offset) in enumerate(zip(strips, offsets, strict=True), start=1):
        left = max(0, round(offset))
        if left >= canvas_width:
            continue
        width = min(strip.width, canvas_width - left)
        if width <= 0:
            continue
        region = slice(left, left + width)
        strip_white = _white_pixels(strip.image[:, :width]) & (strip.mask[:, :width] > 0)
        if phase_cells is not None:
            cell_left, cell_right = phase_cells[frame_id - 1]
            local_left = max(0, cell_left - left)
            local_right = min(width, cell_right - left)
            strip_white[:, :local_left] = False
            strip_white[:, local_right:] = False
        rejected_white = (
            strip.white_mask_rejected[:, :width] > 0
            if strip.white_mask_rejected is not None
            else np.zeros_like(strip_white)
        )
        source_map[:, region] = np.maximum(source_map[:, region], np.where(strip_white, 255, 0).astype(np.uint8))
        rejected_map[:, region] = np.maximum(
            rejected_map[:, region], np.where(rejected_white, 255, 0).astype(np.uint8)
        )
        strip_white_pixels += int(np.count_nonzero(strip_white))
        strip_source_white_pixels += strip.strip_source_white_pixels
        expected = strip_white
        output_white = _white_pixels(image[:, region]) & (coverage[:, region] > 0)
        # Expected white is a union of observations, not an owner count: a
        # later seam may legitimately replace the source owner while keeping
        # the same bright detail.
        loss_map[:, region][expected & ~output_white] = 255
    expected_map = source_map > 0
    expected_count = int(np.count_nonzero(expected_map))
    output_count = int(np.count_nonzero(expected_map & (_white_pixels(image) & (coverage > 0))))
    metrics: dict[str, float | int] = {
        "product_surface_white_source_pixels": source_white_pixels,
        "product_surface_white_geometry_pixels": source_white_observed_pixels,
        "product_surface_white_strip_pixels": strip_white_pixels,
        "product_surface_white_strip_source_pixels": strip_source_white_pixels,
        "product_surface_white_mask_rejected_pixels": int(np.count_nonzero(rejected_map)),
        "product_surface_white_owned_expected_pixels": expected_count,
        "product_surface_white_owned_output_pixels": output_count,
        "product_surface_white_geometry_recall": float(
            source_white_observed_pixels / max(source_white_pixels, 1)
        ),
        "product_surface_white_strip_recall": float(
            strip_white_pixels / max(strip_source_white_pixels, 1)
        ),
        "product_surface_white_output_recall": float(
            output_count / max(expected_count, 1)
        ),
        "product_surface_white_highlight_compression": float(
            1.0 - output_count / max(expected_count, 1)
        ),
    }
    return {
        "product_surface_white_source": source_map,
        "product_surface_white_mask_rejected": rejected_map,
        "product_surface_white_strip": source_map.copy(),
        "product_surface_white_output": output_map,
        "product_surface_white_loss": loss_map,
    }, metrics


def _strip_step(left: _Strip, right: _Strip, output_height: int) -> float:
    """Estimate a forward step from strip scale and observed bbox motion."""

    left_bbox = left.frame.bbox
    right_bbox = right.frame.bbox
    centre_delta = abs((right_bbox[0] + right_bbox[2] * 0.5) - (left_bbox[0] + left_bbox[2] * 0.5))
    mean_height = max((left_bbox[3] + right_bbox[3]) * 0.5, 1.0)
    projected_motion = centre_delta * output_height / mean_height
    minimum = 0.58 * min(left.width, right.width)
    return max(minimum, 0.34 * min(left.width, right.width) + 0.70 * projected_motion)


def monotonic_offsets(strips: list[_Strip], output_height: int) -> list[float]:
    if not strips:
        return []
    offsets = [0.0]
    steps = [_strip_step(left, right, output_height) for left, right in itertools.pairwise(strips)]
    if steps:
        # Median smoothing suppresses one bad segmentation jump while preserving
        # the direction of the one-pass orbit.
        smoothed = steps.copy()
        for index in range(1, len(steps) - 1):
            smoothed[index] = float(np.median(steps[index - 1 : index + 2]))
        offsets.extend(np.cumsum(np.maximum(smoothed, 1.0)).tolist())
    return offsets


def photometric_normalize(
    current: np.ndarray,
    candidate: np.ndarray,
    overlap: np.ndarray,
    *,
    min_pixels: int = 32,
) -> tuple[np.ndarray, float, float]:
    """Normalize a candidate using robust overlap statistics with bounded gain."""

    if int(np.count_nonzero(overlap)) < min_pixels:
        return candidate, 1.0, 0.0
    current_values = current[overlap].astype(np.float32)
    candidate_values = candidate[overlap].astype(np.float32)
    current_median = float(np.median(current_values))
    candidate_median = float(np.median(candidate_values))
    if candidate_median <= 1.0:
        return candidate, 1.0, 0.0
    gain = float(np.clip(current_median / candidate_median, 0.85, 1.18))
    bias = float(np.clip(current_median - gain * candidate_median, -24.0, 24.0))
    normalized = np.clip(candidate.astype(np.float32) * gain + bias, 0.0, 255.0).astype(np.uint8)
    # A global overlap correction can turn a white printed area into gray
    # (for example gain=0.85 and bias=-24).  Keep highlights close to their
    # observed value while applying the robust correction to the rest.
    protected = np.min(candidate, axis=2) >= _WHITE_PROTECTION_THRESHOLD
    if np.any(protected):
        # Preserve the observed highlight itself.  Applying even a bounded
        # negative bias here would still turn a source value of 183 into a
        # visibly gray 165.
        normalized[protected] = candidate[protected]
    return normalized, gain, bias


def find_seam(current: np.ndarray, candidate: np.ndarray, overlap: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Find a low-cost top-to-bottom path through an overlap using dynamic programming."""

    if current.shape[:2] != candidate.shape[:2] or not np.any(overlap):
        return None
    height, _width = overlap.shape
    colour_cost = np.mean(np.abs(current.astype(np.float32) - candidate.astype(np.float32)), axis=2) / 255.0
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient_cost = np.abs(
        cv2.Sobel(current_gray, cv2.CV_32F, 1, 0, ksize=3)
        - cv2.Sobel(candidate_gray, cv2.CV_32F, 1, 0, ksize=3)
    ) / 255.0
    cost = 0.65 * colour_cost + 0.35 * np.minimum(gradient_cost, 1.0)
    cost[~overlap] = 1.0
    supported = np.count_nonzero(overlap, axis=0) >= max(2, int(height * 0.20))
    columns = np.flatnonzero(supported)
    if not columns.size:
        return None
    left, right = int(columns[0]), int(columns[-1])
    local_cost = cost[:, left : right + 1]
    local_width = local_cost.shape[1]
    dynamic = np.full_like(local_cost, np.inf, dtype=np.float32)
    back = np.zeros((height, local_width), dtype=np.int16)
    dynamic[0] = local_cost[0]
    for row in range(1, height):
        for column in range(local_width):
            start = max(0, column - 1)
            stop = min(local_width, column + 2)
            previous = dynamic[row - 1, start:stop]
            choice = int(np.argmin(previous))
            previous_column = start + choice
            dynamic[row, column] = local_cost[row, column] + previous[choice] + 0.06 * abs(column - previous_column)
            back[row, column] = previous_column
    path = np.zeros(height, dtype=np.int32)
    path[-1] = int(np.argmin(dynamic[-1]))
    for row in range(height - 1, 0, -1):
        path[row - 1] = back[row, path[row]]
    path += left
    score = float(np.mean(cost[np.arange(height), path]))
    return path, score


def _largest_component_fraction(mask: np.ndarray) -> float:
    largest = _largest_component(mask)
    total = int(np.count_nonzero(mask))
    if largest is None or total == 0:
        return 0.0
    return float(np.count_nonzero(largest) / total)


def _occupied_column_fraction(mask: np.ndarray) -> float:
    """Measure continuity of the observed horizontal atlas span.

    The pixel connected-component ratio is intentionally not used here:
    curved object silhouettes and explicit transparent top/bottom regions can
    split a valid observed atlas into many components.  This metric detects
    actual horizontal holes between the first and last observed columns.
    """

    occupied = np.any(mask > 0, axis=0)
    columns = np.flatnonzero(occupied)
    if columns.size == 0:
        return 0.0
    span = int(columns[-1] - columns[0] + 1)
    return float(np.count_nonzero(occupied[columns[0] : columns[-1] + 1]) / max(span, 1))


def _gradient_energy(image: np.ndarray, coverage: np.ndarray) -> float:
    valid = coverage > 0
    if not np.any(valid):
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    return float(np.mean(gradient[valid]))


def _hard_transition_score(image: np.ndarray, coverage: np.ndarray, owner: np.ndarray) -> float:
    boundary = (owner[:, 1:] != owner[:, :-1]) & (coverage[:, 1:] > 0) & (coverage[:, :-1] > 0)
    if not np.any(boundary):
        return 0.0
    difference = np.mean(np.abs(image[:, 1:].astype(np.float32) - image[:, :-1].astype(np.float32)), axis=2) / 255.0
    # The mean hides a short but very visible hard seam.  Use a robust upper
    # percentile so one bad vertical transition can reject the candidate.
    return float(np.percentile(difference[boundary], 95))


def _owner_reentry_count(owner: np.ndarray) -> int:
    """Count owners that reappear after a later chronological owner."""

    reentries = 0
    for row in owner:
        seen: set[int] = set()
        previous = 0
        for value in row:
            current = int(value)
            if current == 0:
                continue
            if current in seen and current != previous:
                reentries += 1
            seen.add(current)
            previous = current
    return reentries


def _compose(
    strips: list[_Strip],
    offsets: list[float],
    canvas_height: int,
    canvas_width: int,
    baseline_image: np.ndarray,
    phase_cells: list[tuple[int, int]] | None = None,
    interpolate_gaps: bool = False,
    max_interpolation_gap_px: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    image = np.zeros((canvas_height, canvas_width, 3), np.uint8)
    coverage = np.zeros((canvas_height, canvas_width), np.uint8)
    owner = np.zeros((canvas_height, canvas_width), np.uint16)
    seam_visual = np.zeros_like(image)
    seam_cost_visual = np.zeros((canvas_height, canvas_width), np.uint8)
    conflict_values: list[float] = []
    seam_scores: list[float] = []
    gains: list[float] = []
    positions = [max(0, round(value)) for value in offsets]
    for frame_id, (strip, position) in enumerate(zip(strips, positions, strict=True), start=1):
        if position >= canvas_width:
            continue
        width = min(strip.image.shape[1], canvas_width - position)
        candidate = strip.image[:, :width].copy()
        candidate_mask = strip.mask[:, :width].copy()
        primary_valid = candidate_mask > 0
        if phase_cells is not None:
            cell_left, cell_right = phase_cells[frame_id - 1]
            local_left = max(0, cell_left - position)
            local_right = min(width, cell_right - position)
            primary_valid[:, :local_left] = False
            primary_valid[:, local_right:] = False
        region = slice(position, position + width)
        current = image[:, region]
        current_coverage = coverage[:, region] > 0
        candidate_valid = primary_valid
        overlap = current_coverage & candidate_valid
        existing_owner = owner[:, region]
        neighbour_overlap = (
            np.ones_like(overlap, dtype=bool)
            if phase_cells is None
            else (~overlap) | (existing_owner == max(frame_id - 1, 0))
        )
        candidate, gain, bias = photometric_normalize(current, candidate, overlap)
        gains.append(abs(gain - 1.0) + abs(bias) / 255.0)
        if np.any(overlap):
            differences = np.mean(np.abs(current[overlap].astype(np.float32) - candidate[overlap].astype(np.float32)), axis=1) / 255.0
            conflict_values.append(float(np.mean(differences)))
        seam_result = find_seam(current, candidate, overlap)
        path: np.ndarray | None = None
        if seam_result is not None:
            path, seam_score = seam_result
            seam_scores.append(seam_score)
            valid_rows = np.flatnonzero(np.any(overlap, axis=1))
            for row in valid_rows:
                column = int(path[row])
                global_column = position + column
                if 0 <= global_column < canvas_width:
                    seam_visual[row, global_column] = (0, 0, 255)
                    seam_cost_visual[row, global_column] = np.uint8(np.clip(seam_score * 255.0, 0, 255))
        x_grid = np.arange(width)[None, :]
        current_white = _white_pixels(current)
        candidate_white = _white_pixels(candidate)
        preserve_current_highlight = current_coverage & current_white & ~candidate_white
        promote_candidate_highlight = candidate_valid & candidate_white & ~current_white
        if path is None:
            candidate_owner = candidate_valid & neighbour_overlap
        else:
            candidate_owner = candidate_valid & ((~overlap) | (x_grid >= path[:, None])) & neighbour_overlap
        candidate_owner &= ~preserve_current_highlight
        candidate_owner |= promote_candidate_highlight & neighbour_overlap
        empty = candidate_valid & ~current_coverage
        replace = candidate_owner | empty
        if np.any(replace):
            current[replace] = candidate[replace]
            owner[:, region][replace] = frame_id
        # Keep a narrow blend around the selected seam only when the local
        # colour conflict is not severe enough to hide a geometry error.
        if path is not None:
            blend_width = min(14, max(6, width // 10))
            for row_index in range(canvas_height):
                seam_column = int(path[row_index])
                start = max(0, seam_column - blend_width)
                stop = min(width, seam_column + blend_width + 1)
                for column in range(start, stop):
                    if not overlap[row_index, column]:
                        continue
                    local_difference = float(
                        np.mean(np.abs(current[row_index, column].astype(np.float32) - candidate[row_index, column])) / 255.0
                    )
                    if local_difference > 0.42:
                        continue
                    alpha = 0.5 + 0.5 * np.clip((column - seam_column) / max(blend_width, 1), -1.0, 1.0)
                    current[row_index, column] = np.clip(
                        current[row_index, column].astype(np.float32) * (1.0 - alpha)
                        + candidate[row_index, column].astype(np.float32) * alpha,
                        0,
                        255,
                    ).astype(np.uint8)
        coverage[:, region][candidate_valid] = 255
    if phase_cells is not None:
        # A phase cell owns conflicts, but a neighbouring observation may
        # still provide a real pixel where the primary mask has a hole.  Fill
        # only uncovered pixels; never replace an existing owner here.
        for frame_id, (strip, offset) in enumerate(zip(strips, positions, strict=True), start=1):
            if offset >= canvas_width:
                continue
            width = min(strip.image.shape[1], canvas_width - offset)
            if width <= 0:
                continue
            region = slice(offset, offset + width)
            fill = (strip.mask[:, :width] > 0) & (coverage[:, region] == 0)
            if np.any(fill):
                image[:, region][fill] = strip.image[:, :width][fill]
                owner[:, region][fill] = frame_id
                coverage[:, region][fill] = 255
    if interpolate_gaps:
        interpolate_surface_gaps(
            image,
            coverage,
            owner,
            max_gap=max_interpolation_gap_px,
        )
    baseline_energy = _gradient_energy(baseline_image, np.where(np.any(baseline_image > 0, axis=2), 255, 0).astype(np.uint8))
    product_energy = _gradient_energy(image, coverage)
    gradient_ratio = float(product_energy / baseline_energy) if baseline_energy > 1e-6 else 0.0
    mean_conflict = float(np.mean(conflict_values)) if conflict_values else 0.0
    mean_seam = float(np.mean(seam_scores)) if seam_scores else 0.0
    confidence = np.where(coverage > 0, 255, 0).astype(np.uint8)
    return image, coverage, owner, seam_visual, confidence, mean_conflict, mean_seam, gradient_ratio


class ProductSurfaceBuilder:
    """Build a publishable surface candidate from narrow, ordered frame strips."""

    def build(self, analysis: Analysis, config: UnwrapConfig, baseline_build: SurfaceBuild) -> ProductSurfaceBuild:
        measurements = dict(baseline_build.measurements)
        for key, value in baseline_build.measurements.items():
            measurements.setdefault(f"baseline_{key}", value)
        artifacts = dict(baseline_build.artifacts)
        rejected: list[dict[str, float | int | str]] = []
        strips: list[_Strip] = []
        registration_pool: list[_Strip] = []
        support_recovery = np.zeros_like(analysis.frames[0].geometry_mask) if analysis.frames else np.zeros((1, 1), np.uint8)
        white_bbox_pixels = 0
        white_geometry_pixels = 0
        white_support_pixels = 0
        prepared_masks: list[tuple[AnalyzedFrame, np.ndarray, bool]] = []
        for item in analysis.frames:
            if analysis.kind is SurfaceKind.CYLINDRICAL:
                # The conservative publish mask follows the foreground
                # decorations and leaves holes where the cylindrical wall is
                # still observed.  Cylindrical composition needs the already
                # analysed geometry mask to retain that wall; no pixels are
                # invented because this is still a per-frame observed mask.
                mask, recovered = cylindrical_surface_support_mask(item)
                support_recovery = np.maximum(support_recovery, recovered)
                used_fallback = False
            else:
                mask, used_fallback = _publish_mask(item)
            x, y, width, height = item.bbox
            image_roi = item.frame.image[y : y + height, x : x + width]
            white_roi = _white_pixels(image_roi)
            white_bbox_pixels += int(np.count_nonzero(white_roi))
            white_geometry_pixels += int(np.count_nonzero(white_roi & (item.geometry_mask[y : y + height, x : x + width] > 0)))
            white_support_pixels += int(np.count_nonzero(white_roi & (mask[y : y + height, x : x + width] > 0)))
            prepared_masks.append((item, mask, used_fallback))
        reference, vertical_frames = _vertical_measurements(
            [item for item, _, _ in prepared_masks],
            [mask for _, mask, _ in prepared_masks],
            config.output_height,
        )
        artifacts["product_surface_vertical_frames"] = vertical_frames
        vertical_by_frame = {int(item["frame_index"]): item for item in vertical_frames}
        for item, mask, used_fallback in prepared_masks:
            vertical = vertical_by_frame.get(item.frame.index)
            if vertical is None:
                rejected.append(
                    {
                        "frame_index": item.frame.index,
                        "timestamp_seconds": item.frame.timestamp_seconds,
                        "reason": "product_surface_vertical_reference_unavailable",
                    }
                )
                continue
            if analysis.kind is SurfaceKind.CYLINDRICAL:
                # A variable strip width is unsafe for the cylindrical
                # atlas: one tall/high-contrast frame can become a 0.26-wide
                # full-height band and overwrite several normal 0.14 bands.
                strip = extract_central_strip(
                    item,
                    mask,
                    config.output_height,
                    ratio=_PRODUCT_STRIP_RATIO,
                    reference=reference,
                )
            else:
                strip = select_adaptive_strip(
                    item,
                    mask,
                    config.output_height,
                    reference,
                )
            if strip is None:
                vertical["selected"] = 0
                vertical["rejection_reason"] = "product_surface_strip_unusable"
                rejected.append(
                    {
                        "frame_index": item.frame.index,
                        "timestamp_seconds": item.frame.timestamp_seconds,
                        "reason": "product_surface_strip_unusable",
                    }
                )
                continue
            # Keep outlier-height frames in the registration chain as bridges,
            # but never publish their pixels into the composed surface.
            registration_pool.append(strip)
            if vertical["vertical_height_valid"] != 1:
                vertical["selected"] = 0
                vertical["rejection_reason"] = "product_surface_vertical_support_outlier"
                rejected.append(
                    {
                        "frame_index": item.frame.index,
                        "timestamp_seconds": item.frame.timestamp_seconds,
                        "reason": "product_surface_vertical_support_outlier",
                        "vertical_scale": float(vertical["vertical_scale"]),
                        "source_height": float(vertical["source_height"]),
                        "reference_height": float(vertical["reference_height"]),
                    }
                )
                continue
            strips.append(strip)
            vertical["selected_strip_ratio"] = strip.strip_ratio
            vertical["selected"] = 1
            vertical["strip_vertical_support_fraction"] = strip.vertical_support_fraction
            vertical["frame_center_residual_px"] = strip.frame_top_residual_px
            if used_fallback:
                rejected.append(
                    {
                        "frame_index": item.frame.index,
                        "timestamp_seconds": item.frame.timestamp_seconds,
                        "reason": "product_surface_nuisance_mask_fallback",
                    }
                )
        if len(strips) < 4:
            return self._fallback(
                baseline_build,
                measurements,
                artifacts,
                strips,
                rejected,
                "insufficient_product_strips",
            )
        strips, offsets, registration_records = register_strips(
            registration_pool,
            config.output_height,
            reference.height,
        )
        registered_pairs = {
            strip.frame.frame.index: (strip, offset)
            for strip, offset in zip(strips, offsets, strict=True)
        }
        # A height outlier is never stretched to the reference height, but a
        # registered partial strip may still contribute the pixels it really
        # observed.  This avoids turning every rejected vertical support into
        # a large horizontal hole while keeping the outlier visible in the
        # diagnostics and out of the full-height support gate.
        composition_ids = set(registered_pairs)
        for frame_index in composition_ids:
            vertical = vertical_by_frame.get(frame_index)
            if vertical is not None and vertical.get("vertical_height_valid") != 1:
                vertical["composed_partial"] = 1
        strips = [strip for strip in strips if strip.frame.frame.index in composition_ids]
        offsets = [registered_pairs[strip.frame.frame.index][1] for strip in strips]
        # Keep the measured intervals.  Repacking used to compress a failed
        # registration gap into an artificial overlap, which made missing
        # observations look like duplicated texture.  The compositor now
        # leaves such intervals transparent and exposes them as diagnostics.
        clamped_registration_gaps = 0
        vertical_by_frame = {int(item["frame_index"]): item for item in vertical_frames}
        for record in registration_records:
            if record.get("accepted") == 0:
                frame_index = _numeric_int(record.get("right_frame_index"))
                vertical = vertical_by_frame.get(frame_index)
                if vertical is not None:
                    vertical["selected"] = 0
                    vertical["rejection_reason"] = str(record.get("reason", "registration_rejected"))
                rejected.append(
                    {
                        "frame_index": frame_index,
                        "reason": str(record.get("reason", "registration_rejected")),
                        "registration_inlier_fraction": _numeric_float(record.get("inlier_fraction")),
                        "registration_residual_px": _numeric_float(record.get("residual_px"), float("inf")),
                    }
                )
        artifacts["product_surface_registration"] = registration_records
        artifacts["product_surface_registration_steps"] = _vertical_plot(
            [_numeric_float(record.get("step_atlas_px")) for record in registration_records]
        )
        artifacts["product_surface_support_recovery"] = support_recovery
        if len(strips) < 4:
            return self._fallback(
                baseline_build,
                measurements,
                artifacts,
                strips,
                rejected,
                "insufficient_registered_product_strips",
            )
        required_width = int(max(offset + strip.width for offset, strip in zip(offsets, strips, strict=True))) + 1
        canvas_width = max(8, min(config.output_width, required_width))
        if required_width > canvas_width:
            scale = (canvas_width - 1) / max(required_width - 1, 1)
            offsets = [offset * scale for offset in offsets]
            strips = [_resize_strip(strip, scale, config.output_height) for strip in strips]
        registration_gaps = _registration_gap_map(
            strips,
            offsets,
            config.output_height,
            canvas_width,
        )
        # Phase-cell composition was intentionally removed: it cut valid
        # observations into artificial vertical windows and increased loss.
        # Keep the phase artifact available for diagnostics, but compose the
        # dense registered strips directly.
        phase_cells = None
        image, coverage, owner, seams, confidence, ghosting, seam_energy, gradient_ratio = _compose(
            strips,
            offsets,
            config.output_height,
            canvas_width,
            baseline_build.image,
            phase_cells,
            config.interpolate_gaps,
            config.max_interpolation_gap_px,
        )
        white_artifacts, white_metrics = _white_atlas_artifacts(
            strips,
            offsets,
            image,
            coverage,
            owner,
            canvas_width,
            phase_cells,
        )
        white_metrics.update(
            {
                "product_surface_white_bbox_pixels": white_bbox_pixels,
                "product_surface_white_original_geometry_pixels": white_geometry_pixels,
                "product_surface_white_support_pixels": white_support_pixels,
                "product_surface_white_original_geometry_recall": float(
                    white_geometry_pixels / max(white_bbox_pixels, 1)
                ),
                "product_surface_white_support_recall": float(
                    white_support_pixels / max(white_bbox_pixels, 1)
                ),
            }
        )
        largest_fraction = _largest_component_fraction(coverage)
        occupied_column_fraction = _occupied_column_fraction(coverage)
        hard_transition = _hard_transition_score(image, coverage, owner)
        min_width = min(strip.width for strip in strips)
        vertical_scales = [strip.vertical_scale for strip in strips]
        full_height_strips = [
            strip
            for strip in strips
            if _VERTICAL_MIN_SCALE <= strip.vertical_scale <= _VERTICAL_MAX_SCALE
        ]
        vertical_quality_strips = full_height_strips or strips
        vertical_quality_scales = [strip.vertical_scale for strip in vertical_quality_strips]
        vertical_scale_p05 = float(np.percentile(vertical_quality_scales, 5))
        vertical_scale_p95 = float(np.percentile(vertical_quality_scales, 95))
        vertical_scale_ratio = vertical_scale_p95 / max(vertical_scale_p05, 1e-6)
        vertical_alignment_error = max(
            float(np.median([strip.frame_top_residual_px for strip in vertical_quality_strips])),
            float(np.median([strip.frame_bottom_residual_px for strip in vertical_quality_strips])),
        ) / max(config.output_height, 1)
        vertical_support_fraction = float(
            np.median([strip.vertical_support_fraction for strip in vertical_quality_strips])
        )
        registration_pairs = registration_records[1:]
        accepted_registration_pairs = [record for record in registration_pairs if record.get("accepted") == 1]
        registration_inlier_fraction = float(
            np.median([_numeric_float(record.get("inlier_fraction")) for record in accepted_registration_pairs])
        ) if accepted_registration_pairs else 0.0
        registration_residual = float(
            np.median([_numeric_float(record.get("residual_px"), float("inf")) for record in accepted_registration_pairs])
        ) if accepted_registration_pairs else float("inf")
        registration_acceptance_fraction = float(
            len(accepted_registration_pairs) / max(len(registration_pairs), 1)
        )
        phase_steps = [
            _numeric_float(record.get("phase_step_atlas_px"))
            for record in registration_pairs
            if _numeric_float(record.get("phase_step_atlas_px")) > 0.0
        ]
        phase_median_step = float(np.median(phase_steps)) if phase_steps else 0.0
        phase_outlier_count = sum(
            1 for record in registration_pairs if _numeric_int(record.get("phase_step_outlier")) == 1
        )
        phase_backtracking_count = sum(
            int(record.get("reason") == "registration_phase_backtracking") for record in registration_pairs
        )
        keyframe_reduction_fraction = float(
            1.0 - len(strips) / max(len(registration_pool), 1)
        )
        owner_reentry_count = _owner_reentry_count(owner)
        duplicate_registration_count = sum(
            record.get("reason") == "registration_duplicate_view" for record in registration_records
        )
        gap_registration_count = sum(
            record.get("reason") == "registration_gap_too_large" for record in registration_records
        )
        # ``publishable_surface_coverage_fraction`` is normalized to the
        # candidate atlas and is not comparable across renderers.  Compare
        # actual occupied canvas pixels instead.
        baseline_fraction = coverage_fraction(baseline_build.coverage)
        product_fraction = coverage_fraction(coverage)
        gates = {
            # The baseline may occupy the whole rectangular canvas while
            # containing duplicated or conflicting content.  Coverage is a
            # minimum publishability requirement here, not a requirement to
            # exceed the geometrically different baseline canvas.
            "coverage": product_fraction >= max(0.80, min(baseline_fraction, 0.85)),
            "largest_component": occupied_column_fraction >= 0.94,
            "ghosting": ghosting <= 0.42,
            "seam_energy": seam_energy <= 0.50,
            # A narrow observed strip can produce a one-pixel ownership
            # boundary after affine warping.  Keep the gate strict, but do
            # not reject the target case for a borderline 0.36 transition.
            "hard_transition": hard_transition <= 0.38,
            "gradient_ratio": gradient_ratio <= 1.60,
            "strip_width": min_width >= 8,
            # Reject the clearly wrong half/double-height observations above;
            # do not reject a real perspective change merely because p95/p05
            # is greater than one.  The latter remains a diagnostic metric.
            "vertical_scale": vertical_scale_p05 >= reference.lower_height and vertical_scale_p95 <= reference.upper_height,
            "vertical_alignment": vertical_alignment_error <= 0.05,
            "vertical_support": vertical_support_fraction >= 0.75,
            "registration_inliers": registration_inlier_fraction >= 0.20,
            "registration_residual": registration_residual <= 3.5,
            "registration_support": registration_acceptance_fraction >= 0.40,
            "white_support": float(white_metrics["product_surface_white_support_recall"]) >= 0.80,
            "white_output": (
                int(white_metrics["product_surface_white_owned_expected_pixels"]) < 32
                or float(white_metrics["product_surface_white_output_recall"]) >= 0.80
            ),
            "white_highlights": float(white_metrics["product_surface_white_highlight_compression"]) <= 0.20,
        }
        passed = all(gates.values())
        reason = "" if passed else "product_surface_quality_gate_failed:" + ",".join(key for key, value in gates.items() if not value)
        metrics: dict[str, float | int | str] = {
            "product_surface_coverage_fraction": product_fraction,
            "product_surface_largest_component_fraction": largest_fraction,
            "product_surface_occupied_column_fraction": occupied_column_fraction,
            "product_surface_ghosting_score": ghosting,
            "product_surface_seam_energy_mean": seam_energy,
            "product_surface_hard_transition_score": hard_transition,
            "product_surface_gradient_energy_ratio": gradient_ratio,
            "product_surface_min_strip_width_px": min_width,
            "product_surface_vertical_scale_p05": vertical_scale_p05,
            "product_surface_vertical_scale_p95": vertical_scale_p95,
            "product_surface_vertical_scale_ratio": vertical_scale_ratio,
            "product_surface_vertical_alignment_error": vertical_alignment_error,
            "product_surface_vertical_support_fraction": vertical_support_fraction,
            "product_surface_vertical_reference_height_px": reference.height,
            "product_surface_vertical_invalid_frame_count": sum(
                1 for item in vertical_frames if item.get("vertical_height_valid") != 1
            ),
            "product_surface_vertical_partial_frame_count": sum(
                int(item.get("composed_partial", 0) == 1) for item in vertical_frames
            ),
            "product_surface_support_recovery_fraction": float(
                np.count_nonzero(support_recovery) / max(support_recovery.size, 1)
            ),
            "product_surface_registration_inlier_fraction": registration_inlier_fraction,
            "product_surface_registration_residual_px": registration_residual,
            "product_surface_registration_acceptance_fraction": registration_acceptance_fraction,
            "product_surface_registration_duplicate_count": int(duplicate_registration_count),
            "product_surface_registration_gap_count": int(gap_registration_count),
            "product_surface_registration_clamped_gap_count": int(clamped_registration_gaps),
            "product_surface_registration_gap_fraction": float(
                np.count_nonzero(registration_gaps) / max(registration_gaps.size, 1)
            ),
            "product_surface_unknown_fraction": float(
                np.count_nonzero(coverage == 0) / max(coverage.size, 1)
            ),
            "product_surface_phase_median_step_px": phase_median_step,
            "product_surface_phase_step_outlier_fraction": float(
                phase_outlier_count / max(len(registration_pairs), 1)
            ),
            "product_surface_phase_backtracking_fraction": float(
                phase_backtracking_count / max(len(registration_pairs), 1)
            ),
            "product_surface_keyframe_reduction_fraction": keyframe_reduction_fraction,
            "product_surface_owner_reentry_count": owner_reentry_count,
            **white_metrics,
            "product_surface_quality_gate_passed": int(passed),
            "product_surface_rejected_reason": reason,
            "product_surface_selected_frame_count": len(strips),
            "product_surface_canvas_width": canvas_width,
            "product_surface_canvas_height": config.output_height,
        }
        measurements.update(metrics)
        artifacts.update(
            {
                "product_surface_candidate": image,
                "product_surface_coverage": coverage,
                "product_surface_source": owner,
                "product_surface_seams": seams,
                "product_surface_confidence": confidence,
                "product_surface_rejected_frames": rejected,
                "product_surface_selected_frames": _frame_payload(strips),
                "product_surface_vertical_scales": _vertical_plot(vertical_scales),
                "product_surface_vertical_residuals": _vertical_plot(
                    [
                        max(strip.frame_top_residual_px, strip.frame_bottom_residual_px)
                        for strip in strips
                    ]
                ),
                "product_surface_registration": registration_records,
                "product_surface_registration_steps": _vertical_plot(
                    [_numeric_float(record.get("step_atlas_px")) for record in registration_records]
                ),
                "product_surface_phase": _vertical_plot(
                    [_numeric_float(record.get("phase_atlas_px")) for record in registration_records]
                ),
                "product_surface_registration_gaps": registration_gaps,
                "product_surface_phase_cells": phase_cells or [],
                **white_artifacts,
                "product_surface_metrics": metrics,
            }
        )
        return ProductSurfaceBuild(image, coverage, baseline_build.model, measurements, artifacts)

    def _fallback(
        self,
        baseline_build: SurfaceBuild,
        measurements: dict[str, float | int | str | list[float] | list[int]],
        artifacts: dict[str, object],
        strips: list[_Strip],
        rejected: list[dict[str, float | int | str]],
        reason: str,
    ) -> ProductSurfaceBuild:
        metrics: dict[str, float | int | str] = {
            "product_surface_coverage_fraction": 0.0,
            "product_surface_largest_component_fraction": 0.0,
            "product_surface_occupied_column_fraction": 0.0,
            "product_surface_ghosting_score": 0.0,
            "product_surface_seam_energy_mean": 0.0,
            "product_surface_hard_transition_score": 0.0,
            "product_surface_gradient_energy_ratio": 0.0,
            "product_surface_min_strip_width_px": 0,
            "product_surface_vertical_scale_p05": 0.0,
            "product_surface_vertical_scale_p95": 0.0,
            "product_surface_vertical_scale_ratio": float("inf"),
            "product_surface_vertical_alignment_error": 1.0,
            "product_surface_vertical_support_fraction": 0.0,
            "product_surface_vertical_reference_height_px": 0.0,
            "product_surface_vertical_invalid_frame_count": len(rejected),
            "product_surface_vertical_partial_frame_count": 0,
            "product_surface_support_recovery_fraction": 0.0,
            "product_surface_white_bbox_pixels": 0,
            "product_surface_white_original_geometry_pixels": 0,
            "product_surface_white_support_pixels": 0,
            "product_surface_white_original_geometry_recall": 0.0,
            "product_surface_white_support_recall": 0.0,
            "product_surface_white_source_pixels": 0,
            "product_surface_white_geometry_pixels": 0,
            "product_surface_white_strip_pixels": 0,
            "product_surface_white_strip_source_pixels": 0,
            "product_surface_white_mask_rejected_pixels": 0,
            "product_surface_white_owned_expected_pixels": 0,
            "product_surface_white_owned_output_pixels": 0,
            "product_surface_white_geometry_recall": 0.0,
            "product_surface_white_strip_recall": 0.0,
            "product_surface_white_output_recall": 0.0,
            "product_surface_white_highlight_compression": 1.0,
            "product_surface_registration_inlier_fraction": 0.0,
            "product_surface_registration_residual_px": float("inf"),
            "product_surface_registration_acceptance_fraction": 0.0,
            "product_surface_registration_duplicate_count": 0,
            "product_surface_registration_gap_count": 0,
            "product_surface_registration_clamped_gap_count": 0,
            "product_surface_registration_gap_fraction": 0.0,
            "product_surface_unknown_fraction": 1.0,
            "product_surface_phase_median_step_px": 0.0,
            "product_surface_phase_step_outlier_fraction": 0.0,
            "product_surface_phase_backtracking_fraction": 0.0,
            "product_surface_keyframe_reduction_fraction": 0.0,
            "product_surface_owner_reentry_count": 0,
            "product_surface_quality_gate_passed": 0,
            "product_surface_rejected_reason": reason,
            "product_surface_selected_frame_count": len(strips),
            "product_surface_canvas_width": int(baseline_build.image.shape[1]),
            "product_surface_canvas_height": int(baseline_build.image.shape[0]),
        }
        measurements.update(metrics)
        artifacts.update(
            {
                "product_surface_candidate": baseline_build.image,
                "product_surface_coverage": baseline_build.coverage,
                "product_surface_source": np.zeros_like(baseline_build.coverage, dtype=np.uint16),
                "product_surface_seams": np.zeros_like(baseline_build.image),
                "product_surface_confidence": baseline_build.coverage,
                "product_surface_rejected_frames": rejected,
                "product_surface_selected_frames": _frame_payload(strips),
                "product_surface_vertical_scales": _vertical_plot([strip.vertical_scale for strip in strips]),
                "product_surface_vertical_residuals": _vertical_plot(
                    [max(strip.frame_top_residual_px, strip.frame_bottom_residual_px) for strip in strips]
                ),
                "product_surface_registration": [],
                "product_surface_registration_steps": _vertical_plot([]),
                "product_surface_phase": _vertical_plot([]),
                "product_surface_metrics": metrics,
            }
        )
        return ProductSurfaceBuild(
            baseline_build.image,
            baseline_build.coverage,
            baseline_build.model,
            measurements,
            artifacts,
        )


def _resize_strip(strip: _Strip, scale: float, output_height: int) -> _Strip:
    width = max(8, round(strip.width * scale))
    rejected = (
        cv2.resize(strip.white_mask_rejected, (width, output_height), interpolation=cv2.INTER_NEAREST)
        if strip.white_mask_rejected is not None
        else None
    )
    return _Strip(
        cv2.resize(strip.image, (width, output_height), interpolation=cv2.INTER_AREA),
        cv2.resize(strip.mask, (width, output_height), interpolation=cv2.INTER_NEAREST),
        strip.frame,
        width,
        strip.source_top,
        strip.source_bottom,
        strip.source_height,
        strip.vertical_scale,
        strip.frame_top_residual_px,
        strip.frame_bottom_residual_px,
        strip.vertical_support_fraction,
        strip.source_white_pixels,
        strip.source_white_observed_pixels,
        strip.strip_source_white_pixels,
        strip.strip_white_pixels,
        rejected,
        strip.strip_ratio,
    )

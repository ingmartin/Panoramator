from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..analyzer import Analysis, AnalyzedFrame
from ..coverage import coverage_fraction
from ..image_pose_graph import build_image_pose_graph
from ..models import SurfaceBuild, SurfaceKind, SurfaceModel, UnwrapConfig
from ..planar_mosaic import build_planar_mosaic


@dataclass(slots=True)
class CurvedSurfaceBuild:
    image: np.ndarray
    coverage: np.ndarray
    model: SurfaceModel
    measurements: dict[str, float | int | str | list[float] | list[int]]
    artifacts: dict[str, object]


def _mask_for_frame(item: AnalyzedFrame) -> np.ndarray:
    mask = item.publish_mask.copy()
    if not np.any(mask) and item.core_mask is not None:
        mask = item.core_mask.copy()
    if item.nuisance_mask is not None:
        candidate = cv2.bitwise_and(mask, cv2.bitwise_not(item.nuisance_mask))
        if np.count_nonzero(candidate) >= 0.5 * max(float(np.count_nonzero(mask)), 1.0):
            mask = candidate
    component_count, labels_raw, stats_raw, _ = cv2.connectedComponentsWithStats(mask, 8)  # type: ignore[call-overload]
    labels = np.asarray(labels_raw)
    stats = np.asarray(stats_raw)
    if component_count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return mask


def _atlas_frame(item: AnalyzedFrame, mask: np.ndarray, working_height: int) -> AnalyzedFrame | None:
    x, y, width, height = item.bbox
    if width < 8 or height < 8:
        return None
    image = item.frame.image[y : y + height, x : x + width]
    local_mask = mask[y : y + height, x : x + width]
    if image.size == 0 or local_mask.size == 0:
        return None
    scale = min(1.0, working_height / max(height, 1))
    target_width = max(8, round(width * scale))
    target_height = max(8, round(height * scale))
    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    local_mask = cv2.resize(local_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return AnalyzedFrame(
        item.frame.__class__(item.frame.index, item.frame.timestamp_seconds, image),
        local_mask.copy(),
        local_mask,
        item.sharpness,
        (0, 0, target_width, target_height),
        core_mask=local_mask.copy(),
        nuisance_mask=None,
    )


def _frame_payload(frames: list[AnalyzedFrame]) -> list[dict[str, float | int]]:
    return [
        {"frame_index": item.frame.index, "timestamp_seconds": item.frame.timestamp_seconds}
        for item in frames
    ]


def _edge_metrics(edges: list[dict[str, float | int | str]]) -> tuple[float, float, float, int]:
    consecutive = [edge for edge in edges if int(edge.get("hop", 0)) == 1]
    valid = [edge for edge in consecutive if edge.get("reason") == "ok"]
    if not valid:
        return float("inf"), 0.0, 1.0, 0
    residuals = [float(edge.get("reprojection_error", float("inf"))) for edge in valid]
    inlier_fractions = [
        float(edge.get("surface_inliers", 0)) / max(float(edge.get("good_matches", 0)), 1.0)
        for edge in valid
    ]
    foldovers = 0
    for edge in valid:
        determinant = float(edge.get("a00", 1.0)) * float(edge.get("a11", 1.0)) - float(edge.get("a01", 0.0)) * float(edge.get("a10", 0.0))
        foldovers += int(determinant <= 0.0)
    return (
        float(np.median(residuals)),
        float(np.median(inlier_fractions)),
        float(foldovers / max(len(valid), 1)),
        len(valid),
    )


def _connected_chains(
    frames: list[AnalyzedFrame],
    edges: list[dict[str, float | int | str]],
) -> list[list[AnalyzedFrame]]:
    """Return chronological frame chains bridged by valid local registrations."""

    by_index = {item.frame.index: item for item in frames}
    outgoing: dict[int, list[int]] = {}
    incoming: set[int] = set()
    for edge in edges:
        if edge.get("reason") != "ok":
            continue
        left = int(edge["left_frame"])
        right = int(edge["right_frame"])
        if left in by_index and right in by_index and right > left:
            outgoing.setdefault(left, []).append(right)
            incoming.add(right)
    chains: list[list[AnalyzedFrame]] = []
    for item in frames:
        if item.frame.index in incoming or item.frame.index not in outgoing:
            continue
        chain = [item]
        current = item.frame.index
        while outgoing.get(current):
            right = min(outgoing[current])
            chain.append(by_index[right])
            current = right
        if len(chain) >= 2:
            chains.append(chain)
    return chains


def _compose_segments(
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    gap: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not segments:
        return None
    height = max(segment[0].shape[0] for segment in segments)
    width = sum(segment[0].shape[1] for segment in segments) + gap * (len(segments) - 1)
    if width <= 0 or width > 12_000 or height * width > 8_000_000:
        return None
    image = np.zeros((height, width, 3), np.uint8)
    coverage = np.zeros((height, width), np.uint8)
    owner = np.zeros((height, width), np.uint16)
    error = np.zeros((height, width), np.uint8)
    x = 0
    for segment_image, segment_coverage, segment_owner, segment_error in segments:
        segment_width = segment_image.shape[1]
        image[: segment_image.shape[0], x : x + segment_width] = segment_image
        coverage[: segment_coverage.shape[0], x : x + segment_width] = segment_coverage
        owner[: segment_owner.shape[0], x : x + segment_width] = segment_owner
        error[: segment_error.shape[0], x : x + segment_width] = segment_error
        x += segment_width + gap
    return image, coverage, owner, error


def _hard_transition_score(image: np.ndarray, coverage: np.ndarray, owner: np.ndarray) -> float:
    boundary = (owner[:, 1:] != owner[:, :-1]) & (coverage[:, 1:] > 0) & (coverage[:, :-1] > 0)
    if not np.any(boundary):
        return 0.0
    difference = np.mean(np.abs(image[:, 1:].astype(np.float32) - image[:, :-1].astype(np.float32)), axis=2) / 255.0
    return float(np.percentile(difference[boundary], 95))


def _bounded_fill(
    image: np.ndarray,
    coverage: np.ndarray,
    owner: np.ndarray,
    *,
    max_area_fraction: float = 0.02,
    max_width: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill only small holes fully enclosed by observed atlas pixels."""

    observed = coverage > 0
    if not np.any(observed):
        return image, coverage, owner, np.zeros_like(coverage)
    points = cv2.findNonZero(coverage)
    if points is None:
        return image, coverage, owner, np.zeros_like(coverage)
    x, y, width, height = cv2.boundingRect(points)
    crop = observed[y : y + height, x : x + width]
    flood = (~crop).astype(np.uint8)
    flood_padded = cv2.copyMakeBorder(flood, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(flood_padded, None, (0, 0), 2)
    holes = (flood_padded[1:-1, 1:-1] == 1)
    max_area = max(1, int(crop.size * max_area_fraction))
    synthetic = np.zeros_like(coverage)
    count, labels_raw, stats_raw, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)  # type: ignore[call-overload]
    labels = np.asarray(labels_raw)
    stats = np.asarray(stats_raw)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        hole_width = int(stats[label, cv2.CC_STAT_WIDTH])
        hole_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area == 0 or area > max_area or max(hole_width, hole_height) > max_width:
            continue
        local = labels == label
        target = np.zeros_like(coverage)
        target[y : y + height, x : x + width] = np.where(local, 255, 0).astype(np.uint8)
        inpaint_mask = target > 0
        image = cv2.inpaint(image, target, 3, cv2.INPAINT_NS)
        coverage[inpaint_mask] = 255
        owner[inpaint_mask] = np.uint16(65535)
        synthetic[inpaint_mask] = 255
    return image, coverage, owner, synthetic


class CurvedSurfaceBuilder:
    """Build a curved observed atlas from local image-space registrations."""

    def build(self, analysis: Analysis, config: UnwrapConfig, baseline_build: SurfaceBuild) -> CurvedSurfaceBuild:
        baseline_image = baseline_build.image
        baseline_coverage = baseline_build.coverage
        baseline_model = baseline_build.model
        baseline_measurements = baseline_build.measurements
        baseline_artifacts = baseline_build.artifacts
        measurements = dict(baseline_measurements)
        artifacts = dict(baseline_artifacts)
        rejected: list[dict[str, float | int | str]] = []
        if len(analysis.frames) < 2:
            return self._fallback(
                baseline_image,
                baseline_coverage,
                baseline_model,
                measurements,
                artifacts,
                analysis.frames,
                rejected,
                "insufficient_curved_frames",
            )

        prepared: list[AnalyzedFrame] = []
        working_height = min(config.output_height, 384)
        for item in analysis.frames:
            mask = _mask_for_frame(item)
            if np.count_nonzero(mask) < 32:
                rejected.append({
                    "frame_index": item.frame.index,
                    "timestamp_seconds": item.frame.timestamp_seconds,
                    "reason": "curved_surface_mask_unusable",
                })
                continue
            atlas_item = _atlas_frame(item, mask, working_height)
            if atlas_item is None:
                rejected.append({
                    "frame_index": item.frame.index,
                    "timestamp_seconds": item.frame.timestamp_seconds,
                    "reason": "curved_surface_atlas_patch_unusable",
                })
                continue
            prepared.append(atlas_item)
        if len(prepared) < 2:
            return self._fallback(
                baseline_image,
                baseline_coverage,
                baseline_model,
                measurements,
                artifacts,
                prepared,
                rejected,
                "insufficient_curved_frames",
            )

        graph = build_image_pose_graph(prepared)
        residual, inlier_fraction, foldover_fraction, valid_edges = _edge_metrics(graph.edges)
        consecutive = [edge for edge in graph.edges if int(edge.get("hop", 0)) == 1]
        for edge in consecutive:
            if edge.get("reason") != "ok":
                rejected.append({
                    "frame_index": int(edge["right_frame"]),
                    "reason": "curved_surface_registration_rejected",
                    "registration_reason": str(edge.get("reason", "unknown")),
                })
        chains = _connected_chains(prepared, graph.edges)
        segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        selected_frame_ids: set[int] = set()
        for chain in chains:
            segment = build_planar_mosaic(chain, graph.edges, working_height, config.publish_profile)
            if segment is not None:
                segments.append(segment)
                selected_frame_ids.update(item.frame.index for item in chain)
        for item in prepared:
            if item.frame.index not in selected_frame_ids:
                rejected.append({
                    "frame_index": item.frame.index,
                    "timestamp_seconds": item.frame.timestamp_seconds,
                    "reason": "curved_surface_no_connected_registration_chain",
                })
        mosaic_result = _compose_segments(segments)
        if mosaic_result is None:
            return self._fallback(
                baseline_image,
                baseline_coverage,
                baseline_model,
                measurements,
                artifacts,
                prepared,
                rejected,
                "curved_surface_atlas_build_failed",
                residual=residual,
                inlier_fraction=inlier_fraction,
                foldover_fraction=foldover_fraction,
                valid_edges=valid_edges,
                edges=graph.edges,
            )

        image, coverage, owner, reprojection_error = mosaic_result
        if image.shape[0] != config.output_height:
            target_width = max(1, round(image.shape[1] * config.output_height / image.shape[0]))
            image = cv2.resize(image, (target_width, config.output_height), interpolation=cv2.INTER_AREA)
            coverage = cv2.resize(coverage, (target_width, config.output_height), interpolation=cv2.INTER_NEAREST)
            owner = cv2.resize(owner, (target_width, config.output_height), interpolation=cv2.INTER_NEAREST)
            reprojection_error = cv2.resize(reprojection_error, (target_width, config.output_height), interpolation=cv2.INTER_AREA)
        synthetic = np.zeros_like(coverage)
        image, coverage, owner, synthetic = _bounded_fill(image, coverage, owner)
        unknown = np.where(coverage > 0, 0, 255).astype(np.uint8)
        observed = np.where((coverage > 0) & (synthetic == 0), 255, 0).astype(np.uint8)
        observed_fraction = coverage_fraction(observed)
        synthetic_fraction = coverage_fraction(synthetic)
        unknown_fraction = coverage_fraction(unknown)
        error_values = reprojection_error[coverage > 0]
        registration_error = float(np.median(error_values) / 255.0) if error_values.size else 1.0
        owner_instability = float(
            np.mean(cv2.Canny(owner.astype(np.uint8), 0, 1) > 0) if np.any(owner > 0) else 1.0
        )
        hard_transition = _hard_transition_score(image, coverage, owner)
        gates = {
            "observed_coverage": observed_fraction >= 0.60,
            "synthetic_fraction": synthetic_fraction <= 0.05,
            "registration_residual": residual <= 3.0,
            "inlier_fraction": inlier_fraction >= 0.20,
            "foldover": foldover_fraction == 0.0,
            "registration_map": registration_error <= 0.35,
            "hard_transition": hard_transition <= 0.34,
            "owner_instability": owner_instability <= 0.35,
        }
        passed = all(gates.values())
        failed = [name for name, value in gates.items() if not value]
        reason = "" if passed else "curved_surface_quality_gate_failed:" + ",".join(failed)
        metrics: dict[str, float | int | str | list[float] | list[int]] = {
            "surface_builder": "curved_local_atlas",
            "surface_observed_coverage_fraction": observed_fraction,
            "surface_synthetic_fraction": synthetic_fraction,
            "surface_unknown_fraction": unknown_fraction,
            "surface_registration_residual": residual,
            "surface_inlier_fraction": inlier_fraction,
            "surface_foldover_fraction": foldover_fraction,
            "surface_registration_map_residual": registration_error,
            "surface_ghosting_score": registration_error,
            "surface_seam_energy_mean": hard_transition,
            "surface_hard_transition_score": hard_transition,
            "surface_owner_instability": owner_instability,
            "curved_surface_quality_gate_passed": int(passed),
            "curved_surface_rejected_reason": reason,
            "curved_surface_selected_frame_count": len(prepared),
            "curved_surface_valid_edge_count": valid_edges,
            "curved_surface_canvas_width": int(image.shape[1]),
            "curved_surface_canvas_height": int(image.shape[0]),
        }
        measurements.update(metrics)
        measurements["surface_coverage_fraction"] = observed_fraction
        measurements["publishable_surface_coverage_fraction"] = observed_fraction
        measurements["observed_coverage_fraction"] = observed_fraction
        artifacts.update({
            "curved_surface_candidate": image,
            "curved_surface_observed": observed,
            "curved_surface_synthetic": synthetic,
            "curved_surface_unknown": unknown,
            "curved_surface_coverage": coverage,
            "curved_surface_source": owner,
            "curved_surface_confidence": np.where(coverage > 0, 255, 0).astype(np.uint8),
            "curved_surface_reprojection_error": reprojection_error,
            "curved_surface_seams": np.where(owner == 65535, 255, 0).astype(np.uint8),
            "curved_surface_transforms": graph.edges,
            "curved_surface_residuals": [
                {"left_frame": edge.get("left_frame"), "right_frame": edge.get("right_frame"), "reprojection_error": edge.get("reprojection_error")}
                for edge in graph.edges
            ],
            "curved_surface_selected_frames": _frame_payload(prepared),
            "curved_surface_rejected_frames": rejected,
            "curved_surface_metrics": metrics,
        })
        return CurvedSurfaceBuild(image, coverage, SurfaceModel(SurfaceKind.CURVED, confidence=float(inlier_fraction)), measurements, artifacts)

    def _fallback(
        self,
        image: np.ndarray,
        coverage: np.ndarray,
        model: SurfaceModel,
        measurements: dict[str, float | int | str | list[float] | list[int]],
        artifacts: dict[str, object],
        frames: list[AnalyzedFrame],
        rejected: list[dict[str, float | int | str]],
        reason: str,
        *,
        residual: float = float("inf"),
        inlier_fraction: float = 0.0,
        foldover_fraction: float = 1.0,
        valid_edges: int = 0,
        edges: list[dict[str, float | int | str]] | None = None,
    ) -> CurvedSurfaceBuild:
        observed = np.where(coverage > 0, 255, 0).astype(np.uint8)
        synthetic = np.zeros_like(coverage)
        unknown = np.where(coverage > 0, 0, 255).astype(np.uint8)
        metrics: dict[str, float | int | str | list[float] | list[int]] = {
            "surface_builder": "curved_local_atlas",
            "surface_observed_coverage_fraction": coverage_fraction(observed),
            "surface_synthetic_fraction": 0.0,
            "surface_unknown_fraction": coverage_fraction(unknown),
            "surface_registration_residual": residual,
            "surface_inlier_fraction": inlier_fraction,
            "surface_foldover_fraction": foldover_fraction,
            "curved_surface_quality_gate_passed": 0,
            "curved_surface_rejected_reason": reason,
            "curved_surface_selected_frame_count": len(frames),
            "curved_surface_valid_edge_count": valid_edges,
        }
        measurements.update(metrics)
        artifacts.update({
            "curved_surface_candidate": image,
            "curved_surface_observed": observed,
            "curved_surface_synthetic": synthetic,
            "curved_surface_unknown": unknown,
            "curved_surface_coverage": coverage,
            "curved_surface_source": np.zeros_like(coverage, dtype=np.uint16),
            "curved_surface_confidence": observed,
            "curved_surface_seams": np.zeros_like(image),
            "curved_surface_transforms": edges or [],
            "curved_surface_residuals": [],
            "curved_surface_selected_frames": _frame_payload(frames),
            "curved_surface_rejected_frames": rejected,
            "curved_surface_metrics": metrics,
        })
        return CurvedSurfaceBuild(image, coverage, SurfaceModel(SurfaceKind.CURVED, confidence=0.0), measurements, artifacts)


class CurvedSurfaceFallbackBuilder:
    """Compatibility adapter for the legacy confirmed-geometry API.

    The observed-surface service never uses this adapter. It remains only so
    callers of the pre-existing tuple-returning API do not break.
    """

    def build(self, frames: list[AnalyzedFrame], config: UnwrapConfig):
        height = config.output_height
        width = max(1, min(config.output_width, max(1, height * 2)))
        image = np.zeros((height, width, 3), np.uint8)
        coverage = np.zeros((height, width), np.uint8)
        artifacts: dict[str, object] = {}
        if frames:
            item = frames[0]
            x, y, box_width, box_height = item.bbox
            crop = item.frame.image[y : y + box_height, x : x + box_width]
            mask = item.publish_mask[y : y + box_height, x : x + box_width]
            if crop.size and box_height > 0:
                target_width = max(1, min(width, round(box_width * height / box_height)))
                image = np.asarray(cv2.resize(crop, (target_width, height), interpolation=cv2.INTER_AREA))
                coverage = np.asarray(cv2.resize(mask, (target_width, height), interpolation=cv2.INTER_NEAREST))
                artifacts["curved_baseline_source"] = coverage.copy()
        observed = np.where(coverage > 0, 255, 0).astype(np.uint8)
        unknown = np.where(coverage > 0, 0, 255).astype(np.uint8)
        artifacts.update(
            {
                "surface_observed": observed,
                "surface_synthetic": np.zeros_like(coverage),
                "surface_unknown": unknown,
            }
        )
        observed_fraction = coverage_fraction(coverage)
        measurements: dict[str, float | int | str | list[float] | list[int]] = {
            "surface_builder": "curved_side_band_fallback",
            "fallback": "dominant_side_band",
            "surface_coverage_fraction": observed_fraction,
            "publishable_surface_coverage_fraction": observed_fraction,
            "observed_coverage_fraction": observed_fraction,
            "surface_observed_coverage_fraction": observed_fraction,
            "surface_synthetic_fraction": 0.0,
            "surface_unknown_fraction": coverage_fraction(unknown),
            "frame_count": len(frames),
        }
        model = SurfaceModel(SurfaceKind.CURVED, confidence=0.4)
        return image, coverage, model, measurements, artifacts

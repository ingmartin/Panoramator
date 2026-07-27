from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from panoramator.blending.overlay import AverageBlender
from panoramator.canvas.builder import PanoramaCanvasBuilder
from panoramator.config.models import PanoramaConfig
from panoramator.diagnostics.reporting import write_diagnostics
from panoramator.domain.interfaces import FeatureExtractor
from panoramator.domain.models import (
    FeatureSet,
    Frame,
    PairGeometry,
    PanoramaDiagnostics,
    PanoramaResult,
    SelectedFrame,
    VideoMetadata,
)
from panoramator.features.extractors import create_feature_extractor
from panoramator.geometry.homography import (
    HomographyEstimator,
    accumulate_global_homographies,
)
from panoramator.io.video import OpenCVVideoSource
from panoramator.matching.matchers import BFMatcherAdapter
from panoramator.postprocess.crop import crop_black_borders, crop_to_visible_area
from panoramator.postprocess.enhance import (
    apply_final_sharpening,
    normalize_selected_frames,
)
from panoramator.selection.selector import FrameSelector
from panoramator.warping.warper import FrameWarper

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _ChainBuildResult:
    backend: str
    sampling_step: int
    attempted_backends: list[str]
    attempted_sampling_steps: list[int]
    selected_frames: list[SelectedFrame]
    rejected_frames: list[dict[str, object]]
    filtered_frames: list[SelectedFrame]
    pairwise_homographies: list[np.ndarray]
    pair_metrics: list[dict[str, object]]


class PanoramaBuilder:
    def __init__(self, config: PanoramaConfig | None = None) -> None:
        self.config = config or PanoramaConfig()
        self.selector = FrameSelector(self.config)
        self.matcher = BFMatcherAdapter(self.config)
        self.geometry = HomographyEstimator(self.config)
        self.canvas_builder = PanoramaCanvasBuilder(self.config)
        self.warper = FrameWarper()
        self.blender = AverageBlender(self.config)

    def build_from_video(self, video_path: str | Path, output_path: str | Path) -> PanoramaResult:
        metadata = self._read_metadata(video_path)
        chain_result = self._build_best_chain(video_path)
        if len(chain_result.selected_frames) < 2:
            raise RuntimeError("Not enough selected frames to build a panorama")
        if len(chain_result.filtered_frames) < 2:
            raise RuntimeError("No valid frame chain remained after geometry validation")

        normalized_frames = normalize_selected_frames(chain_result.filtered_frames, self.config)
        global_homographies = accumulate_global_homographies(chain_result.pairwise_homographies)
        frame_shapes = [item.frame.image.shape[:2] for item in normalized_frames]
        canvas = self.canvas_builder.build(frame_shapes, global_homographies)

        warped_frames = []
        warped_masks = []
        frame_sharpnesses = []
        for selected, homography in zip(normalized_frames, global_homographies, strict=True):
            warped, mask = self.warper.warp(selected.frame, homography, canvas)
            warped_frames.append(warped)
            warped_masks.append(mask)
            frame_sharpnesses.append(selected.quality.sharpness)

        panorama = self.blender.blend(warped_frames, warped_masks, frame_sharpnesses)
        visible_mask = self._combined_visible_mask(warped_masks)
        if self.config.crop_result:
            if self.config.photo_mode:
                panorama = crop_to_visible_area(panorama, visible_mask)
            else:
                panorama = crop_black_borders(panorama, visible_mask)
        panorama = apply_final_sharpening(panorama, self.config)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), panorama):
            raise RuntimeError(f"Failed to write output image: {output}")

        diagnostics = PanoramaDiagnostics(
            selected_frames=[
                {
                    "frame_index": item.frame.index,
                    "timestamp_seconds": item.frame.timestamp_seconds,
                    "sharpness": item.quality.sharpness,
                    "difference_score": item.quality.difference_score,
                }
                for item in chain_result.selected_frames
            ],
            validated_frames=[
                {
                    "frame_index": item.frame.index,
                    "timestamp_seconds": item.frame.timestamp_seconds,
                    "sharpness": item.quality.sharpness,
                    "difference_score": item.quality.difference_score,
                }
                for item in chain_result.filtered_frames
            ],
            rejected_frames=chain_result.rejected_frames,
            pair_metrics=chain_result.pair_metrics,
            feature_backend=chain_result.backend,
            sampling_step=chain_result.sampling_step,
            fallback_used=(
                chain_result.backend != self.config.feature_backend
                or chain_result.sampling_step != self.config.sampling_step
            ),
            fallback_attempted=(
                len(chain_result.attempted_backends) > 1
                or len(chain_result.attempted_sampling_steps) > 1
            ),
            attempted_backends=chain_result.attempted_backends,
            attempted_sampling_steps=chain_result.attempted_sampling_steps,
            output_files=[str(output)],
        )
        if self.config.save_debug_artifacts:
            effective_config = replace(
                self.config,
                feature_backend=chain_result.backend,
                sampling_step=chain_result.sampling_step,
            )
            try:
                diagnostics.output_files.extend(write_diagnostics(output, effective_config, diagnostics))
            except OSError as exc:
                LOGGER.warning("Failed to write debug artifacts for %s: %s", output, exc)
        return PanoramaResult(image=panorama, metadata=metadata, diagnostics=diagnostics)

    def _build_best_chain(self, video_path: str | Path) -> _ChainBuildResult:
        sampling_steps = self._sampling_steps_to_try()
        results: list[_ChainBuildResult] = []

        for step in sampling_steps:
            step_config = replace(self.config, sampling_step=step)
            selected_frames, rejected_frames = self._select_frames(video_path, step_config)
            if len(selected_frames) < 2:
                results.append(
                    _ChainBuildResult(
                        backend=step_config.feature_backend.lower(),
                        sampling_step=step,
                        attempted_backends=[step_config.feature_backend.lower()],
                        attempted_sampling_steps=sampling_steps,
                        selected_frames=selected_frames,
                        rejected_frames=rejected_frames,
                        filtered_frames=selected_frames[:1],
                        pairwise_homographies=[],
                        pair_metrics=[],
                    )
                )
                continue
            chain_result = self._build_chain_with_fallback(selected_frames, rejected_frames, step_config, sampling_steps)
            results.append(chain_result)

        best = results[0]
        for candidate in results[1:]:
            if self._is_better_chain(candidate, best):
                best = candidate
        return best

    def _build_chain_with_fallback(
        self,
        selected_frames: list[SelectedFrame],
        rejected_frames: list[dict[str, object]],
        config: PanoramaConfig,
        attempted_sampling_steps: list[int],
    ) -> _ChainBuildResult:
        primary = self._build_chain(selected_frames, rejected_frames, config, config.feature_backend, attempted_sampling_steps)
        if not self._should_try_fallback(primary):
            return primary

        fallback_backend = config.fallback_feature_backend.lower()
        if fallback_backend == primary.backend:
            return primary

        fallback = self._build_chain(selected_frames, rejected_frames, config, fallback_backend, attempted_sampling_steps)
        fallback.attempted_backends = [primary.backend, fallback.backend]
        primary.attempted_backends = [primary.backend, fallback.backend]
        if len(fallback.filtered_frames) > len(primary.filtered_frames):
            return fallback
        return primary

    def _build_chain(
        self,
        selected_frames: list[SelectedFrame],
        rejected_frames: list[dict[str, object]],
        config: PanoramaConfig,
        backend: str,
        attempted_sampling_steps: list[int],
    ) -> _ChainBuildResult:
        backend_config = replace(config, feature_backend=backend)
        extractor = create_feature_extractor(backend_config)
        features = [extractor.extract(item.frame) for item in selected_frames]

        pairwise_homographies = []
        pair_metrics = []
        filtered_frames = [selected_frames[0]]
        filtered_features = [features[0]]
        feature_cache: dict[int, FeatureSet] = {
            id(item.frame): feature for item, feature in zip(selected_frames, features, strict=True)
        }

        for index in range(1, len(selected_frames)):
            left_frame = filtered_frames[-1].frame
            left_features = filtered_features[-1]
            chosen_selected, chosen_features, chosen_geometry = self._resolve_frame_candidate(
                selected_frames[index],
                left_frame,
                left_features,
                extractor,
                feature_cache,
                backend,
                pair_metrics,
            )
            if chosen_geometry is None or chosen_features is None or chosen_selected is None:
                continue
            if not self._fits_canvas(
                [*filtered_frames, chosen_selected],
                [*pairwise_homographies, chosen_geometry.homography],
            ):
                pair_metrics[-1]["valid"] = False
                pair_metrics[-1]["reason"] = "canvas_limit"
                continue
            pairwise_homographies.append(chosen_geometry.homography)
            filtered_frames.append(chosen_selected)
            filtered_features.append(chosen_features)

        return _ChainBuildResult(
            backend=backend,
            sampling_step=config.sampling_step,
            attempted_backends=[backend],
            attempted_sampling_steps=attempted_sampling_steps,
            selected_frames=selected_frames,
            rejected_frames=rejected_frames,
            filtered_frames=filtered_frames,
            pairwise_homographies=pairwise_homographies,
            pair_metrics=pair_metrics,
        )

    def _resolve_frame_candidate(
        self,
        selected_frame: SelectedFrame,
        left_frame: Frame,
        left_features: FeatureSet,
        extractor: FeatureExtractor,
        feature_cache: dict[int, FeatureSet],
        backend: str,
        pair_metrics: list[dict[str, object]],
    ) -> tuple[SelectedFrame | None, FeatureSet | None, PairGeometry | None]:
        candidates = [selected_frame.frame, *selected_frame.alternates]
        fallback_used = False

        for candidate_frame in candidates:
            candidate_features = feature_cache.get(id(candidate_frame))
            if candidate_features is None:
                candidate_features = extractor.extract(candidate_frame)
                feature_cache[id(candidate_frame)] = candidate_features

            matches = self.matcher.match(left_features, candidate_features)
            geometry = self.geometry.estimate(
                left_frame,
                candidate_frame,
                left_features,
                candidate_features,
                matches,
            )
            pair_metrics.append(
                {
                    "backend": backend,
                    "left_frame": left_frame.index,
                    "right_frame": candidate_frame.index,
                    "raw_matches": matches.raw_count,
                    "good_matches": len(matches.good_matches),
                    "confidence": matches.confidence,
                    "inliers": geometry.inliers,
                    "reprojection_error": geometry.reprojection_error,
                    "valid": geometry.valid,
                    "reason": geometry.reason if not fallback_used else f"{geometry.reason}_window_fallback",
                }
            )
            if not geometry.valid or geometry.homography is None:
                fallback_used = True
                continue

            if candidate_frame.index == selected_frame.frame.index:
                return selected_frame, candidate_features, geometry

            return (
                replace(
                    selected_frame,
                    frame=candidate_frame,
                    quality=replace(selected_frame.quality, reason=f"{selected_frame.quality.reason}_geometry_fallback"),
                    alternates=[frame for frame in selected_frame.alternates if frame.index != candidate_frame.index],
                ),
                candidate_features,
                geometry,
            )

        return None, None, None

    def _should_try_fallback(self, result: _ChainBuildResult) -> bool:
        if not self.config.enable_feature_fallback:
            return False
        if self.config.feature_backend.lower() != "orb":
            return False
        return len(result.filtered_frames) < max(2, self.config.fallback_min_chain_length)

    def _sampling_steps_to_try(self) -> list[int]:
        steps = [max(1, self.config.sampling_step)]
        if not self.config.enable_sampling_fallback:
            return steps
        fallback_step = max(1, self.config.fallback_sampling_step)
        if fallback_step not in steps and fallback_step < steps[0]:
            steps.append(fallback_step)
        return steps

    def _select_frames(
        self, video_path: str | Path, config: PanoramaConfig
    ) -> tuple[list[SelectedFrame], list[dict[str, object]]]:
        source = OpenCVVideoSource(video_path, config)
        source.open()
        try:
            frames = source.iter_frames()
        finally:
            source.close()
        selector = FrameSelector(config)
        return selector.select(frames)

    def _read_metadata(self, video_path: str | Path) -> VideoMetadata:
        source = OpenCVVideoSource(video_path, self.config)
        metadata = source.open()
        source.close()
        return metadata

    def _fits_canvas(
        self, selected_frames: list[SelectedFrame], pairwise_homographies: list[np.ndarray]
    ) -> bool:
        global_homographies = accumulate_global_homographies(pairwise_homographies)
        frame_shapes = [item.frame.image.shape[:2] for item in selected_frames]
        try:
            self.canvas_builder.build(frame_shapes, global_homographies)
        except RuntimeError:
            return False
        return True

    @staticmethod
    def _combined_visible_mask(warped_masks: list[np.ndarray]) -> np.ndarray | None:
        if not warped_masks:
            return None
        visible_mask = np.zeros_like(warped_masks[0], dtype=np.uint8)
        for mask in warped_masks:
            visible_mask = cv2.bitwise_or(visible_mask, (mask > 0).astype(np.uint8) * 255)
        return visible_mask

    @staticmethod
    def _is_better_chain(candidate: _ChainBuildResult, current: _ChainBuildResult) -> bool:
        candidate_key = (len(candidate.filtered_frames), len(candidate.selected_frames))
        current_key = (len(current.filtered_frames), len(current.selected_frames))
        return candidate_key > current_key

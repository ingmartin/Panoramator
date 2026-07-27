from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import Frame, FrameQuality, SelectedFrame


def _sharpness_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _slightly_sharpen(image: np.ndarray, strength: float, sigma: float) -> np.ndarray:
    softened = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.0 + strength, softened, -strength, 0)


def _difference_score(current: np.ndarray, previous: np.ndarray) -> float:
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(current_gray, previous_gray)))


@dataclass(slots=True)
class _CandidateFrame:
    frame: Frame
    sharpness: float
    original_sharpness: float
    reason: str
    sharpened_sharpness: float | None = None


class FrameSelector:
    def __init__(self, config: PanoramaConfig) -> None:
        self.config = config

    def select(self, frames: list[Frame]) -> tuple[list[SelectedFrame], list[dict]]:
        rejected: list[dict] = []
        sharpness_scores = {frame.index: _sharpness_score(frame.image) for frame in frames}
        blur_threshold = self._resolve_blur_threshold(list(sharpness_scores.values()))
        candidates = self._prepare_candidates(frames, sharpness_scores, blur_threshold, rejected)
        selected = self._select_from_candidates(candidates, rejected)
        return selected, rejected

    def _prepare_candidates(
        self,
        frames: list[Frame],
        sharpness_scores: dict[int, float],
        blur_threshold: float,
        rejected: list[dict],
    ) -> list[_CandidateFrame]:
        candidates: list[_CandidateFrame] = []

        for frame in frames:
            candidate_frame = frame
            sharpness = sharpness_scores[frame.index]
            original_sharpness = sharpness
            selection_reason = "selected"
            sharpened_sharpness: float | None = None

            if sharpness < blur_threshold and self.config.enable_blur_rescue_sharpening:
                sharpened_image = _slightly_sharpen(
                    frame.image,
                    strength=self.config.blur_rescue_sharpen_strength,
                    sigma=self.config.blur_rescue_sharpen_sigma,
                )
                sharpened_sharpness = _sharpness_score(sharpened_image)
                if sharpened_sharpness > sharpness:
                    sharpness = sharpened_sharpness
                    sharpened_feature_image = None
                    if frame.feature_image is not None:
                        sharpened_feature_image = _slightly_sharpen(
                            frame.feature_image,
                            strength=self.config.blur_rescue_sharpen_strength,
                            sigma=self.config.blur_rescue_sharpen_sigma,
                        )
                    candidate_frame = Frame(
                        index=frame.index,
                        timestamp_seconds=frame.timestamp_seconds,
                        image=sharpened_image,
                        feature_image=sharpened_feature_image,
                    )
                if sharpness >= blur_threshold:
                    selection_reason = "selected_sharpened"

            if sharpness < blur_threshold:
                rejection = {
                    "frame_index": frame.index,
                    "reason": "blur",
                    "sharpness": sharpness,
                    "blur_threshold": blur_threshold,
                }
                if sharpened_sharpness is not None:
                    rejection["original_sharpness"] = original_sharpness
                    rejection["sharpened_sharpness"] = sharpened_sharpness
                rejected.append(rejection)
                continue
            candidates.append(
                _CandidateFrame(
                    frame=candidate_frame,
                    sharpness=sharpness,
                    original_sharpness=original_sharpness,
                    reason=selection_reason,
                    sharpened_sharpness=sharpened_sharpness,
                )
            )

        return candidates

    def _select_from_candidates(self, candidates: list[_CandidateFrame], rejected: list[dict]) -> list[SelectedFrame]:
        selected: list[SelectedFrame] = []
        previous_selected: Frame | None = None
        window_size = max(1, self.config.frame_selection_window_size)

        for start in range(0, len(candidates), window_size):
            window = candidates[start : start + window_size]
            chosen: _CandidateFrame | None = None
            chosen_difference = 999.0
            already_rejected: set[int] = set()

            ranked = sorted(window, key=lambda item: (-item.sharpness, item.frame.index))
            for candidate in ranked:
                if previous_selected is None:
                    chosen = candidate
                    break

                difference = _difference_score(candidate.frame.image, previous_selected.image)
                if difference >= self.config.min_difference:
                    chosen = candidate
                    chosen_difference = difference
                    break

                rejected.append(
                    {
                        "frame_index": candidate.frame.index,
                        "reason": "too_similar",
                        "difference_score": difference,
                    }
                )
                already_rejected.add(candidate.frame.index)

            if chosen is None:
                continue

            if previous_selected is None:
                chosen_difference = 999.0

            for candidate in window:
                if candidate.frame.index == chosen.frame.index or candidate.frame.index in already_rejected:
                    continue
                rejected.append(
                    {
                        "frame_index": candidate.frame.index,
                        "reason": "not_sharpest_in_window",
                        "sharpness": candidate.sharpness,
                        "selected_frame_index": chosen.frame.index,
                        "selected_sharpness": chosen.sharpness,
                    }
                )

            alternates = [
                candidate.frame
                for candidate in ranked
                if candidate.frame.index != chosen.frame.index and candidate.frame.index not in already_rejected
            ]
            selected.append(
                SelectedFrame(
                    frame=chosen.frame,
                    quality=FrameQuality(
                        sharpness=chosen.sharpness,
                        difference_score=chosen_difference,
                        accepted=True,
                        reason=chosen.reason if len(window) == 1 else f"{chosen.reason}_windowed",
                    ),
                    alternates=alternates,
                )
            )
            previous_selected = chosen.frame

        return selected

    def _resolve_blur_threshold(self, sharpness_scores: list[float]) -> float:
        if not sharpness_scores:
            return self.config.blur_threshold
        if not self.config.adaptive_blur_threshold:
            return self.config.blur_threshold

        percentile = float(np.clip(self.config.adaptive_blur_percentile, 0.0, 1.0))
        adaptive_threshold = float(np.quantile(np.asarray(sharpness_scores, dtype=np.float64), percentile))
        return min(self.config.blur_threshold, adaptive_threshold)

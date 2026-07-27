from __future__ import annotations

from typing import Protocol

from panoramator.domain.models import (
    CanvasModel,
    FeatureSet,
    Frame,
    MatchSet,
    PairGeometry,
)


class FeatureExtractor(Protocol):
    backend_name: str

    def extract(self, frame: Frame) -> FeatureSet:
        ...


class FeatureMatcher(Protocol):
    def match(self, left: FeatureSet, right: FeatureSet) -> MatchSet:
        ...


class GeometryEstimator(Protocol):
    def estimate(
        self,
        left_frame: Frame,
        right_frame: Frame,
        left_features: FeatureSet,
        right_features: FeatureSet,
        matches: MatchSet,
    ) -> PairGeometry:
        ...


class CanvasBuilder(Protocol):
    def build(self, frame_shapes: list[tuple[int, int]], homographies: list) -> CanvasModel:
        ...

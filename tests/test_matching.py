from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import FeatureSet
from panoramator.matching import matchers


def _feature_set(backend: str, descriptors: np.ndarray | None) -> FeatureSet:
    return FeatureSet(keypoints=[], descriptors=descriptors, backend=backend)


def test_match_returns_empty_result_when_descriptors_are_missing() -> None:
    adapter = matchers.BFMatcherAdapter(PanoramaConfig())

    result = adapter.match(_feature_set("orb", None), _feature_set("orb", np.ones((1, 3), dtype=np.uint8)))

    assert result.raw_count == 0
    assert result.good_matches == []
    assert result.confidence == 0.0


def test_match_uses_ratio_test_and_backend_norm(monkeypatch) -> None:
    seen: dict[str, int] = {}

    class _FakeMatcher:
        def knnMatch(self, left, right, k):
            return [
                [SimpleNamespace(distance=4.0), SimpleNamespace(distance=8.0)],
                [SimpleNamespace(distance=9.0), SimpleNamespace(distance=10.0)],
                [SimpleNamespace(distance=1.0)],
            ]

    def _fake_factory(norm_type, crossCheck=False):
        seen["norm_type"] = norm_type
        return _FakeMatcher()

    monkeypatch.setattr(matchers.cv2, "BFMatcher", _fake_factory)

    adapter = matchers.BFMatcherAdapter(PanoramaConfig(ratio_test=0.75))
    result = adapter.match(
        _feature_set("orb", np.ones((3, 3), dtype=np.uint8)),
        _feature_set("orb", np.ones((3, 3), dtype=np.uint8)),
    )

    assert seen["norm_type"] == matchers.cv2.NORM_HAMMING
    assert result.raw_count == 3
    assert len(result.good_matches) == 1
    assert result.confidence == 1 / 3


def test_match_uses_l2_for_non_orb_backends(monkeypatch) -> None:
    seen: dict[str, int] = {}

    class _FakeMatcher:
        def knnMatch(self, left, right, k):
            return []

    def _fake_factory(norm_type, crossCheck=False):
        seen["norm_type"] = norm_type
        return _FakeMatcher()

    monkeypatch.setattr(matchers.cv2, "BFMatcher", _fake_factory)

    adapter = matchers.BFMatcherAdapter(PanoramaConfig())
    adapter.match(
        _feature_set("sift", np.ones((1, 2), dtype=np.float32)),
        _feature_set("sift", np.ones((1, 2), dtype=np.float32)),
    )

    assert seen["norm_type"] == matchers.cv2.NORM_L2

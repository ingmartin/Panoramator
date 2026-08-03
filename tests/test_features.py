from __future__ import annotations

import numpy as np
import pytest

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import Frame
from panoramator.features import extractors


def test_create_feature_extractor_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="feature_backend must be one of"):
        extractors.create_feature_extractor(PanoramaConfig(feature_backend="orbish"))  # type: ignore[arg-type]


def test_orb_extractor_uses_feature_image_and_rescales_keypoints(monkeypatch) -> None:
    class _FakeORB:
        def detectAndCompute(self, gray, mask):
            keypoint = extractors.cv2.KeyPoint(x=1.0, y=2.0, size=4.0)
            return [keypoint], np.array([[1, 2, 3]], dtype=np.uint8)

    monkeypatch.setattr(extractors.cv2, "ORB_create", lambda nfeatures: _FakeORB())

    frame = Frame(
        index=0,
        timestamp_seconds=0.0,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        feature_image=np.zeros((4, 4, 3), dtype=np.uint8),
    )

    features = extractors.ORBFeatureExtractor(PanoramaConfig(max_features=123)).extract(frame)

    assert features.backend == "orb"
    assert features.descriptors is not None
    assert features.descriptors.shape == (1, 3)
    assert features.keypoints[0].pt == (2.0, 4.0)


def test_sift_extractor_requires_backend_support(monkeypatch) -> None:
    monkeypatch.delattr(extractors.cv2, "SIFT_create", raising=False)

    with pytest.raises(RuntimeError, match="SIFT is not available"):
        extractors.SIFTFeatureExtractor(PanoramaConfig())


def test_sift_extractor_returns_original_coordinates_when_shapes_match(monkeypatch) -> None:
    class _FakeSIFT:
        def detectAndCompute(self, gray, mask):
            keypoint = extractors.cv2.KeyPoint(x=3.0, y=5.0, size=2.0)
            return [keypoint], np.array([[5.0, 8.0]], dtype=np.float32)

    monkeypatch.setattr(extractors.cv2, "SIFT_create", lambda nfeatures: _FakeSIFT(), raising=False)
    frame = Frame(index=0, timestamp_seconds=0.0, image=np.zeros((6, 6, 3), dtype=np.uint8))

    features = extractors.SIFTFeatureExtractor(PanoramaConfig(max_features=77)).extract(frame)

    assert features.backend == "sift"
    assert features.keypoints[0].pt == (3.0, 5.0)


def test_rescale_keypoints_handles_empty_input() -> None:
    assert extractors._rescale_keypoints([], (4, 4, 3), (8, 8, 3)) == []

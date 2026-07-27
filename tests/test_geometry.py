import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.geometry.homography import HomographyEstimator


def test_validate_transform_rejects_large_scale_deviation() -> None:
    estimator = HomographyEstimator(PanoramaConfig(max_scale_deviation=0.1))
    homography = np.array([[1.3, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    valid, reason = estimator._validate_transform(homography)
    assert not valid
    assert reason == "scale_deviation"


def test_validate_transform_accepts_small_affine_transform() -> None:
    estimator = HomographyEstimator(PanoramaConfig())
    homography = np.array([[1.01, -0.03, 40.0], [0.03, 1.01, 5.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    valid, reason = estimator._validate_transform(homography)
    assert valid
    assert reason == "ok"

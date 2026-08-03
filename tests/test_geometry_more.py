from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import FeatureSet, Frame, MatchSet
from panoramator.geometry import homography as homography_module
from panoramator.geometry.homography import (
    HomographyEstimator,
    _affine_to_homography,
    accumulate_global_homographies,
)


def _frame(index: int) -> Frame:
    return Frame(index=index, timestamp_seconds=float(index), image=np.zeros((10, 10, 3), dtype=np.uint8))


def _feature_set(points: list[tuple[float, float]], backend: str = "orb") -> FeatureSet:
    keypoints = [SimpleNamespace(pt=point) for point in points]
    return FeatureSet(keypoints=keypoints, descriptors=np.ones((len(points), 2), dtype=np.float32), backend=backend)


def test_estimate_returns_not_enough_matches() -> None:
    estimator = HomographyEstimator(PanoramaConfig(min_match_count=3))
    matches = MatchSet(raw_count=2, good_matches=[SimpleNamespace(queryIdx=0, trainIdx=0)], confidence=0.5)

    result = estimator.estimate(_frame(0), _frame(1), _feature_set([(0, 0)]), _feature_set([(1, 1)]), matches)

    assert result.valid is False
    assert result.reason == "not_enough_matches"


def test_estimate_translation_model_returns_valid_geometry() -> None:
    estimator = HomographyEstimator(
        PanoramaConfig(
            motion_model="translation",
            min_match_count=2,
            min_inlier_count=2,
            max_reprojection_error=0.1,
        )
    )
    left = _feature_set([(0, 0), (1, 1)])
    right = _feature_set([(2, 3), (3, 4)])
    matches = MatchSet(
        raw_count=2,
        good_matches=[SimpleNamespace(queryIdx=0, trainIdx=0), SimpleNamespace(queryIdx=1, trainIdx=1)],
        confidence=1.0,
    )

    result = estimator.estimate(_frame(0), _frame(1), left, right, matches)

    assert result.valid is True
    assert result.reason == "ok"
    assert result.homography is not None
    assert np.allclose(result.homography, np.array([[1.0, 0.0, -2.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]]))


def test_estimate_rejects_pair_with_too_few_inliers(monkeypatch) -> None:
    estimator = HomographyEstimator(PanoramaConfig(min_match_count=4, min_inlier_count=3, min_inlier_ratio=0.5))
    monkeypatch.setattr(
        estimator,
        "_estimate_transform",
        lambda src, dst: (np.eye(3, dtype=np.float64), np.array([[1], [1], [0], [0]], dtype=np.uint8), "ok"),
    )
    matches = MatchSet(
        raw_count=4,
        good_matches=[SimpleNamespace(queryIdx=index, trainIdx=index) for index in range(4)],
        confidence=1.0,
    )
    features = _feature_set([(float(index), float(index)) for index in range(4)])

    result = estimator.estimate(_frame(0), _frame(1), features, features, matches)

    assert result.valid is False
    assert result.reason == "not_enough_inliers"


def test_estimate_rejects_large_reprojection_error(monkeypatch) -> None:
    estimator = HomographyEstimator(
        PanoramaConfig(min_match_count=1, min_inlier_count=1, max_reprojection_error=0.1)
    )
    monkeypatch.setattr(
        estimator,
        "_estimate_transform",
        lambda src, dst: (np.eye(3, dtype=np.float64), np.ones((1, 1), dtype=np.uint8), "ok"),
    )
    monkeypatch.setattr(estimator, "_compute_reprojection_error", lambda *args: 1.5)
    matches = MatchSet(raw_count=1, good_matches=[SimpleNamespace(queryIdx=0, trainIdx=0)], confidence=1.0)

    result = estimator.estimate(_frame(0), _frame(1), _feature_set([(0, 0)]), _feature_set([(1, 1)]), matches)

    assert result.valid is False
    assert result.reason == "reprojection_error"


def test_estimate_preserves_transform_failure_reason(monkeypatch) -> None:
    estimator = HomographyEstimator(PanoramaConfig(min_match_count=1))
    monkeypatch.setattr(estimator, "_estimate_transform", lambda src, dst: (None, None, "affine_failed"))
    matches = MatchSet(raw_count=1, good_matches=[SimpleNamespace(queryIdx=0, trainIdx=0)], confidence=1.0)

    result = estimator.estimate(_frame(0), _frame(1), _feature_set([(0, 0)]), _feature_set([(1, 1)]), matches)

    assert result.valid is False
    assert result.inliers == 0
    assert result.reprojection_error == float("inf")
    assert result.reason == "affine_failed"


def test_estimate_transform_reports_partial_affine_failure(monkeypatch) -> None:
    estimator = HomographyEstimator(PanoramaConfig(motion_model="partial_affine"))
    monkeypatch.setattr(homography_module.cv2, "estimateAffinePartial2D", lambda *args, **kwargs: (None, None))

    transform, mask, reason = estimator._estimate_transform(np.zeros((2, 1, 2), dtype=np.float32), np.zeros((2, 1, 2), dtype=np.float32))

    assert transform is None
    assert mask is None
    assert reason == "partial_affine_failed"


def test_estimate_transform_reports_affine_and_homography_failures(monkeypatch) -> None:
    points = np.zeros((4, 1, 2), dtype=np.float32)
    affine_estimator = HomographyEstimator(PanoramaConfig(motion_model="affine"))
    homography_estimator = HomographyEstimator(PanoramaConfig(motion_model="homography"))
    monkeypatch.setattr(homography_module.cv2, "estimateAffine2D", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(homography_module.cv2, "findHomography", lambda *args, **kwargs: (None, None))

    assert affine_estimator._estimate_transform(points, points) == (None, None, "affine_failed")
    assert homography_estimator._estimate_transform(points, points) == (None, None, "homography_failed")


def test_compute_reprojection_error_is_zero_for_exact_transform() -> None:
    source = np.array([[[0.0, 0.0]], [[1.0, 1.0]]], dtype=np.float32)
    destination = np.array([[[2.0, 3.0]], [[3.0, 4.0]]], dtype=np.float32)
    translation = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    error = HomographyEstimator._compute_reprojection_error(
        source, destination, translation, np.ones((2, 1), dtype=np.uint8)
    )

    assert error == 0.0


def test_validate_transform_rejects_rotation_and_perspective() -> None:
    estimator = HomographyEstimator(PanoramaConfig(motion_model="homography", max_rotation_degrees=5.0))
    rotated = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    perspective = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.01, 0.0, 1.0]], dtype=np.float64)

    assert estimator._validate_transform(rotated) == (False, "rotation_deviation")
    assert estimator._validate_transform(perspective) == (False, "perspective_deviation")


def test_validate_transform_rejects_homography_with_excessive_projected_frame_size() -> None:
    estimator = HomographyEstimator(PanoramaConfig(motion_model="homography", max_homography_corner_scale=2.0))
    near_singular = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.0009, 0.0, 1.0]], dtype=np.float64
    )

    assert estimator._validate_transform(near_singular, (1_000, 1_000)) == (False, "projected_frame_scale")


def test_compute_reprojection_error_is_infinite_without_inliers() -> None:
    error = HomographyEstimator._compute_reprojection_error(
        np.zeros((1, 1, 2), dtype=np.float32),
        np.zeros((1, 1, 2), dtype=np.float32),
        np.eye(3, dtype=np.float64),
        np.zeros((1, 1), dtype=np.uint8),
    )

    assert error == float("inf")


def test_accumulate_global_homographies_normalizes_each_step() -> None:
    first = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 2.0]], dtype=np.float64)
    second = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    result = accumulate_global_homographies([first, second])

    assert len(result) == 3
    assert np.allclose(result[0], np.eye(3))
    assert np.allclose(result[1], np.array([[0.5, 0.0, 1.0], [0.0, 0.5, 1.5], [0.0, 0.0, 1.0]]))
    assert np.allclose(result[2], result[1] @ second)


def test_affine_to_homography_appends_last_row() -> None:
    affine = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype=np.float64)

    result = _affine_to_homography(affine)

    assert np.allclose(result, np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]))

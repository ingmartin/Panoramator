from __future__ import annotations

import numpy as np
import pytest

from panoramator.camera.models import CameraParameters
from panoramator.canvas.builder import PanoramaCanvasBuilder
from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import CanvasModel, Frame
from panoramator.motion_analysis.analyzer import MotionAnalysis, MotionAnalyzer
from panoramator.projection.models import (
    CylindricalProjection,
    SphericalProjection,
    create_projection,
)
from panoramator.projection.preprocess import project_frame_for_geometry
from panoramator.strategy.resolver import resolve_strategy
from panoramator.warping.warper import FrameWarper


def test_config_validates_and_normalizes_capture_settings() -> None:
    config = PanoramaConfig(capture_mode="ROTATION", projection="CYLINDRICAL")
    assert (config.capture_mode, config.projection) == ("rotation", "cylindrical")


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"focal_length_px": float("nan")}, "focal_length_px must be > 0"),
        ({"horizontal_fov_degrees": float("inf")}, "horizontal_fov_degrees must be between 1 and 179"),
        ({"projection_center_x": float("nan")}, "projection_center_x must be finite"),
        ({"focal_length_px": 100.0, "horizontal_fov_degrees": 90.0}, "Set either focal_length_px"),
    ],
)
def test_config_rejects_non_finite_or_ambiguous_camera_parameters(settings: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PanoramaConfig(**settings)


def test_explicit_projection_overrides_auto_strategy() -> None:
    decision = resolve_strategy(PanoramaConfig(projection="spherical"), MotionAnalysis("rotation", 0.8, "test"))
    assert (decision.capture_mode, decision.projection) == ("rotation", "spherical")
    assert "manual_projection" in decision.reason


def test_ambiguous_analysis_uses_compatible_linear_planar_fallback() -> None:
    decision = resolve_strategy(PanoramaConfig(), MotionAnalysis.fallback())
    assert (decision.capture_mode, decision.projection, decision.confidence) == ("linear", "planar", 0.0)


def test_cylindrical_projection_round_trips_points() -> None:
    projection = CylindricalProjection(CameraParameters(100.0, 50.0, 30.0))
    points = np.array([[0.0, 0.0], [50.0, 30.0], [95.0, 55.0]])
    assert np.allclose(projection.unproject_points(projection.project_points(points)), points)


def test_spherical_projection_round_trips_points_and_factory_rejects_unknown_name() -> None:
    camera = CameraParameters(100.0, 50.0, 30.0)
    projection = SphericalProjection(camera)
    points = np.array([[0.0, 0.0], [50.0, 30.0], [95.0, 55.0]])

    assert np.allclose(projection.unproject_points(projection.project_points(points)), points)
    with pytest.raises(ValueError, match="Unsupported projection"):
        create_projection("conical", camera)


def test_camera_uses_fov_and_explicit_principal_point() -> None:
    camera = CameraParameters.from_config(
        PanoramaConfig(horizontal_fov_degrees=90.0, projection_center_x=3.0, projection_center_y=4.0), (20, 20)
    )

    assert camera.focal_length_px == pytest.approx(10.0)
    assert (camera.center_x, camera.center_y) == (3.0, 4.0)


def test_camera_prefers_explicit_focal_length() -> None:
    camera = CameraParameters.from_config(PanoramaConfig(focal_length_px=123.0), (20, 20))

    assert camera.focal_length_px == 123.0


def test_camera_rejects_principal_point_outside_source_frame() -> None:
    config = PanoramaConfig(projection_center_x=21.0)

    with pytest.raises(ValueError, match="projection center must lie within the source frame"):
        CameraParameters.from_config(config, (20, 20))


def test_curved_canvas_samples_edges_not_only_corners() -> None:
    config = PanoramaConfig(projection_contour_samples=16)
    projection = CylindricalProjection(CameraParameters(50.0, 50.0, 50.0))

    canvas = PanoramaCanvasBuilder(config).build([(100, 100)], [np.eye(3)], projection)

    assert canvas.height == 100
    assert canvas.projection is projection


def test_curved_canvas_applies_global_transform_after_local_projection() -> None:
    config = PanoramaConfig(projection_contour_samples=8)
    projection = CylindricalProjection(CameraParameters(50.0, 50.0, 50.0))
    translation = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    canvas = PanoramaCanvasBuilder(config).build([(100, 100), (100, 100)], [np.eye(3), translation], projection)

    # The second projected frame must retain the complete 20 px surface shift.
    assert canvas.width >= 98


def test_feature_projection_changes_image_but_preserves_original_frame() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :4] = 255
    frame = Frame(0, 0.0, image)
    projection = CylindricalProjection(CameraParameters(30.0, 30.0, 20.0))

    prepared = project_frame_for_geometry(frame, projection)

    assert prepared is not frame
    assert prepared.image.shape == frame.image.shape
    assert np.array_equal(frame.image, image)
    assert not np.array_equal(prepared.image, image)


def test_curved_warper_uses_single_remap_and_nearest_mask() -> None:
    frame = Frame(0, 0.0, np.full((20, 20, 3), 127, dtype=np.uint8))
    canvas = CanvasModel(20, 20, np.eye(3), [np.eye(3)], create_projection("cylindrical", CameraParameters(30.0, 10.0, 10.0)))
    image, mask = FrameWarper().warp(frame, np.eye(3), canvas)
    assert image.shape == (20, 20, 3)
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask[10, 10] == 255


def test_curved_warper_marks_projective_horizon_as_invalid_instead_of_remapping_it() -> None:
    frame = Frame(0, 0.0, np.full((20, 20, 3), 127, dtype=np.uint8))
    canvas = CanvasModel(20, 20, np.eye(3), [], create_projection("cylindrical", CameraParameters(30.0, 10.0, 10.0)))
    horizon_homography = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.0, 1.0]])

    _, mask = FrameWarper().warp(frame, horizon_homography, canvas)

    assert np.all(mask[:, 10] == 0)


def test_curved_warper_rejects_periodic_backside_of_cylindrical_inverse_map() -> None:
    frame = Frame(0, 0.0, np.full((20, 100, 3), 127, dtype=np.uint8))
    projection = create_projection("cylindrical", CameraParameters(50.0, 50.0, 10.0))
    canvas = CanvasModel(500, 20, np.eye(3), [], projection)
    # At x=407 the inverse local angle is close to pi for this translated
    # frame. ``tan`` maps it near the source centre although it is behind the
    # camera and must remain invalid.
    transform = np.array([[1.0, 0.0, 200.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    _, mask = FrameWarper().warp(frame, transform, canvas)

    assert mask[10, 407] == 0


def test_motion_analyzer_is_conservative_without_chain_evidence() -> None:
    assert MotionAnalyzer().analyze([], []) == MotionAnalysis.fallback()


def test_motion_analyzer_falls_back_for_malformed_or_nonfinite_geometry() -> None:
    metrics = [{"valid": True, "reprojection_error": 1.0}]

    assert MotionAnalyzer().analyze([np.eye(2)], metrics) == MotionAnalysis.fallback()
    assert MotionAnalyzer().analyze([np.full((3, 3), np.nan)], metrics) == MotionAnalysis.fallback()
    assert MotionAnalyzer().analyze([np.eye(3)], [{"valid": True, "reprojection_error": float("nan")}]) == MotionAnalysis.fallback()


def test_motion_analyzer_classifies_stable_rotation_and_orbit_risk() -> None:
    rotation = np.array([[0.999, -0.05, 5.0], [0.05, 0.999, 1.0], [0.0, 0.0, 1.0]])
    valid_metrics = [{"valid": True, "reprojection_error": 1.0}, {"valid": True, "reprojection_error": 1.1}]

    rotation_analysis = MotionAnalyzer().analyze([rotation, rotation], valid_metrics)
    orbit_analysis = MotionAnalyzer().analyze([np.diag([1.0, 1.0, 1.0]), np.diag([1.3, 1.3, 1.0])], valid_metrics)

    assert rotation_analysis.capture_mode == "rotation"
    assert rotation_analysis.confidence > 0.5
    assert orbit_analysis.capture_mode == "orbit"


def test_motion_analyzer_requires_a_better_cylindrical_preview_for_auto_rotation() -> None:
    planar = [np.array([[1.0, 0.0, 8.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])] * 2
    cylindrical = [np.array([[1.0, 0.0, 8.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])] * 2
    planar_metrics = [
        {"valid": True, "reprojection_error": 3.0, "inliers": 16, "good_matches": 20},
        {"valid": True, "reprojection_error": 3.0, "inliers": 16, "good_matches": 20},
    ]
    cylindrical_metrics = [
        {"valid": True, "reprojection_error": 1.0, "inliers": 17, "good_matches": 20},
        {"valid": True, "reprojection_error": 1.0, "inliers": 17, "good_matches": 20},
    ]

    analysis = MotionAnalyzer().analyze(planar, planar_metrics, (cylindrical, cylindrical_metrics))

    assert analysis.capture_mode == "rotation"
    assert analysis.reason == "cylindrical_preview_explains_rotation_better"
    assert analysis.measurements["cylindrical_residual_gain"] == pytest.approx(2 / 3)


def test_motion_analyzer_keeps_auto_linear_when_cylindrical_preview_is_not_better() -> None:
    transforms = [np.array([[1.0, 0.0, 8.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])] * 2
    metrics = [{"valid": True, "reprojection_error": 1.0, "inliers": 16, "good_matches": 20}] * 2

    analysis = MotionAnalyzer().analyze(transforms, metrics, (transforms, metrics))

    assert analysis.capture_mode == "linear"

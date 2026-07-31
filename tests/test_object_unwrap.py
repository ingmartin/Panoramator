from __future__ import annotations

import json

import cv2
import numpy as np

from panoramator.domain.models import Frame
from panoramator.object_unwrap.analyzer import AnalyzedFrame
from panoramator.object_unwrap.coverage import least_covered_seam
from panoramator.object_unwrap.cylinder.builder import CylinderUnwrapBuilder
from panoramator.object_unwrap.cylinder.pose import solve_monotonic_trajectory
from panoramator.object_unwrap.diagnostics import write_artifacts
from panoramator.object_unwrap.image_pose_graph import build_image_pose_graph
from panoramator.object_unwrap.models import (
    SurfaceKind,
    UnwrapConfig,
    UnwrapDiagnostics,
    UnwrapStatus,
)


def _frame(index: int, shift: int) -> AnalyzedFrame:
    image = np.zeros((100, 80, 3), np.uint8)
    image[10:90, 15:65] = (20 + shift, 120, 220)
    cv2.line(image, (25 + shift % 20, 10), (25 + shift % 20, 89), (255, 255, 255), 2)
    mask = np.zeros((100, 80), np.uint8)
    mask[10:90, 15:65] = 255
    return AnalyzedFrame(Frame(index, float(index), image), mask, mask.copy(), 100.0, (15, 10, 50, 80))


def test_cylinder_builder_creates_alpha_coverage_and_low_coverage_seam() -> None:
    image, coverage, model, measurements, artifacts = CylinderUnwrapBuilder().build(
        [_frame(0, 0), _frame(1, 8), _frame(2, 16)],
        UnwrapConfig(surface_kind=SurfaceKind.CYLINDRICAL, output_width=200, output_height=80, rectification_smoothing_window=7),
    )

    assert image.shape[0] == 80
    assert image.shape[1] <= 200
    assert coverage.shape == image.shape[:2]
    assert 0 < measurements["coverage_fraction"] <= 1
    assert model.kind is SurfaceKind.CYLINDRICAL
    assert least_covered_seam(coverage) == 0
    assert artifacts["source"].shape == coverage.shape
    assert artifacts["reprojection_error"].shape == coverage.shape
    assert measurements["rendering"] == "feature_mosaic_then_global_rectification"
    assert artifacts["angular_mosaic"].shape[0] == 80


def test_artifacts_keep_png_in_diagnostic_file_list(tmp_path) -> None:
    output = tmp_path / "unwrap.png"
    diagnostics = UnwrapDiagnostics(UnwrapStatus.OK, "ok", "", SurfaceKind.CYLINDRICAL, output_files=[str(output)])
    files = write_artifacts(output, UnwrapConfig(), diagnostics, np.full((4, 4), 255, np.uint8))

    report = json.loads((tmp_path / "unwrap_debug" / "run.json").read_text())
    assert str(output) in files
    assert report["output_files"] == files


def test_artifacts_include_source_and_reprojection_error_maps(tmp_path) -> None:
    output = tmp_path / "unwrap.png"
    diagnostics = UnwrapDiagnostics(UnwrapStatus.OK, "ok", "", SurfaceKind.CYLINDRICAL)
    files = write_artifacts(
        output,
        UnwrapConfig(),
        diagnostics,
        np.full((4, 4), 255, np.uint8),
        {"source": np.ones((4, 4), np.uint8), "reprojection_error": np.zeros((4, 4), np.uint8)},
    )

    assert str(tmp_path / "unwrap_debug" / "source.png") in files
    assert str(tmp_path / "unwrap_debug" / "reprojection_error.png") in files


def test_monotonic_trajectory_rejects_reversed_outlier_without_reordering_frames() -> None:
    trajectory = solve_monotonic_trajectory([(0.10, 0.9), (0.11, 0.9), (-0.8, 0.95), (0.09, 0.9)])

    assert len(trajectory.angles) == 5
    assert all(right >= left for left, right in zip(trajectory.angles, trajectory.angles[1:]))
    assert trajectory.accepted_pairs == 3
    assert trajectory.repeated_observation is False
    assert trajectory.steps[2] == 0.0
    assert trajectory.rejection_reasons[2] == "reversed_motion"


def test_unwrap_config_limits_source_map_frame_ids() -> None:
    with np.testing.assert_raises_regex(ValueError, "65535"):
        UnwrapConfig(max_frames=65_536).validate()


def test_image_pose_graph_marks_surface_supported_translation() -> None:
    base = np.zeros((100, 120, 3), np.uint8)
    for x in range(18, 100, 16):
        for y in range(18, 86, 16):
            cv2.circle(base, (x, y), 4, (40 + x, 120, 220 - y), -1)
    shifted = np.roll(base, 6, axis=1)
    mask = np.full((100, 120), 255, np.uint8)
    frames = [
        AnalyzedFrame(Frame(0, 0.0, base), mask, mask.copy(), 100.0, (0, 0, 120, 100)),
        AnalyzedFrame(Frame(1, 1.0, shifted), mask, mask.copy(), 100.0, (0, 0, 120, 100)),
    ]
    graph = build_image_pose_graph(frames)

    assert len(graph.edges) == 1
    assert graph.edges[0]["reason"] == "ok"
    assert graph.edges[0]["surface_inliers"] >= 8

from __future__ import annotations

import cv2
import numpy as np
import pytest

from panoramator.domain.models import Frame
from panoramator.object_unwrap import observed_surface
from panoramator.object_unwrap.analyzer import Analysis, AnalyzedFrame
from panoramator.object_unwrap.models import SurfaceKind, SurfaceModel, UnwrapConfig
from panoramator.object_unwrap.observed_surface import ObservedSurfaceBuilder
from panoramator.object_unwrap.service import _SurfaceBuild


def _branch_frame(index: int, x: int, *, nuisance_heavy: bool = False) -> AnalyzedFrame:
    image = np.zeros((48, 48, 3), np.uint8)
    image[:, :] = (20 + index, 40 + index, 60 + index)
    cv2.line(image, (x + 4, 6), (x + 4, 41), (255, 255, 255), 2)
    cv2.line(image, (x + 9, 6), (x + 9, 41), (0, 220, 120), 2)
    geometry = np.zeros((48, 48), np.uint8)
    geometry[6:42, x : x + 18] = 255
    publish = geometry.copy()
    core = np.zeros_like(geometry)
    core[6:42, x + 2 : x + 16] = 255
    nuisance = np.zeros_like(geometry)
    if nuisance_heavy:
        nuisance[6:42, x + 4 : x + 14] = 255
    return AnalyzedFrame(
        Frame(index, float(index), image),
        geometry,
        publish,
        100.0,
        (x, 6, 18, 36),
        core_mask=core,
        nuisance_mask=nuisance,
    )


def _baseline_build() -> _SurfaceBuild:
    return _SurfaceBuild(
        image=np.full((64, 160, 3), 90, np.uint8),
        coverage=np.full((64, 160), 255, np.uint8),
        model=SurfaceModel(SurfaceKind.CYLINDRICAL),
        measurements={
            "atlas_width": 160,
            "pixels_per_radian": 24.0,
            "angular_steps": [0.18] * 11,
            "quality_gate_passed": 1,
            "rectification_applied": 1,
            "observed_coverage_fraction": 1.0,
            "surface_coverage_fraction": 1.0,
        },
        artifacts={"mosaic": np.full((64, 160, 3), 80, np.uint8)},
        fallback_used=False,
    )


def test_observed_surface_builder_falls_back_to_baseline_for_insufficient_frames() -> None:
    frames = [_branch_frame(index, 10 + index) for index in range(5)]
    analysis = Analysis(frames, SurfaceKind.CYLINDRICAL, measurements={"analysis_score": 5})

    result = ObservedSurfaceBuilder().build(analysis, UnwrapConfig(output_height=64), _baseline_build())

    assert np.array_equal(result.image, _baseline_build().image)
    assert np.array_equal(result.coverage, _baseline_build().coverage)
    assert result.measurements["observed_branch_applied"] == 0
    assert result.measurements["observed_branch_abort_reason"] == "insufficient_analyzed_frames"
    assert result.measurements["observed_branch_input_frame_count"] == 5
    assert result.measurements["observed_branch_axis_valid_frame_count"] == 0
    assert "baseline_surface_coverage_fraction" in result.measurements


def test_observed_surface_builder_builds_coverage_first_strip_with_required_artifacts() -> None:
    frames = [_branch_frame(index, 8 + index, nuisance_heavy=index == 3) for index in range(12)]
    analysis = Analysis(frames, SurfaceKind.CYLINDRICAL, measurements={"analysis_score": 9})

    result = ObservedSurfaceBuilder().build(analysis, UnwrapConfig(output_height=64), _baseline_build())

    assert result.measurements["observed_branch_applied"] == 1
    assert result.measurements["observed_branch_axis_valid_frame_count"] == 12
    assert result.measurements["observed_branch_selected_frame_count"] >= 12
    assert result.measurements["observed_branch_used_angular_steps"] == 1
    assert result.measurements["observed_branch_used_phase_correlation"] == 0
    assert result.measurements["observed_branch_canvas_width"] == 160
    assert result.measurements["observed_branch_mask_fallback_count"] == 1
    assert float(result.measurements["observed_branch_coverage_fraction"]) > 0.0
    assert float(result.measurements["observed_branch_largest_component_fraction"]) >= 0.82
    assert result.artifacts["observed_branch_mosaic"] is not None
    assert result.artifacts["observed_branch_coverage"] is not None
    assert result.artifacts["observed_branch_source"] is not None


def test_observed_surface_builder_falls_back_when_visual_quality_is_worse_than_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [_branch_frame(index, 8 + index) for index in range(12)]
    analysis = Analysis(frames, SurfaceKind.CYLINDRICAL, measurements={"analysis_score": 9})
    baseline = _baseline_build()
    baseline.image[:, ::8] = 200

    def fake_compose_mosaic(*args, **kwargs):
        image = np.zeros((64, 160, 3), np.uint8)
        image[:, ::2] = 255
        coverage = np.full((64, 160), 255, np.uint8)
        source = np.ones((64, 160), np.uint16)
        return image, coverage, source, 0.1, 0.1

    def fake_gradient_energy(image: np.ndarray, coverage: np.ndarray) -> float:
        return 80.0 if image is baseline.image else 160.0

    monkeypatch.setattr(observed_surface, "_compose_mosaic", fake_compose_mosaic)
    monkeypatch.setattr(observed_surface, "_gradient_energy", fake_gradient_energy)

    result = ObservedSurfaceBuilder().build(analysis, UnwrapConfig(output_height=64), baseline)

    assert np.array_equal(result.image, baseline.image)
    assert result.measurements["observed_branch_applied"] == 0
    assert result.measurements["observed_branch_abort_reason"] == "branch_visual_quality_rejected"
    assert result.measurements["observed_branch_visual_quality_passed"] == 0
    assert result.measurements["observed_branch_gradient_energy_ratio"] == 2.0

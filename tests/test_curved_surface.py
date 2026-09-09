from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from panoramator.domain.models import Frame
from panoramator.object_unwrap.analyzer import Analysis, AnalyzedFrame
from panoramator.object_unwrap.curved.builder import CurvedSurfaceBuilder, _bounded_fill
from panoramator.object_unwrap.models import (
    SurfaceKind,
    SurfaceModel,
    SurfaceOutputMode,
    UnwrapConfig,
)
from panoramator.object_unwrap.service import ObjectUnwrapper, _SurfaceBuild


def _frame(index: int) -> AnalyzedFrame:
    image = np.zeros((64, 80, 3), np.uint8)
    image[8:56, 12:68] = (40, 100, 160)
    cv2.rectangle(image, (18 + index, 16), (46 + index, 45), (220, 240, 255), 2)
    mask = np.zeros((64, 80), np.uint8)
    mask[8:56, 12:68] = 255
    return AnalyzedFrame(Frame(index, float(index), image), mask, mask.copy(), 100.0, (12, 8, 56, 48))


def _baseline() -> _SurfaceBuild:
    return _SurfaceBuild(
        np.zeros((64, 128, 3), np.uint8),
        np.zeros((64, 128), np.uint8),
        SurfaceModel(SurfaceKind.CURVED),
        {"surface_coverage_fraction": 0.0},
        {},
        False,
    )


def _graph(frame_count: int) -> SimpleNamespace:
    edges = []
    for index in range(frame_count - 1):
        edges.append(
            {
                "left_frame": index,
                "right_frame": index + 1,
                "hop": 1,
                "good_matches": 20,
                "surface_inliers": 12,
                "reprojection_error": 1.0,
                "reason": "ok",
                "a00": 1.0,
                "a01": 0.0,
                "a02": 4.0,
                "a10": 0.0,
                "a11": 1.0,
                "a12": 0.0,
            }
        )
    return SimpleNamespace(edges=edges)


def test_bounded_fill_only_fills_small_internal_holes() -> None:
    image = np.full((32, 48, 3), 90, np.uint8)
    coverage = np.full((32, 48), 255, np.uint8)
    coverage[14:16, 22:24] = 0
    owner = np.ones_like(coverage, np.uint16)

    _, filled, filled_owner, synthetic = _bounded_fill(image, coverage, owner)

    assert np.all(filled[14:16, 22:24] > 0)
    assert np.all(filled_owner[14:16, 22:24] == 65535)
    assert np.all(synthetic[14:16, 22:24] > 0)


def test_curved_builder_uses_local_atlas_and_writes_provenance(monkeypatch) -> None:
    from panoramator.object_unwrap.curved import builder as module

    frames = [_frame(index) for index in range(4)]
    mosaic = np.full((64, 96, 3), 120, np.uint8)
    coverage = np.zeros((64, 96), np.uint8)
    coverage[8:56, 8:88] = 255
    source = np.ones_like(coverage, np.uint16)
    error = np.zeros_like(coverage, np.uint8)
    monkeypatch.setattr(module, "build_image_pose_graph", lambda frames: _graph(len(frames)))
    monkeypatch.setattr(module, "build_planar_mosaic", lambda *args, **kwargs: (mosaic, coverage, source, error))

    result = CurvedSurfaceBuilder().build(
        Analysis(frames, SurfaceKind.CURVED),
        UnwrapConfig(output_height=64),
        _baseline(),
    )

    assert result.model.kind is SurfaceKind.CURVED
    assert result.measurements["surface_builder"] == "curved_local_atlas"
    assert result.measurements["surface_inlier_fraction"] == 0.6
    assert "curved_surface_synthetic" in result.artifacts
    assert "curved_surface_unknown" in result.artifacts
    assert "curved_surface_transforms" in result.artifacts


def test_observed_curved_route_does_not_call_cylinder_builder(monkeypatch) -> None:
    from panoramator.object_unwrap import service

    frames = [_frame(index) for index in range(4)]
    analysis = Analysis(frames, SurfaceKind.CURVED)
    baseline = _baseline()
    called = {"curved": False, "cylinder": False}

    def curved_build(self, current_analysis, config, current_baseline):
        called["curved"] = True
        return SimpleNamespace(
            image=current_baseline.image,
            coverage=current_baseline.coverage,
            model=SurfaceModel(SurfaceKind.CURVED),
            measurements={"curved_surface_quality_gate_passed": 1},
            artifacts={},
        )

    def cylinder_build(*args, **kwargs):
        called["cylinder"] = True
        raise AssertionError("cylindrical builder must not run for observed curved output")

    monkeypatch.setattr(service.CurvedSurfaceBuilder, "build", curved_build)
    monkeypatch.setattr(service.CylinderUnwrapBuilder, "build", cylinder_build)
    unwrapper = ObjectUnwrapper(UnwrapConfig(surface_output_mode=SurfaceOutputMode.OBSERVED_SURFACE))
    unwrapper._build_observed_surface(analysis, baseline)

    assert called == {"curved": True, "cylinder": False}

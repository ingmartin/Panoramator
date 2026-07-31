from __future__ import annotations

import cv2
import numpy as np
import pytest

from panoramator.cli.main import unwrap_command
from panoramator.domain.models import Frame
from panoramator.object_unwrap import analyzer as analyzer_module
from panoramator.object_unwrap.analyzer import Analysis, AnalyzedFrame, VideoAnalyzer
from panoramator.object_unwrap.curved.builder import CurvedSurfaceFallbackBuilder
from panoramator.object_unwrap.curved.reconstruction import reconstruct_from_silhouettes
from panoramator.object_unwrap.curved.template import CurvedSurfaceTemplate
from panoramator.object_unwrap.curved.uv import unwrap_mesh
from panoramator.object_unwrap.cylinder.mapper import (
    central_band,
    horizontal_shift,
    normalized_wall,
)
from panoramator.object_unwrap.diagnostics import write_artifacts
from panoramator.object_unwrap.models import (
    SurfaceKind,
    SurfaceModel,
    UnwrapConfig,
    UnwrapDiagnostics,
    UnwrapResult,
    UnwrapStatus,
)
from panoramator.object_unwrap.pose import optimize_rotation_angles
from panoramator.object_unwrap.segmentation import (
    masked_sharpness,
    object_mask,
    stable_surface_bbox,
)
from panoramator.object_unwrap.service import ObjectUnwrapper


def _analyzed_frame(index: int = 0) -> AnalyzedFrame:
    image = np.full((32, 24, 3), 100, np.uint8)
    mask = np.full((32, 24), 255, np.uint8)
    return AnalyzedFrame(Frame(index, float(index), image), mask, 100.0, (0, 0, 24, 32))


def test_segmentation_helpers_return_foreground_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((40, 50, 3), np.uint8)

    def fake_grabcut(image, labels, rectangle, background, foreground, iterations, mode):
        labels[8:34, 12:39] = cv2.GC_FGD

    monkeypatch.setattr(cv2, "grabCut", fake_grabcut)
    mask = object_mask(image, min_area_ratio=0.1)

    assert mask is not None
    assert stable_surface_bbox(mask) == (12, 8, 27, 26)
    assert masked_sharpness(image, mask) == 0.0
    assert object_mask(np.zeros((7, 7, 3), np.uint8)) is None


def test_analyzer_reports_blur_and_auto_selects_cylindrical(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = Frame(0, 0.0, np.zeros((20, 20, 3), np.uint8))
    monkeypatch.setattr(analyzer_module, "object_mask", lambda image, minimum: np.ones((20, 20), np.uint8))
    monkeypatch.setattr(analyzer_module, "stable_surface_bbox", lambda mask: (3, 2, 10, 16))
    monkeypatch.setattr(analyzer_module, "masked_sharpness", lambda image, mask: 10.0)

    blurred = VideoAnalyzer().analyze([frame, frame], UnwrapConfig(blur_threshold=20.0))
    assert blurred.status is UnwrapStatus.EXCESSIVE_MOTION_BLUR

    monkeypatch.setattr(analyzer_module, "masked_sharpness", lambda image, mask: 30.0)
    selected = VideoAnalyzer().analyze([frame, frame], UnwrapConfig(surface_kind=SurfaceKind.AUTO, blur_threshold=20.0))
    assert selected.status is None
    assert selected.kind is SurfaceKind.CYLINDRICAL


def test_surface_utility_functions_cover_empty_and_valid_geometry() -> None:
    template = CurvedSurfaceTemplate(width_to_height=2.0)
    assert template.compatibility([]) == 0.0
    assert template.compatibility([np.pad(np.ones((4, 8), np.uint8), 2)]) == 1.0
    reconstruction = reconstruct_from_silhouettes([np.ones((5, 5), np.uint8)] * 8)
    assert reconstruction.vertices is None and reconstruction.faces is None
    assert 0 < reconstruction.confidence < 1

    vertices = np.array([[1, 0, 0], [0, 2, 1], [-1, 4, 0]], dtype=np.float32)
    uv = unwrap_mesh(vertices, np.empty((0, 3), dtype=np.int32))
    assert uv.shape == (3, 2)
    assert np.all((uv >= 0) & (uv <= 1))
    with pytest.raises(ValueError, match="shape"):
        unwrap_mesh(np.empty((2, 2)), np.empty((0, 3), dtype=np.int32))


def test_mapper_normalizes_band_and_recovers_known_translation() -> None:
    rng = np.random.default_rng(7)
    image = np.zeros((20, 30, 3), np.uint8)
    image[:, 8:22] = rng.integers(0, 256, size=(20, 14, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), np.uint8)
    mask[:, 8:22] = 255
    wall, wall_mask = normalized_wall(image, mask, (8, 0, 14, 20), 40)
    band, band_mask = central_band(wall, wall_mask, 0.5)
    shifted = np.roll(band, 3, axis=1)

    shift, response = horizontal_shift(band, shifted)
    assert wall.shape[0] == 40 and wall_mask.shape == wall.shape[:2]
    assert band.shape[1] >= 8 and band_mask.shape == band.shape[:2]
    assert shift == pytest.approx(3.0 * 256 / band.shape[1], abs=1.0)
    assert response > 0.9


def test_pose_estimate_and_curved_fallback_are_explicit_about_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    estimate = optimize_rotation_angles([10.0, -5.0], 100.0)
    assert estimate.angles_degrees == [0.0, 18.0, 9.0]
    assert estimate.confidence == pytest.approx(0.25)
    assert optimize_rotation_angles([10.0], 0).confidence == 0.0

    from panoramator.object_unwrap.curved import builder as curved_builder_module

    monkeypatch.setattr(
        curved_builder_module.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.uint8), SurfaceModel(SurfaceKind.CYLINDRICAL, confidence=0.8), {}, {}
        ),
    )
    _, _, model, measurements, _ = CurvedSurfaceFallbackBuilder().build([_analyzed_frame()], UnwrapConfig())
    assert model.kind is SurfaceKind.CURVED
    assert model.confidence == pytest.approx(0.4)
    assert measurements["fallback"] == "dominant_side_band"


def test_unwrapper_writes_validated_result_with_synthetic_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from panoramator.object_unwrap import service

    class FakeSource:
        def __init__(self, *args) -> None:
            self.closed = False

        def open(self) -> None:
            return None

        def iter_frames(self):
            return [Frame(0, 0.0, np.zeros((8, 8, 3), np.uint8)), Frame(1, 1.0, np.zeros((8, 8, 3), np.uint8))]

        def close(self) -> None:
            self.closed = True

    analyzed = [_analyzed_frame(0), _analyzed_frame(1)]
    monkeypatch.setattr(service, "OpenCVVideoSource", FakeSource)
    monkeypatch.setattr(service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL))
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((6, 7, 3), 120, np.uint8),
            np.full((6, 7), 255, np.uint8),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {"surface_coverage_fraction": 1.0, "pose_residual_radians": 0.01, "accepted_pose_pairs": 2},
            {},
        ),
    )

    output = tmp_path / "surface.png"
    result = ObjectUnwrapper(UnwrapConfig(min_accepted_pose_pair_fraction=1.0)).unwrap_video("input.mp4", output)

    assert result.diagnostics.status is UnwrapStatus.OK
    assert result.output_path == output
    assert cv2.imread(str(output), cv2.IMREAD_UNCHANGED).shape == (6, 7, 4)
    assert (tmp_path / "surface_diagnostics.json").exists()


def test_unwrapper_returns_failure_diagnostics_without_writing_an_image(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from panoramator.object_unwrap import service

    monkeypatch.setattr(service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis([], SurfaceKind.AUTO, UnwrapStatus.OBJECT_NOT_DETECTED, "missing", "retry"))
    monkeypatch.setattr(service.OpenCVVideoSource, "open", lambda self: None)
    monkeypatch.setattr(service.OpenCVVideoSource, "iter_frames", lambda self: [])
    monkeypatch.setattr(service.OpenCVVideoSource, "close", lambda self: None)

    output = tmp_path / "surface.png"
    result = ObjectUnwrapper().unwrap_video("input.mp4", output)

    assert result.image is None
    assert result.diagnostics.status is UnwrapStatus.OBJECT_NOT_DETECTED
    assert not output.exists()
    assert (tmp_path / "surface_diagnostics.json").exists()


def test_artifact_writer_raises_when_coverage_cannot_be_saved(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from panoramator.object_unwrap.models import UnwrapDiagnostics

    monkeypatch.setattr(cv2, "imwrite", lambda path, image: False)
    diagnostics = UnwrapDiagnostics(UnwrapStatus.OK, "ok", "", SurfaceKind.CYLINDRICAL)
    with pytest.raises(RuntimeError, match="coverage"):
        write_artifacts(tmp_path / "surface.png", diagnostics, np.ones((2, 2), np.uint8))


def test_unwrap_cli_fails_when_partial_result_has_no_output(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from argparse import Namespace

    from panoramator.cli import main as cli_module

    missing_output = UnwrapResult(
        None,
        None,
        None,
        UnwrapDiagnostics(UnwrapStatus.PARTIAL_SURFACE, "partial", "retry", SurfaceKind.CYLINDRICAL),
    )
    monkeypatch.setattr(cli_module.ObjectUnwrapper, "unwrap_video", lambda self, video, output: missing_output)

    exit_code = unwrap_command(
        Namespace(
            surface_kind="auto", allow_partial=False, sampling_step=12, max_frames=48, min_coverage=0.9,
            output_width=1536, output_height=512, no_global_pose_optimization=False, video_path="in.mp4", output_path="out.png",
        )
    )

    assert exit_code == 2
    assert "Status: partial_surface" in capsys.readouterr().out

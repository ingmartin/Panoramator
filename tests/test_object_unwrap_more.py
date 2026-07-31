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
from panoramator.object_unwrap.cylinder.builder import CylinderUnwrapBuilder
from panoramator.object_unwrap.cylinder.pose import CylinderTrajectory
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
from panoramator.object_unwrap.planar_mosaic import build_planar_mosaic
from panoramator.object_unwrap.pose import optimize_rotation_angles
from panoramator.object_unwrap.rectification import estimate_strip, evaluate_mosaic_quality, rectify_mosaic
from panoramator.object_unwrap.segmentation import (
    masked_sharpness,
    object_mask,
    publish_surface_mask,
    stable_surface_bbox,
)
from panoramator.object_unwrap.service import ObjectUnwrapper


def _analyzed_frame(index: int = 0) -> AnalyzedFrame:
    image = np.full((32, 24, 3), 100, np.uint8)
    mask = np.full((32, 24), 255, np.uint8)
    return AnalyzedFrame(Frame(index, float(index), image), mask, mask.copy(), 100.0, (0, 0, 24, 32))


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


def test_publish_surface_mask_trims_upper_spikes_from_observed_band() -> None:
    mask = np.zeros((48, 32), np.uint8)
    mask[12:40, 6:26] = 255
    mask[3:12, 14:16] = 255

    publish = publish_surface_mask(mask, (6, 12, 20, 28))

    assert np.count_nonzero(publish[3:11, 14:16]) == 0
    assert np.all(publish[16:36, 8:24] == 255)


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


def test_analyzer_temporal_decimation_rejects_near_duplicate_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    base = np.zeros((24, 24, 3), np.uint8)
    base[:, 4:20] = (80, 140, 220)
    shifted = base.copy()
    shifted[:, 8:24] = (30, 200, 90)
    frames = [
        Frame(0, 0.0, base.copy()),
        Frame(1, 1.0, base.copy()),
        Frame(2, 2.0, shifted.copy()),
    ]
    monkeypatch.setattr(analyzer_module, "object_mask", lambda image, minimum: np.ones((24, 24), np.uint8) * 255)
    monkeypatch.setattr(analyzer_module, "stable_surface_bbox", lambda mask: (2, 2, 20, 20))
    monkeypatch.setattr(analyzer_module, "publish_surface_mask", lambda mask, bbox: mask.copy())
    monkeypatch.setattr(analyzer_module, "masked_sharpness", lambda image, mask: 50.0)

    analysis = VideoAnalyzer().analyze(frames, UnwrapConfig())

    assert analysis.status is None
    assert [item.frame.index for item in analysis.frames] == [0, 2]
    assert analysis.measurements == {
        "temporal_decimation_applied": 1,
        "temporal_decimation_kept_frames": 2,
        "temporal_decimation_rejected_frames": 1,
    }
    assert analysis.rejected_frames is not None
    assert analysis.rejected_frames[0]["frame_index"] == 1
    assert analysis.rejected_frames[0]["reason"] == "temporal_decimation_near_duplicate"


def test_analyzer_can_disable_temporal_decimation(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = Frame(0, 0.0, np.zeros((24, 24, 3), np.uint8))
    frames = [frame, Frame(1, 1.0, frame.image.copy()), Frame(2, 2.0, frame.image.copy())]
    monkeypatch.setattr(analyzer_module, "object_mask", lambda image, minimum: np.ones((24, 24), np.uint8) * 255)
    monkeypatch.setattr(analyzer_module, "stable_surface_bbox", lambda mask: (2, 2, 20, 20))
    monkeypatch.setattr(analyzer_module, "publish_surface_mask", lambda mask, bbox: mask.copy())
    monkeypatch.setattr(analyzer_module, "masked_sharpness", lambda image, mask: 50.0)

    analysis = VideoAnalyzer().analyze(frames, UnwrapConfig(enable_temporal_decimation=False))

    assert analysis.status is None
    assert [item.frame.index for item in analysis.frames] == [0, 1, 2]
    assert analysis.measurements == {
        "temporal_decimation_applied": 0,
        "temporal_decimation_kept_frames": 3,
        "temporal_decimation_rejected_frames": 0,
    }


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
            {"surface_coverage_fraction": 1.0, "pose_residual_radians": 0.01, "accepted_pose_pairs": 2, "quality_gate_passed": 1, "rectification_applied": 1},
            {},
        ),
    )

    output = tmp_path / "surface.png"
    result = ObjectUnwrapper(UnwrapConfig(min_accepted_pose_pair_fraction=1.0, allow_partial=True)).unwrap_video("input.mp4", output)

    assert result.diagnostics.status is UnwrapStatus.PARTIAL_SURFACE
    assert result.output_path == output
    assert cv2.imread(str(output), cv2.IMREAD_UNCHANGED).shape == (6, 7, 4)
    assert (tmp_path / "surface_debug" / "run.json").exists()


def test_unwrapper_photo_mode_crops_to_visible_area(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from panoramator.object_unwrap import service

    class FakeSource:
        def __init__(self, *args) -> None:
            pass

        def open(self) -> None:
            return None

        def iter_frames(self):
            return [Frame(0, 0.0, np.zeros((8, 8, 3), np.uint8)), Frame(1, 1.0, np.zeros((8, 8, 3), np.uint8))]

        def close(self) -> None:
            return None

    analyzed = [_analyzed_frame(0), _analyzed_frame(1)]
    monkeypatch.setattr(service, "OpenCVVideoSource", FakeSource)
    monkeypatch.setattr(service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL))
    coverage = np.zeros((6, 10), np.uint8)
    coverage[1:5, 2:8] = 255
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((6, 10, 3), 120, np.uint8),
            coverage.copy(),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {"surface_coverage_fraction": 1.0, "pose_residual_radians": 0.01, "accepted_pose_pairs": 2, "quality_gate_passed": 1, "rectification_applied": 1},
            {},
        ),
    )

    output = tmp_path / "surface.png"
    result = ObjectUnwrapper(
        UnwrapConfig(min_accepted_pose_pair_fraction=1.0, allow_partial=True, photo_mode=True, photo_crop_margin_px=0)
    ).unwrap_video("input.mp4", output)

    saved = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert result.diagnostics.status is UnwrapStatus.PARTIAL_SURFACE
    assert saved.shape[:2] == (4, 6)


def test_unwrapper_uses_planar_fallback_when_geometry_is_not_confirmed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from panoramator.object_unwrap import service

    class FakeSource:
        def __init__(self, *args) -> None:
            pass

        def open(self) -> None:
            return None

        def iter_frames(self):
            return [Frame(0, 0.0, np.zeros((8, 8, 3), np.uint8)), Frame(1, 1.0, np.zeros((8, 8, 3), np.uint8))]

        def close(self) -> None:
            return None

    analyzed = [_analyzed_frame(0), _analyzed_frame(1), _analyzed_frame(2)]
    planar = np.full((5, 7, 3), 90, np.uint8)
    planar_coverage = np.full((5, 7), 255, np.uint8)
    monkeypatch.setattr(service, "OpenCVVideoSource", FakeSource)
    monkeypatch.setattr(service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL))
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((6, 8, 3), 140, np.uint8),
            np.full((6, 8), 255, np.uint8),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {
                "surface_coverage_fraction": 1.0,
                "pose_residual_radians": 0.5,
                "accepted_pose_pairs": 0,
                "quality_gate_passed": 1,
                "rectification_applied": 0,
            },
            {"mosaic": planar.copy(), "mosaic_coverage": planar_coverage.copy()},
        ),
    )

    output = tmp_path / "surface.png"
    result = ObjectUnwrapper(UnwrapConfig(allow_partial=True)).unwrap_video("input.mp4", output)

    saved = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert result.diagnostics.status is UnwrapStatus.PARTIAL_SURFACE
    assert result.diagnostics.message.startswith("A connected image-space mosaic is available")
    assert saved.shape[:2] == planar_coverage.shape
    assert np.all(saved[:, :, :3] == 90)
    assert np.array_equal(saved[:, :, 3], planar_coverage)


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
    assert (tmp_path / "surface_debug" / "run.json").exists()


def test_unwrapper_rejects_output_formats_without_alpha(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from panoramator.object_unwrap import service

    class FakeSource:
        def __init__(self, *args) -> None:
            pass

        def open(self) -> None:
            return None

        def iter_frames(self):
            return [Frame(0, 0.0, np.zeros((8, 8, 3), np.uint8)), Frame(1, 1.0, np.zeros((8, 8, 3), np.uint8))]

        def close(self) -> None:
            return None

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
            {"surface_coverage_fraction": 1.0, "pose_residual_radians": 0.01, "accepted_pose_pairs": 2, "quality_gate_passed": 1, "rectification_applied": 1},
            {},
        ),
    )

    with pytest.raises(ValueError, match="support alpha"):
        ObjectUnwrapper(UnwrapConfig(min_accepted_pose_pair_fraction=1.0, allow_partial=True)).unwrap_video(
            "input.mp4", tmp_path / "surface.jpg"
        )


def test_unwrapper_raises_when_image_cannot_be_written(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from panoramator.object_unwrap import service

    class FakeSource:
        def __init__(self, *args) -> None:
            pass

        def open(self) -> None:
            return None

        def iter_frames(self):
            return [Frame(0, 0.0, np.zeros((8, 8, 3), np.uint8)), Frame(1, 1.0, np.zeros((8, 8, 3), np.uint8))]

        def close(self) -> None:
            return None

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
            {"surface_coverage_fraction": 1.0, "pose_residual_radians": 0.01, "accepted_pose_pairs": 2, "quality_gate_passed": 1, "rectification_applied": 1},
            {},
        ),
    )
    monkeypatch.setattr(cv2, "imwrite", lambda path, image: False)

    with pytest.raises(RuntimeError, match="Failed to write unwrap image"):
        ObjectUnwrapper(UnwrapConfig(min_accepted_pose_pair_fraction=1.0, allow_partial=True)).unwrap_video(
            "input.mp4", tmp_path / "surface.png"
        )


def test_unwrapper_can_disable_debug_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from panoramator.object_unwrap import service

    monkeypatch.setattr(service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis([], SurfaceKind.AUTO, UnwrapStatus.OBJECT_NOT_DETECTED, "missing", "retry"))
    monkeypatch.setattr(service.OpenCVVideoSource, "open", lambda self: None)
    monkeypatch.setattr(service.OpenCVVideoSource, "iter_frames", lambda self: [])
    monkeypatch.setattr(service.OpenCVVideoSource, "close", lambda self: None)

    output = tmp_path / "surface.png"
    result = ObjectUnwrapper(UnwrapConfig(save_debug_artifacts=False)).unwrap_video("input.mp4", output)

    assert result.diagnostics.status is UnwrapStatus.OBJECT_NOT_DETECTED
    assert result.diagnostics.output_files == []
    assert not (tmp_path / "surface_debug").exists()


def test_artifact_writer_raises_when_coverage_cannot_be_saved(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from panoramator.object_unwrap.models import UnwrapDiagnostics

    monkeypatch.setattr(cv2, "imwrite", lambda path, image: False)
    diagnostics = UnwrapDiagnostics(UnwrapStatus.OK, "ok", "", SurfaceKind.CYLINDRICAL)
    with pytest.raises(RuntimeError, match="coverage"):
        write_artifacts(tmp_path / "surface.png", UnwrapConfig(), diagnostics, np.ones((2, 2), np.uint8))


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


def test_quality_gate_rejects_high_error_owner_boundaries() -> None:
    image = np.full((6, 8, 3), 120, np.uint8)
    coverage = np.full((6, 8), 255, np.uint8)
    source = np.zeros((6, 8), np.uint16)
    source[:, :4] = 1
    source[:, 4:] = 2
    error = np.zeros((6, 8), np.uint8)
    error[:, 3:5] = 80

    gate = evaluate_mosaic_quality(image, coverage, source, error, 20.0, 0.72, 40.0, 0.04)

    assert gate.passed is False
    assert gate.mean_boundary_error >= 40.0
    assert gate.weighted_seam_footprint > 0.04


def test_quality_gate_accepts_localized_boundary_defect_with_small_footprint() -> None:
    image = np.full((48, 120, 3), 90, np.uint8)
    image[:, 58:62] = 240
    coverage = np.full((48, 120), 255, np.uint8)
    source = np.ones((48, 120), np.uint16)
    source[20:24, 58:62] = 2
    error = np.zeros((48, 120), np.uint8)
    error[20:24, 58:62] = 80

    gate = evaluate_mosaic_quality(image, coverage, source, error, 85.0, 0.20, 40.0, 0.01)

    assert gate.severe_boundary_fraction > 0.20
    assert gate.severe_boundary_footprint < 0.01
    assert gate.weighted_seam_footprint < 0.01
    assert gate.passed is True


def test_strip_estimate_and_rectification_straighten_observed_band() -> None:
    height, width = 60, 48
    image = np.zeros((height, width, 3), np.uint8)
    coverage = np.zeros((height, width), np.uint8)
    source = np.ones((height, width), np.uint16)
    error = np.zeros((height, width), np.uint8)
    for x in range(width):
        top = 10 + x // 6
        bottom = 42 + x // 6
        coverage[top:bottom + 1, x] = 255
        image[top:bottom + 1, x] = (30 + x, 150, 220)
    strip = estimate_strip(coverage, min_column_fraction=0.8, smoothing_window=7, max_axis_step=4.0)

    assert strip is not None
    rectified, rectified_coverage, rectified_source, rectified_error = rectify_mosaic(
        image, coverage, source, error, strip, output_height=40
    )

    top_rows = [int(np.flatnonzero(rectified_coverage[:, x])[0]) for x in range(width)]
    bottom_rows = [int(np.flatnonzero(rectified_coverage[:, x])[-1]) for x in range(width)]
    assert max(top_rows) - min(top_rows) <= 1
    assert max(bottom_rows) - min(bottom_rows) <= 1
    assert rectified.shape[0] == 40
    assert rectified.shape[1] > width
    assert rectified_source.shape == rectified.shape[:2]
    assert rectified_error.shape == rectified.shape[:2]


def test_planar_mosaic_prefers_stronger_overlap_and_records_conflict_error() -> None:
    left = np.full((12, 12, 3), 20, np.uint8)
    right = np.full((12, 12, 3), 180, np.uint8)
    mask = np.full((12, 12), 255, np.uint8)
    frames = [
        AnalyzedFrame(Frame(0, 0.0, left), mask, mask.copy(), 1.0, (0, 0, 12, 12)),
        AnalyzedFrame(Frame(1, 1.0, right), mask, mask.copy(), 2.0, (0, 0, 12, 12)),
    ]
    edges = [
        {
            "left_frame": 0,
            "right_frame": 1,
            "reason": "ok",
            "a00": 1.0,
            "a01": 0.0,
            "a02": 0.0,
            "a10": 0.0,
            "a11": 1.0,
            "a12": 0.0,
        }
    ]

    mosaic = build_planar_mosaic(frames, edges, output_height=12)

    assert mosaic is not None
    image, coverage, owner, error = mosaic
    assert np.all(coverage == 255)
    assert np.all(owner == 2)
    assert np.all(image == 180)
    assert int(error.max()) > 0


def test_planar_mosaic_requires_a_connected_edge_chain() -> None:
    image = np.full((12, 12, 3), 20, np.uint8)
    mask = np.full((12, 12), 255, np.uint8)
    frames = [
        AnalyzedFrame(Frame(0, 0.0, image), mask, mask.copy(), 1.0, (0, 0, 12, 12)),
        AnalyzedFrame(Frame(1, 1.0, image), mask, mask.copy(), 1.0, (0, 0, 12, 12)),
    ]

    assert build_planar_mosaic(frames, [], output_height=12) is None


def test_strip_estimate_rejects_columns_with_spikes_and_weak_support() -> None:
    coverage = np.zeros((60, 36), np.uint8)
    for x in range(36):
        coverage[12:44, x] = 255
    coverage[4:12, 10] = 255
    coverage[:, 24] = 0
    coverage[20:28, 24] = 255

    strip = estimate_strip(coverage, min_column_fraction=0.8, smoothing_window=7, max_axis_step=4.0)

    assert strip is not None
    assert strip.valid_columns[10] == 0
    assert strip.valid_columns[24] == 0
    assert strip.measurements["rectification_column_fraction"] < 1.0


def test_strip_estimate_regularizes_local_band_height_kinks() -> None:
    coverage = np.zeros((80, 60), np.uint8)
    for x in range(60):
        top = 14 + x // 20
        bottom = 54 + x // 20
        if 26 <= x <= 32:
            bottom += 10
        coverage[top:bottom + 1, x] = 255

    strip = estimate_strip(coverage, min_column_fraction=0.8, smoothing_window=9, max_axis_step=5.0)

    assert strip is not None
    heights = strip.bottom - strip.top
    assert float(np.max(heights) - np.min(heights)) < 12.0
    assert strip.max_bottom_step < 6.0


def test_cylinder_builder_prefers_baseline_planar_mosaic_over_angular_mosaic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from panoramator.object_unwrap.cylinder import builder as builder_module

    baseline = np.full((20, 30, 3), 80, np.uint8)
    baseline[:, 5:25] = (10, 160, 220)
    baseline_coverage = np.full((20, 30), 255, np.uint8)
    baseline_source = np.ones((20, 30), np.uint16)
    baseline_error = np.zeros((20, 30), np.uint8)

    ghosted = np.full((20, 30, 3), 180, np.uint8)
    ghosted[:, 2:28] = (220, 30, 30)

    monkeypatch.setattr(
        builder_module,
        "build_planar_mosaic",
        lambda frames, edges, output_height: (baseline.copy(), baseline_coverage.copy(), baseline_source.copy(), baseline_error.copy()),
    )
    monkeypatch.setattr(
        builder_module.CylinderUnwrapBuilder,
        "_feature_mosaic",
        staticmethod(
            lambda fragments, angles, min_angle, angle_span, atlas_width: (
                ghosted.copy(),
                baseline_coverage.copy(),
                baseline_source.copy(),
                baseline_error.copy(),
            )
        ),
    )
    monkeypatch.setattr(
        builder_module,
        "solve_monotonic_trajectory",
        lambda observations: CylinderTrajectory(
            angles=[0.0, 0.2, 0.4],
            steps=[0.2, 0.2],
            accepted_pairs=2,
            residual_radians=0.01,
            sweep_radians=0.4,
            repeated_observation=False,
            accepted=[True, True],
            rejection_reasons=["", ""],
        ),
    )
    monkeypatch.setattr(
        builder_module,
        "estimate_strip",
        lambda coverage, min_column_fraction, smoothing_window, max_axis_step: None,
    )

    image, coverage, _model, measurements, artifacts = CylinderUnwrapBuilder().build(
        [_analyzed_frame(0), _analyzed_frame(1), _analyzed_frame(2)],
        UnwrapConfig(surface_kind=SurfaceKind.CYLINDRICAL, output_height=20, output_width=30),
    )

    assert np.array_equal(image, baseline)
    assert np.array_equal(coverage, baseline_coverage)
    assert np.array_equal(artifacts["mosaic"], baseline)
    assert np.array_equal(artifacts["angular_mosaic"], ghosted)
    assert measurements["rectification_applied"] == 0

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
from panoramator.object_unwrap.cylinder.mapper import (
    central_band,
    horizontal_shift,
    normalized_wall,
)
from panoramator.object_unwrap.cylinder.pose import CylinderTrajectory
from panoramator.object_unwrap.diagnostics import write_artifacts
from panoramator.object_unwrap.models import (
    PublishProfile,
    SurfaceKind,
    SurfaceModel,
    UnwrapConfig,
    UnwrapDiagnostics,
    UnwrapResult,
    UnwrapStatus,
)
from panoramator.object_unwrap.planar_mosaic import build_planar_mosaic
from panoramator.object_unwrap.pose import optimize_rotation_angles
from panoramator.object_unwrap.rectification import (
    estimate_strip,
    evaluate_mosaic_quality,
    rectify_mosaic,
)
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


def test_object_mask_rejects_grabcut_failures_and_border_sized_components(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((20, 20, 3), np.uint8)

    def failing_grabcut(*args, **kwargs):
        raise cv2.error("grabcut", "grabcut", "failed")

    monkeypatch.setattr(cv2, "grabCut", failing_grabcut)
    assert object_mask(image) is None

    def full_frame_grabcut(image, labels, rectangle, background, foreground, iterations, mode):
        labels[:, :] = cv2.GC_FGD

    monkeypatch.setattr(cv2, "grabCut", full_frame_grabcut)
    assert object_mask(image, min_area_ratio=0.01) is None


def test_stable_surface_bbox_rejects_shallow_and_sparse_bands() -> None:
    shallow = np.zeros((32, 24), np.uint8)
    shallow[10:18, 4:20] = 255
    assert stable_surface_bbox(shallow) is None

    sparse = np.zeros((48, 32), np.uint8)
    for row in range(10, 30):
        sparse[row, (row - 10) % 8] = 255
    assert stable_surface_bbox(sparse) is None


def test_publish_surface_mask_trims_upper_spikes_from_observed_band() -> None:
    mask = np.zeros((48, 32), np.uint8)
    mask[12:40, 6:26] = 255
    mask[3:12, 14:16] = 255

    publish = publish_surface_mask(mask, (6, 12, 20, 28))

    assert np.count_nonzero(publish[3:11, 14:16]) == 0
    assert np.all(publish[16:36, 8:24] == 255)


def test_publish_surface_mask_uses_nearest_run_and_ignores_unsupported_columns() -> None:
    mask = np.zeros((40, 16), np.uint8)
    mask[5:9, 4] = 255
    mask[18:31, 4] = 255
    mask[6:10, 5] = 255
    mask[17:30, 5] = 255
    mask[19:30, 6:10] = 255

    publish = publish_surface_mask(mask, (4, 16, 6, 14))

    assert np.count_nonzero(publish[5:10, 4:6]) == 0
    assert np.all(publish[20:29, 4:10] == 255)
    assert np.count_nonzero(publish[:, 8]) > 0


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
    assert analysis.measurements is not None
    assert analysis.measurements["temporal_decimation_applied"] == 1
    assert analysis.measurements["temporal_decimation_kept_frames"] == 2
    assert analysis.measurements["temporal_decimation_rejected_frames"] == 1
    assert analysis.measurements["temporal_decimation_observed_detail"] >= 0.0
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


def test_analyzer_temporal_decimation_rejects_low_surface_contribution(monkeypatch: pytest.MonkeyPatch) -> None:
    base = np.full((24, 24, 3), 120, np.uint8)
    base[:, 4:20] = (80, 140, 220)
    shifted = base.copy()
    shifted[:, 5:21] = (82, 138, 218)
    frames = [
        Frame(0, 0.0, base.copy()),
        Frame(1, 1.0, shifted.copy()),
        Frame(2, 2.0, np.roll(base, 10, axis=1)),
    ]
    monkeypatch.setattr(analyzer_module, "object_mask", lambda image, minimum: np.ones((24, 24), np.uint8) * 255)
    monkeypatch.setattr(analyzer_module, "stable_surface_bbox", lambda mask: (2, 2, 20, 20))
    monkeypatch.setattr(analyzer_module, "publish_surface_mask", lambda mask, bbox: mask.copy())
    monkeypatch.setattr(analyzer_module, "masked_sharpness", lambda image, mask: 50.0)

    analysis = VideoAnalyzer().analyze(
        frames,
        UnwrapConfig(
            temporal_decimation_max_mask_iou=0.99,
            temporal_decimation_min_band_difference=0.0,
            temporal_decimation_min_bbox_shift=0.0,
            temporal_decimation_min_new_mask_fraction=0.05,
            temporal_decimation_min_detail_gain=0.02,
        ),
    )

    assert analysis.status is None
    assert [item.frame.index for item in analysis.frames] == [0, 2]
    assert analysis.rejected_frames is not None
    assert analysis.rejected_frames[0]["reason"] == "temporal_decimation_low_surface_contribution"
    assert analysis.rejected_frames[0]["new_mask_fraction"] < 0.05


def test_analyzer_temporal_decimation_keeps_frame_with_new_detail_contribution(monkeypatch: pytest.MonkeyPatch) -> None:
    base = np.full((24, 24, 3), 110, np.uint8)
    base[:, 4:20] = 160
    detailed = base.copy()
    detailed[:, 18:24] = 240
    frames = [
        Frame(0, 0.0, base.copy()),
        Frame(1, 1.0, detailed.copy()),
        Frame(2, 2.0, np.roll(detailed, 6, axis=1)),
    ]
    masks = [
        np.pad(np.ones((24, 18), np.uint8) * 255, ((0, 0), (0, 6))),
        np.ones((24, 24), np.uint8) * 255,
        np.ones((24, 24), np.uint8) * 255,
    ]
    state = {"index": 0}

    def fake_object_mask(image, minimum):
        mask = masks[state["index"]]
        state["index"] += 1
        return mask

    monkeypatch.setattr(analyzer_module, "object_mask", fake_object_mask)
    monkeypatch.setattr(analyzer_module, "stable_surface_bbox", lambda mask: (0, 0, mask.shape[1], mask.shape[0]))
    monkeypatch.setattr(analyzer_module, "publish_surface_mask", lambda mask, bbox: mask.copy())
    monkeypatch.setattr(analyzer_module, "masked_sharpness", lambda image, mask: 50.0)

    analysis = VideoAnalyzer().analyze(
        frames,
        UnwrapConfig(
            temporal_decimation_max_mask_iou=1.0,
            temporal_decimation_min_band_difference=1.0,
            temporal_decimation_min_bbox_shift=1.0,
            temporal_decimation_min_new_mask_fraction=0.05,
            temporal_decimation_min_detail_gain=0.005,
        ),
    )

    assert analysis.status is None
    assert [item.frame.index for item in analysis.frames] == [0, 1]
    assert analysis.rejected_frames is not None
    assert analysis.rejected_frames[0]["frame_index"] == 2
    assert analysis.rejected_frames[0]["reason"] == "temporal_decimation_near_duplicate"


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


def test_feature_shift_returns_zero_without_descriptors(monkeypatch: pytest.MonkeyPatch) -> None:
    from panoramator.object_unwrap.cylinder import mapper as mapper_module

    class FakeDetector:
        def detectAndCompute(self, image, mask):
            return [], None

    monkeypatch.setattr(cv2, "ORB_create", lambda nfeatures=1200: FakeDetector())

    image = np.zeros((16, 16, 3), np.uint8)
    mask = np.ones((16, 16), np.uint8) * 255
    shift, response = mapper_module.feature_shift(image, mask, image, mask)

    assert shift == 0.0
    assert response == 0.0


def test_feature_shift_returns_translation_for_consistent_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    from panoramator.object_unwrap.cylinder import mapper as mapper_module

    class FakeKeyPoint:
        def __init__(self, x: float) -> None:
            self.pt = (x, 4.0)

    class FakeMatch:
        def __init__(self, index: int, distance: float) -> None:
            self.queryIdx = index
            self.trainIdx = index
            self.distance = distance

    class FakeDetector:
        def __init__(self) -> None:
            self.calls = 0

        def detectAndCompute(self, image, mask):
            self.calls += 1
            keypoints = [FakeKeyPoint(float(index)) for index in range(8)]
            if self.calls == 1:
                return keypoints, np.ones((8, 32), np.uint8)
            shifted = [FakeKeyPoint(float(index + 3)) for index in range(8)]
            return shifted, np.ones((8, 32), np.uint8)

    class FakeMatcher:
        def knnMatch(self, left, right, k):
            return [[FakeMatch(index, 1.0), FakeMatch(index, 2.0)] for index in range(8)]

    monkeypatch.setattr(cv2, "ORB_create", lambda nfeatures=1200: FakeDetector())
    monkeypatch.setattr(cv2, "BFMatcher", lambda norm: FakeMatcher())
    monkeypatch.setattr(
        cv2,
        "estimateAffinePartial2D",
        lambda source, target, method, ransacReprojThreshold: (
            np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            np.ones((8, 1), np.uint8),
        ),
    )

    image = np.zeros((16, 16, 3), np.uint8)
    mask = np.ones((16, 16), np.uint8) * 255
    shift, response = mapper_module.feature_shift(image, mask, image, mask)

    assert shift == pytest.approx(3.0)
    assert response == pytest.approx(1.0)


def test_angular_increment_rejects_sparse_or_inconsistent_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    from panoramator.object_unwrap.cylinder import mapper as mapper_module

    class FakeKeyPoint:
        def __init__(self, x: float) -> None:
            self.pt = (x, 0.0)

    class FakeMatch:
        def __init__(self, query: int, train: int, distance: float) -> None:
            self.queryIdx = query
            self.trainIdx = train
            self.distance = distance

    class SparseDetector:
        def detectAndCompute(self, image, mask):
            keypoints = [FakeKeyPoint(float(index)) for index in range(8)]
            return keypoints, np.ones((8, 32), np.uint8)

    class InconsistentDetector:
        def detectAndCompute(self, image, mask):
            left = getattr(self, "_left", True)
            self._left = False
            if left:
                keypoints = [FakeKeyPoint(float(index)) for index in range(12)]
            else:
                offsets = [0, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14]
                keypoints = [FakeKeyPoint(float(index + offsets[index])) for index in range(12)]
            return keypoints, np.ones((12, 32), np.uint8)

    class FakeMatcher:
        def knnMatch(self, left, right, k):
            count = min(len(left), len(right))
            return [[FakeMatch(index, index, 1.0), FakeMatch(index, index, 2.0)] for index in range(count)]

    monkeypatch.setattr(cv2, "BFMatcher", lambda norm: FakeMatcher())

    image = np.zeros((24, 24, 3), np.uint8)
    mask = np.ones((24, 24), np.uint8) * 255

    monkeypatch.setattr(cv2, "ORB_create", lambda nfeatures=1600: SparseDetector())
    step, response = mapper_module.angular_increment(image, mask, image, mask, 0.55)
    assert step == 0.0
    assert response == 0.0

    monkeypatch.setattr(cv2, "ORB_create", lambda nfeatures=1600: InconsistentDetector())
    step, response = mapper_module.angular_increment(image, mask, image, mask, 0.55)
    assert step == 0.0
    assert response == 0.0


def test_angular_increment_returns_consistent_surface_step(monkeypatch: pytest.MonkeyPatch) -> None:
    from panoramator.object_unwrap.cylinder import mapper as mapper_module

    class FakeKeyPoint:
        def __init__(self, x: float) -> None:
            self.pt = (x, 0.0)

    class FakeMatch:
        def __init__(self, index: int, distance: float) -> None:
            self.queryIdx = index
            self.trainIdx = index
            self.distance = distance

    class ConsistentDetector:
        def __init__(self) -> None:
            self.calls = 0

        def detectAndCompute(self, image, mask):
            self.calls += 1
            xs = [12.0 + index * 3.0 for index in range(12)]
            if self.calls == 1:
                keypoints = [FakeKeyPoint(x) for x in xs]
            else:
                keypoints = [FakeKeyPoint(x + 2.0) for x in xs]
            return keypoints, np.ones((12, 32), np.uint8)

    class FakeMatcher:
        def knnMatch(self, left, right, k):
            return [[FakeMatch(index, 1.0), FakeMatch(index, 2.0)] for index in range(12)]

    monkeypatch.setattr(cv2, "ORB_create", lambda nfeatures=1600: ConsistentDetector())
    monkeypatch.setattr(cv2, "BFMatcher", lambda norm: FakeMatcher())

    image = np.zeros((40, 80, 3), np.uint8)
    mask = np.ones((40, 80), np.uint8) * 255
    step, response = mapper_module.angular_increment(image, mask, image, mask, 0.55)

    assert step < 0.0
    assert response == pytest.approx(1.0)


def test_flow_angular_increment_rejects_missing_or_unstable_tracks(monkeypatch: pytest.MonkeyPatch) -> None:
    from panoramator.object_unwrap.cylinder import mapper as mapper_module

    image = np.zeros((24, 24, 3), np.uint8)
    right = np.zeros((20, 20, 3), np.uint8)
    mask = np.ones((24, 24), np.uint8) * 255

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", lambda *args, **kwargs: None)
    step, response = mapper_module.flow_angular_increment(image, mask, right, 0.55)
    assert step == 0.0
    assert response == 0.0

    points = np.array([[[float(index), 8.0]] for index in range(12)], dtype=np.float32)
    monkeypatch.setattr(cv2, "goodFeaturesToTrack", lambda *args, **kwargs: points.copy())

    calls = {"count": 0}

    def fake_flow(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            tracked = points.copy()
            status = np.ones((12, 1), np.uint8)
            errors = np.full((12, 1), 30.0, np.float32)
            return tracked, status, errors
        backward = points.copy()
        backward_status = np.ones((12, 1), np.uint8)
        backward_errors = np.zeros((12, 1), np.float32)
        return backward, backward_status, backward_errors

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_flow)
    step, response = mapper_module.flow_angular_increment(image, mask, right, 0.55)
    assert step == 0.0
    assert response == 0.0


def test_flow_angular_increment_handles_forward_failure_and_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from panoramator.object_unwrap.cylinder import mapper as mapper_module

    image = np.zeros((32, 64, 3), np.uint8)
    mask = np.ones((32, 64), np.uint8) * 255
    points = np.array([[[12.0 + float(index), 8.0]] for index in range(12)], dtype=np.float32)

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", lambda *args, **kwargs: points.copy())
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", lambda *args, **kwargs: (None, None, None))

    step, response = mapper_module.flow_angular_increment(image, mask, image.copy(), 0.55)
    assert step == 0.0
    assert response == 0.0

    calls = {"count": 0}

    def successful_flow(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            tracked = points.copy()
            tracked[:, 0, 0] += 2.0
            return tracked, np.ones((12, 1), np.uint8), np.zeros((12, 1), np.float32)
        backward = points.copy()
        return backward, np.ones((12, 1), np.uint8), np.zeros((12, 1), np.float32)

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", successful_flow)
    step, response = mapper_module.flow_angular_increment(image, mask, image.copy(), 0.55)

    assert step < 0.0
    assert response == pytest.approx(1.0)


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
            np.zeros((2, 2, 3), np.uint8),
            np.ones((2, 2), np.uint8),
            SurfaceModel(SurfaceKind.CYLINDRICAL, confidence=0.8),
            {},
            {},
        ),
    )
    _, _, model, measurements, _ = CurvedSurfaceFallbackBuilder().build([_analyzed_frame()], UnwrapConfig())
    assert model.kind is SurfaceKind.CURVED
    assert model.confidence == pytest.approx(0.4)
    assert measurements["fallback"] == "dominant_side_band"


def test_builder_without_global_pose_uses_feature_shift_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from panoramator.object_unwrap.cylinder import builder as builder_module

    frames = [_analyzed_frame(0), _analyzed_frame(1), _analyzed_frame(2)]
    monkeypatch.setattr(
        builder_module,
        "fit_cylinder",
        lambda frames: type(
            "Fit", (), {"model": SurfaceModel(SurfaceKind.CYLINDRICAL), "boxes": [item.bbox for item in frames]}
        )(),
    )
    monkeypatch.setattr(
        builder_module, "build_image_pose_graph", lambda frames: type("Graph", (), {"edges": [], "valid_edges": 0})()
    )
    monkeypatch.setattr(
        builder_module, "build_planar_mosaic", lambda frames, edges, output_height, publish_profile: None
    )
    monkeypatch.setattr(
        builder_module,
        "normalized_wall",
        lambda image, mask, bbox, output_height: (
            np.zeros((output_height, 32, 3), np.uint8),
            np.ones((output_height, 32), np.uint8) * 255,
        ),
    )
    monkeypatch.setattr(builder_module, "central_band", lambda image, mask, ratio: (image, mask))
    monkeypatch.setattr(builder_module, "angular_increment", lambda *args: (0.01, 0.1))
    monkeypatch.setattr(builder_module, "feature_shift", lambda *args: (8.0, 0.6))
    monkeypatch.setattr(builder_module, "horizontal_shift", lambda *args: (64.0, 0.9))

    _image, _coverage, _model, measurements, artifacts = CylinderUnwrapBuilder().build(
        frames,
        UnwrapConfig(
            surface_kind=SurfaceKind.CYLINDRICAL,
            output_height=24,
            output_width=96,
            enable_global_pose_optimization=False,
        ),
    )

    assert measurements["rendering"] == "experimental_frame_projection"
    assert measurements["accepted_pose_pairs"] == 0
    assert measurements["rejected_pose_pairs"] == 2
    assert measurements["quality_gate_passed"] == 0
    assert len(measurements["angular_steps"]) == 2
    assert all(step < 0 for step in measurements["angular_steps"])
    assert all(pair["rejection_reason"] == "global_pose_disabled" for pair in artifacts["pose_pairs"])


def test_builder_global_angles_handles_empty_and_reversed_constraints(monkeypatch: pytest.MonkeyPatch) -> None:
    from panoramator.object_unwrap.cylinder import builder as builder_module

    builder = CylinderUnwrapBuilder()
    fragments = [
        (np.zeros((8, 12, 3), np.uint8), np.ones((8, 12), np.uint8) * 255),
        (np.zeros((8, 12, 3), np.uint8), np.ones((8, 12), np.uint8) * 255),
        (np.zeros((8, 12, 3), np.uint8), np.ones((8, 12), np.uint8) * 255),
    ]

    monkeypatch.setattr(builder_module, "flow_angular_increment", lambda *args: (0.0, 0.1))
    monkeypatch.setattr(builder_module, "angular_increment", lambda *args: (0.0, 0.1))
    angles, responses, steps, residual = builder._global_angles(fragments, 0.55)
    assert angles == [0.0, 0.0, 0.0]
    assert responses == []
    assert steps == []
    assert residual == float("inf")

    def fake_flow(left, left_mask, right, ratio):
        return (-0.30, 0.8)

    values = iter([(0.20, 0.8), (-0.45, 0.9)])

    monkeypatch.setattr(builder_module, "flow_angular_increment", fake_flow)
    monkeypatch.setattr(builder_module, "angular_increment", lambda *args: next(values))
    angles, responses, steps, residual = builder._global_angles(fragments, 0.55)
    assert len(responses) == 3
    assert len(steps) == 2
    assert angles[1] <= angles[0]
    assert angles[2] <= angles[1]
    assert residual >= 0.0


def test_unwrapper_writes_validated_result_with_synthetic_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
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
    monkeypatch.setattr(
        service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL)
    )
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((6, 7, 3), 120, np.uint8),
            np.full((6, 7), 255, np.uint8),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {
                "surface_coverage_fraction": 1.0,
                "pose_residual_radians": 0.01,
                "accepted_pose_pairs": 2,
                "quality_gate_passed": 1,
                "rectification_applied": 1,
            },
            {},
        ),
    )

    output = tmp_path / "surface.png"
    result = ObjectUnwrapper(UnwrapConfig(min_accepted_pose_pair_fraction=1.0, allow_partial=True)).unwrap_video(
        "input.mp4", output
    )

    assert result.diagnostics.status is UnwrapStatus.PARTIAL_SURFACE
    assert result.output_path == output
    assert cv2.imread(str(output), cv2.IMREAD_UNCHANGED).shape == (6, 7, 4)
    assert (tmp_path / "surface_debug" / "run.json").exists()


def test_unwrapper_returns_unstable_geometry_without_planar_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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
    monkeypatch.setattr(
        service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL)
    )
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((6, 7, 3), 120, np.uint8),
            np.full((6, 7), 255, np.uint8),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {
                "surface_coverage_fraction": 1.0,
                "pose_residual_radians": float("inf"),
                "quality_gate_passed": 0,
                "rectification_applied": 0,
            },
            {},
        ),
    )

    output = tmp_path / "surface.png"
    result = ObjectUnwrapper(UnwrapConfig(allow_partial=True)).unwrap_video("input.mp4", output)

    assert result.image is None
    assert result.coverage is not None
    assert result.diagnostics.status is UnwrapStatus.UNSTABLE_CAMERA_GEOMETRY
    assert "stable surface trajectory" in result.diagnostics.message
    assert result.output_path is None


def test_unwrapper_returns_partial_without_output_when_partial_results_are_disabled(
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

    analyzed = [_analyzed_frame(0), _analyzed_frame(1)]
    monkeypatch.setattr(service, "OpenCVVideoSource", FakeSource)
    monkeypatch.setattr(
        service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL)
    )
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((6, 7, 3), 120, np.uint8),
            np.full((6, 7), 255, np.uint8),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {
                "surface_coverage_fraction": 1.0,
                "pose_residual_radians": 0.01,
                "accepted_pose_pairs": 2,
                "quality_gate_passed": 1,
                "rectification_applied": 1,
            },
            {},
        ),
    )

    output = tmp_path / "surface.png"
    result = ObjectUnwrapper(UnwrapConfig(min_accepted_pose_pair_fraction=1.0, allow_partial=False)).unwrap_video(
        "input.mp4", output
    )

    assert result.image is None
    assert result.coverage is not None
    assert result.diagnostics.status is UnwrapStatus.PARTIAL_SURFACE
    assert result.output_path is None
    assert not output.exists()


def test_unwrapper_reports_experimental_renderer_when_global_pose_is_disabled(
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

    analyzed = [_analyzed_frame(0), _analyzed_frame(1)]
    monkeypatch.setattr(service, "OpenCVVideoSource", FakeSource)
    monkeypatch.setattr(
        service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL)
    )
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((6, 7, 3), 120, np.uint8),
            np.full((6, 7), 255, np.uint8),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {
                "surface_coverage_fraction": 1.0,
                "pose_residual_radians": 99.0,
                "accepted_pose_pairs": 0,
                "quality_gate_passed": 0,
                "rectification_applied": 0,
            },
            {},
        ),
    )

    output = tmp_path / "surface.png"
    result = ObjectUnwrapper(
        UnwrapConfig(min_accepted_pose_pair_fraction=1.0, allow_partial=True, enable_global_pose_optimization=False)
    ).unwrap_video("input.mp4", output)

    assert result.image is not None
    assert result.diagnostics.status is UnwrapStatus.PARTIAL_SURFACE
    assert "experimental renderer" in result.diagnostics.message
    assert "global pose optimization" in result.diagnostics.recommendation


def test_unwrapper_rejects_output_without_alpha_support(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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
    monkeypatch.setattr(
        service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL)
    )
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((6, 7, 3), 120, np.uint8),
            np.full((6, 7), 255, np.uint8),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {
                "surface_coverage_fraction": 1.0,
                "pose_residual_radians": 0.01,
                "accepted_pose_pairs": 2,
                "quality_gate_passed": 1,
                "rectification_applied": 1,
            },
            {},
        ),
    )

    with pytest.raises(ValueError, match="must support alpha"):
        ObjectUnwrapper(UnwrapConfig(min_accepted_pose_pair_fraction=1.0, allow_partial=True)).unwrap_video(
            "input.mp4",
            tmp_path / "surface.jpg",
        )


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
    monkeypatch.setattr(
        service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL)
    )
    coverage = np.zeros((6, 10), np.uint8)
    coverage[1:5, 2:8] = 255
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((6, 10, 3), 120, np.uint8),
            coverage.copy(),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {
                "surface_coverage_fraction": 1.0,
                "pose_residual_radians": 0.01,
                "accepted_pose_pairs": 2,
                "quality_gate_passed": 1,
                "rectification_applied": 1,
            },
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
    assert result.diagnostics.measurements["photo_mode_applied"] == 1
    assert result.diagnostics.measurements["photo_mode_crop_policy"] == "inscribed_rectangle"


def test_unwrapper_photo_mode_applies_to_planar_fallback_when_crop_is_safe(
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
    monkeypatch.setattr(
        service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL)
    )
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
    result = ObjectUnwrapper(UnwrapConfig(allow_partial=True, photo_mode=True)).unwrap_video("input.mp4", output)

    saved = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert saved.shape[:2] == planar_coverage.shape
    assert result.diagnostics.measurements["photo_mode_eligible"] == 1
    assert result.diagnostics.measurements["photo_mode_applied"] == 1
    assert result.diagnostics.measurements["photo_mode_crop_policy"] == "inscribed_rectangle"


def test_unwrapper_photo_mode_rejects_excessive_crop_loss(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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
    monkeypatch.setattr(
        service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL)
    )
    coverage = np.zeros((8, 12), np.uint8)
    coverage[1:7, 1:11] = 255
    coverage[3:5, 5:7] = 0
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((8, 12, 3), 120, np.uint8),
            coverage.copy(),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {
                "surface_coverage_fraction": 1.0,
                "pose_residual_radians": 0.01,
                "accepted_pose_pairs": 2,
                "quality_gate_passed": 1,
                "rectification_applied": 1,
            },
            {},
        ),
    )

    output = tmp_path / "surface.png"
    result = ObjectUnwrapper(
        UnwrapConfig(
            min_accepted_pose_pair_fraction=1.0,
            allow_partial=True,
            photo_mode=True,
            photo_crop_margin_px=0,
            photo_crop_max_loss=0.05,
            photo_crop_max_width_loss=0.05,
        )
    ).unwrap_video("input.mp4", output)

    saved = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert saved.shape[:2] == (6, 10)
    assert result.diagnostics.measurements["photo_mode_eligible"] == 1
    assert result.diagnostics.measurements["photo_mode_applied"] == 0
    assert result.diagnostics.measurements["photo_mode_crop_policy"] == "bounding_fallback_excessive_inscribed_loss"


def test_unwrapper_crop_result_removes_outer_transparent_fields(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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
    monkeypatch.setattr(
        service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL)
    )
    coverage = np.zeros((8, 12), np.uint8)
    coverage[2:6, 3:9] = 255
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((8, 12, 3), 120, np.uint8),
            coverage.copy(),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {
                "surface_coverage_fraction": 1.0,
                "pose_residual_radians": 0.01,
                "accepted_pose_pairs": 2,
                "quality_gate_passed": 1,
                "rectification_applied": 1,
            },
            {},
        ),
    )

    output = tmp_path / "surface.png"
    result = ObjectUnwrapper(
        UnwrapConfig(min_accepted_pose_pair_fraction=1.0, allow_partial=True, crop_result=True)
    ).unwrap_video("input.mp4", output)

    saved = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert saved.shape[:2] == (4, 6)
    assert np.all(saved[:, :, 3] == 255)
    assert result.diagnostics.measurements["crop_result_applied"] == 1
    assert result.diagnostics.measurements["crop_result_policy"] == "preserve_alpha"


def test_unwrapper_crop_result_preserves_internal_transparency(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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
    monkeypatch.setattr(
        service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL)
    )
    coverage = np.zeros((8, 12), np.uint8)
    coverage[1:7, 2:10] = 255
    coverage[3:5, 5:7] = 0
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((8, 12, 3), 120, np.uint8),
            coverage.copy(),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {
                "surface_coverage_fraction": 1.0,
                "pose_residual_radians": 0.01,
                "accepted_pose_pairs": 2,
                "quality_gate_passed": 1,
                "rectification_applied": 1,
            },
            {},
        ),
    )

    output = tmp_path / "surface.png"
    ObjectUnwrapper(
        UnwrapConfig(min_accepted_pose_pair_fraction=1.0, allow_partial=True, crop_result=True)
    ).unwrap_video(
        "input.mp4",
        output,
    )

    saved = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert saved.shape[:2] == (6, 8)
    assert np.any(saved[:, :, 3] == 0)
    assert np.any(saved[:, :, 3] == 255)


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
    monkeypatch.setattr(
        service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL)
    )
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


def test_unwrapper_returns_failure_diagnostics_without_writing_an_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from panoramator.object_unwrap import service

    monkeypatch.setattr(
        service.VideoAnalyzer,
        "analyze",
        lambda self, frames, config: Analysis(
            [], SurfaceKind.AUTO, UnwrapStatus.OBJECT_NOT_DETECTED, "missing", "retry"
        ),
    )
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
    monkeypatch.setattr(
        service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL)
    )
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((6, 7, 3), 120, np.uint8),
            np.full((6, 7), 255, np.uint8),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {
                "surface_coverage_fraction": 1.0,
                "pose_residual_radians": 0.01,
                "accepted_pose_pairs": 2,
                "quality_gate_passed": 1,
                "rectification_applied": 1,
            },
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
    monkeypatch.setattr(
        service.VideoAnalyzer, "analyze", lambda self, frames, config: Analysis(analyzed, SurfaceKind.CYLINDRICAL)
    )
    monkeypatch.setattr(
        service.CylinderUnwrapBuilder,
        "build",
        lambda self, frames, config: (
            np.full((6, 7, 3), 120, np.uint8),
            np.full((6, 7), 255, np.uint8),
            SurfaceModel(SurfaceKind.CYLINDRICAL),
            {
                "surface_coverage_fraction": 1.0,
                "pose_residual_radians": 0.01,
                "accepted_pose_pairs": 2,
                "quality_gate_passed": 1,
                "rectification_applied": 1,
            },
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

    monkeypatch.setattr(
        service.VideoAnalyzer,
        "analyze",
        lambda self, frames, config: Analysis(
            [], SurfaceKind.AUTO, UnwrapStatus.OBJECT_NOT_DETECTED, "missing", "retry"
        ),
    )
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


def test_artifact_writer_raises_when_named_image_artifact_cannot_be_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    diagnostics = UnwrapDiagnostics(UnwrapStatus.OK, "ok", "", SurfaceKind.CYLINDRICAL)

    monkeypatch.setattr(cv2, "imwrite", lambda path, image: not str(path).endswith("source.png"))

    with pytest.raises(RuntimeError, match="unwrap artifact"):
        write_artifacts(
            tmp_path / "surface.png",
            UnwrapConfig(),
            diagnostics,
            None,
            {"source": np.ones((2, 2), np.uint8)},
        )


def test_unwrap_config_publish_profiles_expose_distinct_threshold_sets() -> None:
    conservative = UnwrapConfig(publish_profile=PublishProfile.CONSERVATIVE).publish_profile_settings()
    balanced = UnwrapConfig(publish_profile=PublishProfile.BALANCED).publish_profile_settings()
    coverage_first = UnwrapConfig(publish_profile=PublishProfile.COVERAGE_FIRST).publish_profile_settings()

    assert conservative["anchor_conflict_multiplier"] < balanced["anchor_conflict_multiplier"]
    assert balanced["anchor_conflict_multiplier"] < coverage_first["anchor_conflict_multiplier"]
    assert conservative["rectification_column_fraction_delta"] > balanced["rectification_column_fraction_delta"]
    assert balanced["rectification_column_fraction_delta"] > coverage_first["rectification_column_fraction_delta"]


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
            surface_kind="auto",
            allow_partial=False,
            sampling_step=12,
            max_frames=48,
            min_coverage=0.9,
            output_width=1536,
            output_height=512,
            no_global_pose_optimization=False,
            video_path="in.mp4",
            output_path="out.png",
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


def test_quality_gate_flags_anchor_conflict_more_strictly_than_smooth_boundary() -> None:
    image = np.full((40, 80, 3), 180, np.uint8)
    image[:, 36:44] = 185
    for row in range(8, 32):
        image[row, 38:42] = (20, 20, 20)
    coverage = np.full((40, 80), 255, np.uint8)
    source = np.ones((40, 80), np.uint16)
    source[:, 39:41] = 2
    error = np.zeros((40, 80), np.uint8)
    error[8:32, 39:41] = 56

    gate = evaluate_mosaic_quality(
        image,
        coverage,
        source,
        error,
        80.0,
        0.8,
        40.0,
        0.04,
        max_anchor_conflict_footprint=0.002,
        max_owner_instability=1.0,
    )

    assert gate.anchor_conflict_score > 0.0
    assert gate.anchor_conflict_footprint > 0.002
    assert gate.passed is False
    assert np.count_nonzero(gate.saliency_map[8:32, 38:42]) > 0
    assert np.count_nonzero(gate.overlap_conflict_map[8:32, 39:41]) > 0


def test_quality_gate_reports_owner_instability_and_boundary_maps() -> None:
    image = np.full((32, 48, 3), 120, np.uint8)
    image[:, 12:36] = (40, 160, 220)
    coverage = np.full((32, 48), 255, np.uint8)
    source = np.tile(np.array([1, 1, 2, 2, 1, 1], np.uint16), (32, 8))
    error = np.zeros((32, 48), np.uint8)
    error[:, 1:] = np.where(source[:, 1:] != source[:, :-1], 48, 0)

    gate = evaluate_mosaic_quality(image, coverage, source, error, 80.0, 0.8, 40.0, 0.2)

    assert gate.owner_instability_score > 0.0
    assert gate.saliency_weighted_error > 0.0
    assert np.count_nonzero(gate.saliency_error_map) > 0
    assert np.count_nonzero(gate.owner_transition_map) > 0
    assert np.count_nonzero(gate.owner_instability_map) > 0
    assert np.count_nonzero(gate.boundary_map) > 0
    assert np.count_nonzero(gate.seam_risk_map) > 0


def test_strip_estimate_and_rectification_straighten_observed_band() -> None:
    height, width = 60, 48
    image = np.zeros((height, width, 3), np.uint8)
    coverage = np.zeros((height, width), np.uint8)
    source = np.ones((height, width), np.uint16)
    error = np.zeros((height, width), np.uint8)
    for x in range(width):
        top = 10 + x // 6
        bottom = 42 + x // 6
        coverage[top : bottom + 1, x] = 255
        image[top : bottom + 1, x] = (30 + x, 150, 220)
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
    assert np.count_nonzero(owner == 2) >= owner.size * 0.95
    assert float(np.mean(image)) > 20.0
    assert float(np.mean(image)) < 180.0
    assert int(error.max()) > 0


def test_planar_mosaic_keeps_existing_owner_on_high_detail_conflict_without_clear_gain() -> None:
    left = np.full((16, 16, 3), 220, np.uint8)
    right = left.copy()
    left[:, 7:9] = 10
    right[:, 6:8] = 10
    mask = np.full((16, 16), 255, np.uint8)
    frames = [
        AnalyzedFrame(Frame(0, 0.0, left), mask, mask.copy(), 1.0, (0, 0, 16, 16)),
        AnalyzedFrame(Frame(1, 1.0, right), mask, mask.copy(), 1.02, (0, 0, 16, 16)),
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

    mosaic = build_planar_mosaic(frames, edges, output_height=16)

    assert mosaic is not None
    image, coverage, owner, error = mosaic
    assert np.all(coverage == 255)
    assert np.all(owner[:, 8] == 1)
    assert np.all(image[:, 8] == 10)
    assert np.all(owner[:, 7] != 0)
    assert int(error[:, 6:9].max()) > 0


def test_planar_mosaic_blends_smooth_region_while_switching_owner() -> None:
    left = np.full((12, 12, 3), 40, np.uint8)
    right = np.full((12, 12, 3), 200, np.uint8)
    mask = np.full((12, 12), 255, np.uint8)
    frames = [
        AnalyzedFrame(Frame(0, 0.0, left), mask, mask.copy(), 1.0, (0, 0, 12, 12)),
        AnalyzedFrame(Frame(1, 1.0, right), mask, mask.copy(), 4.0, (0, 0, 12, 12)),
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
    image, coverage, owner, _error = mosaic
    assert np.all(coverage == 255)
    assert np.count_nonzero(owner == 2) >= owner.size * 0.95
    assert float(np.mean(image)) > 40.0
    assert float(np.mean(image)) < 200.0


def test_planar_mosaic_publish_profiles_trade_blending_for_cleaner_seams() -> None:
    left = np.full((12, 12, 3), 80, np.uint8)
    left[:, 5:7] = 160
    left = cv2.GaussianBlur(left, (5, 5), 0)
    right = np.full((12, 12, 3), 200, np.uint8)
    mask = np.full((12, 12), 255, np.uint8)
    frames = [
        AnalyzedFrame(Frame(0, 0.0, left), mask, mask.copy(), 1.0, (0, 0, 12, 12)),
        AnalyzedFrame(Frame(1, 1.0, right), mask, mask.copy(), 4.0, (0, 0, 12, 12)),
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

    conservative = build_planar_mosaic(frames, edges, output_height=12, publish_profile=PublishProfile.CONSERVATIVE)
    coverage_first = build_planar_mosaic(frames, edges, output_height=12, publish_profile=PublishProfile.COVERAGE_FIRST)

    assert conservative is not None
    assert coverage_first is not None
    conservative_image, _coverage, conservative_owner, _error = conservative
    coverage_first_image, _coverage, coverage_first_owner, _error = coverage_first
    assert np.count_nonzero(conservative_owner == 2) >= conservative_owner.size * 0.95
    assert np.count_nonzero(coverage_first_owner == 2) >= coverage_first_owner.size * 0.95
    assert float(np.mean(conservative_image[:, 2])) >= 199.0
    assert float(np.mean(coverage_first_image[:, 2])) < float(np.mean(conservative_image[:, 2]))
    assert float(np.mean(coverage_first_image[:, 2])) > 80.0
    assert float(np.mean(coverage_first_image[:, 2])) < 200.0


def test_planar_mosaic_keeps_owner_like_publication_on_anchor_detail() -> None:
    left = np.full((18, 18, 3), 230, np.uint8)
    right = left.copy()
    left[:, 8:10] = 0
    right[:, 7:9] = 255
    mask = np.full((18, 18), 255, np.uint8)
    frames = [
        AnalyzedFrame(Frame(0, 0.0, left), mask, mask.copy(), 1.0, (0, 0, 18, 18)),
        AnalyzedFrame(Frame(1, 1.0, right), mask, mask.copy(), 1.04, (0, 0, 18, 18)),
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

    mosaic = build_planar_mosaic(frames, edges, output_height=18)

    assert mosaic is not None
    image, coverage, owner, _error = mosaic
    assert np.all(coverage == 255)
    assert np.all(owner[:, 9] == 1)
    assert np.all(image[:, 9] == 0)
    assert np.all(image[:, 8] != 128)


def test_planar_mosaic_prefers_existing_owner_inside_stable_detail_patch() -> None:
    left = np.full((18, 18, 3), 210, np.uint8)
    right = left.copy()
    left[:, 8:10] = 0
    right[:, 7:9] = 0
    mask = np.full((18, 18), 255, np.uint8)
    frames = [
        AnalyzedFrame(Frame(0, 0.0, left), mask, mask.copy(), 1.0, (0, 0, 18, 18)),
        AnalyzedFrame(Frame(1, 1.0, right), mask, mask.copy(), 1.05, (0, 0, 18, 18)),
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

    mosaic = build_planar_mosaic(frames, edges, output_height=18)

    assert mosaic is not None
    image, coverage, owner, _error = mosaic
    assert np.all(coverage == 255)
    assert np.all(owner[:, 9] == 1)
    assert np.all(image[:, 9] == 0)


def test_planar_mosaic_allows_owner_change_near_existing_conflict_boundary() -> None:
    left = np.full((16, 20, 3), 90, np.uint8)
    right = np.full((16, 20, 3), 220, np.uint8)
    left[:, 4:7] = 30
    right[:, 6:9] = 240
    mask = np.full((16, 20), 255, np.uint8)
    frames = [
        AnalyzedFrame(Frame(0, 0.0, left), mask, mask.copy(), 1.0, (0, 0, 20, 16)),
        AnalyzedFrame(Frame(1, 1.0, right), mask, mask.copy(), 2.5, (0, 0, 20, 16)),
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

    mosaic = build_planar_mosaic(frames, edges, output_height=16)

    assert mosaic is not None
    image, coverage, owner, error = mosaic
    assert np.all(coverage == 255)
    assert np.count_nonzero(owner[:, 7:9] == 2) >= owner[:, 7:9].size * 0.9
    assert float(np.mean(image[:, 7:9])) >= 235.0
    assert int(error[:, 6:9].max()) > 0


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
        coverage[top : bottom + 1, x] = 255

    strip = estimate_strip(coverage, min_column_fraction=0.8, smoothing_window=9, max_axis_step=5.0)

    assert strip is not None
    heights = strip.bottom - strip.top
    assert float(np.max(heights) - np.min(heights)) < 12.0
    assert strip.max_bottom_step < 6.0


def test_rectification_uses_effective_band_width_for_local_scale_normalization() -> None:
    height, width = 72, 48
    image = np.zeros((height, width, 3), np.uint8)
    coverage = np.zeros((height, width), np.uint8)
    source = np.ones((height, width), np.uint16)
    error = np.zeros((height, width), np.uint8)
    for x in range(width):
        top = 12
        bottom = 38 if x < width // 2 else 52
        coverage[top : bottom + 1, x] = 255
        image[top : bottom + 1, x] = (40 + x, 120, 220)

    strip = estimate_strip(coverage, min_column_fraction=0.8, smoothing_window=9, max_axis_step=5.0)

    assert strip is not None
    assert strip.effective_band_width > width
    rectified, rectified_coverage, _rectified_source, _rectified_error = rectify_mosaic(
        image, coverage, source, error, strip, output_height=40
    )

    assert rectified.shape[1] > width
    left_support = int(np.count_nonzero(rectified_coverage[:, : rectified.shape[1] // 2]))
    right_support = int(np.count_nonzero(rectified_coverage[:, rectified.shape[1] // 2 :]))
    assert right_support > left_support


def test_rectification_stabilizes_isolated_owner_sliver_after_warp() -> None:
    height, width = 40, 18
    image = np.full((height, width, 3), 180, np.uint8)
    coverage = np.zeros((height, width), np.uint8)
    coverage[8:32, :] = 255
    source = np.ones((height, width), np.uint16)
    source[8:32, 9] = 2
    error = np.zeros((height, width), np.uint8)
    error[8:32, 8:11] = 70
    strip = estimate_strip(coverage, min_column_fraction=0.8, smoothing_window=7, max_axis_step=4.0)

    assert strip is not None
    _rectified, rectified_coverage, rectified_source, _rectified_error = rectify_mosaic(
        image, coverage, source, error, strip, output_height=24
    )

    occupied = rectified_coverage > 0
    assert occupied.any()
    center_column = rectified_source[:, rectified_source.shape[1] // 2]
    center_column = center_column[center_column > 0]
    assert center_column.size > 0
    assert np.all(center_column == 1)


def test_rectification_preserves_owner_boundary_on_anchor_detail() -> None:
    height, width = 48, 20
    image = np.full((height, width, 3), 200, np.uint8)
    image[10:38, 9:11] = 0
    coverage = np.zeros((height, width), np.uint8)
    coverage[8:40, :] = 255
    source = np.ones((height, width), np.uint16)
    source[:, 10:] = 2
    error = np.zeros((height, width), np.uint8)
    error[:, 9:11] = 52
    strip = estimate_strip(coverage, min_column_fraction=0.8, smoothing_window=7, max_axis_step=4.0)

    assert strip is not None
    _rectified, rectified_coverage, rectified_source, _rectified_error = rectify_mosaic(
        image, coverage, source, error, strip, output_height=30
    )

    occupied = rectified_coverage > 0
    transitions = rectified_source[:, 1:] != rectified_source[:, :-1]
    anchor_band = occupied[:, 1:] & occupied[:, :-1] & transitions
    assert np.count_nonzero(anchor_band) > 0


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
        lambda frames, edges, output_height, publish_profile: (
            baseline.copy(),
            baseline_coverage.copy(),
            baseline_source.copy(),
            baseline_error.copy(),
        ),
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
    assert artifacts["mosaic_boundary"].shape == baseline_coverage.shape
    assert artifacts["mosaic_saliency"].shape == baseline_coverage.shape
    assert artifacts["mosaic_saliency_error"].shape == baseline_coverage.shape
    assert artifacts["mosaic_overlap_conflict"].shape == baseline_coverage.shape
    assert artifacts["mosaic_owner_transition"].shape == baseline_coverage.shape
    assert artifacts["mosaic_owner_instability"].shape == baseline_coverage.shape
    assert measurements["rectification_applied"] == 0
    assert "quality_gate_anchor_conflict_score" in measurements
    assert "quality_gate_owner_instability" in measurements
    assert "quality_gate_saliency_weighted_error" in measurements

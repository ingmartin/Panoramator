from __future__ import annotations

from pathlib import Path

import numpy as np

from panoramator.application.use_cases import PanoramaBuilder, _ChainBuildResult
from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import (
    Frame,
    FrameQuality,
    MatchSet,
    PanoramaDiagnostics,
    PanoramaResult,
    SelectedFrame,
    VideoMetadata,
)


def _selected_frame(index: int) -> SelectedFrame:
    return SelectedFrame(
        frame=Frame(index=index, timestamp_seconds=float(index), image=np.zeros((4, 4, 3), dtype=np.uint8)),
        quality=FrameQuality(sharpness=10.0 + index, difference_score=float(index), accepted=True, reason="selected"),
    )


def test_build_result_reports_selected_and_validated_frames_separately(tmp_path: Path) -> None:
    config = PanoramaConfig(save_debug_artifacts=False, crop_result=False)
    builder = PanoramaBuilder(config)
    chain_result = _ChainBuildResult(
        backend="sift",
        sampling_step=8,
        attempted_backends=["orb", "sift"],
        attempted_sampling_steps=[15, 8],
        selected_frames=[_selected_frame(0), _selected_frame(1), _selected_frame(2)],
        rejected_frames=[],
        filtered_frames=[_selected_frame(0), _selected_frame(2)],
        pairwise_homographies=[np.eye(3, dtype=np.float64)],
        pair_metrics=[],
    )
    metadata = VideoMetadata(path=Path("input.mp4"), fps=30.0, frame_count=3, width=4, height=4)
    panorama = np.zeros((4, 4, 3), dtype=np.uint8)

    builder._read_metadata = lambda _: metadata
    builder._build_best_chain = lambda _: chain_result
    builder.canvas_builder.build = lambda frame_shapes, homographies: type(
        "Canvas",
        (),
        {"width": 4, "height": 4, "offset_matrix": np.eye(3, dtype=np.float64), "global_homographies": homographies},
    )()
    builder.warper.warp = lambda frame, homography, canvas: (frame.image, np.ones((4, 4), dtype=np.uint8) * 255)
    builder.blender.blend = lambda frames, masks, sharpnesses: panorama

    result = builder.build_from_video("input.mp4", tmp_path / "out.png")

    assert [item["frame_index"] for item in result.diagnostics.selected_frames] == [0, 1, 2]
    assert [item["frame_index"] for item in result.diagnostics.validated_frames] == [0, 2]


def test_debug_artifact_write_failure_does_not_abort_build(tmp_path: Path) -> None:
    config = PanoramaConfig(save_debug_artifacts=True, crop_result=False)
    builder = PanoramaBuilder(config)
    selected = [_selected_frame(0), _selected_frame(1)]
    chain_result = _ChainBuildResult(
        backend="sift",
        sampling_step=8,
        attempted_backends=["orb", "sift"],
        attempted_sampling_steps=[15, 8],
        selected_frames=selected,
        rejected_frames=[],
        filtered_frames=selected,
        pairwise_homographies=[np.eye(3, dtype=np.float64)],
        pair_metrics=[],
    )
    metadata = VideoMetadata(path=Path("input.mp4"), fps=30.0, frame_count=2, width=4, height=4)
    panorama = np.zeros((4, 4, 3), dtype=np.uint8)

    builder._read_metadata = lambda _: metadata
    builder._build_best_chain = lambda _: chain_result
    builder.canvas_builder.build = lambda frame_shapes, homographies: type(
        "Canvas",
        (),
        {"width": 4, "height": 4, "offset_matrix": np.eye(3, dtype=np.float64), "global_homographies": homographies},
    )()
    builder.warper.warp = lambda frame, homography, canvas: (frame.image, np.ones((4, 4), dtype=np.uint8) * 255)
    builder.blender.blend = lambda frames, masks, sharpnesses: panorama

    from panoramator.application import use_cases

    original_write_diagnostics = use_cases.write_diagnostics
    use_cases.write_diagnostics = lambda output, effective_config, diagnostics: (_ for _ in ()).throw(OSError("disk full"))
    try:
        result = builder.build_from_video("input.mp4", tmp_path / "out.png")
    finally:
        use_cases.write_diagnostics = original_write_diagnostics

    assert isinstance(result, PanoramaResult)
    assert isinstance(result.diagnostics, PanoramaDiagnostics)
    assert result.diagnostics.output_files == [str(tmp_path / "out.png")]


def test_photo_mode_crops_to_visible_area(tmp_path: Path) -> None:
    config = PanoramaConfig(save_debug_artifacts=False, crop_result=True, photo_mode=True)
    builder = PanoramaBuilder(config)
    selected = [_selected_frame(0), _selected_frame(1)]
    chain_result = _ChainBuildResult(
        backend="sift",
        sampling_step=8,
        attempted_backends=["orb", "sift"],
        attempted_sampling_steps=[15, 8],
        selected_frames=selected,
        rejected_frames=[],
        filtered_frames=selected,
        pairwise_homographies=[np.eye(3, dtype=np.float64)],
        pair_metrics=[],
    )
    metadata = VideoMetadata(path=Path("input.mp4"), fps=30.0, frame_count=2, width=12, height=12)
    panorama = np.zeros((12, 12, 3), dtype=np.uint8)
    panorama[1:11, 1:11] = 255
    panorama[1:4, 1:4] = 0
    panorama[8:11, 8:11] = 0
    visible_mask = np.zeros((12, 12), dtype=np.uint8)
    visible_mask[1:11, 1:11] = 255

    builder._read_metadata = lambda _: metadata
    builder._build_best_chain = lambda _: chain_result
    builder.canvas_builder.build = lambda frame_shapes, homographies: type(
        "Canvas",
        (),
        {"width": 12, "height": 12, "offset_matrix": np.eye(3, dtype=np.float64), "global_homographies": homographies},
    )()
    builder.warper.warp = lambda frame, homography, canvas: (frame.image, visible_mask.copy())
    builder.blender.blend = lambda frames, masks, sharpnesses: panorama

    result = builder.build_from_video("input.mp4", tmp_path / "out.png")

    assert result.image.shape[:2] == (10, 10)


def test_photo_mode_preserves_black_objects_inside_visible_mask(tmp_path: Path) -> None:
    config = PanoramaConfig(save_debug_artifacts=False, crop_result=True, photo_mode=True, enable_final_sharpening=False)
    builder = PanoramaBuilder(config)
    selected = [_selected_frame(0), _selected_frame(1)]
    chain_result = _ChainBuildResult(
        backend="sift",
        sampling_step=8,
        attempted_backends=["orb", "sift"],
        attempted_sampling_steps=[15, 8],
        selected_frames=selected,
        rejected_frames=[],
        filtered_frames=selected,
        pairwise_homographies=[np.eye(3, dtype=np.float64)],
        pair_metrics=[],
    )
    metadata = VideoMetadata(path=Path("input.mp4"), fps=30.0, frame_count=2, width=12, height=12)
    panorama = np.zeros((12, 12, 3), dtype=np.uint8)
    panorama[1:11, 1:11] = 180
    panorama[4:8, 4:8] = 0
    visible_mask = np.zeros((12, 12), dtype=np.uint8)
    visible_mask[1:11, 1:11] = 255

    builder._read_metadata = lambda _: metadata
    builder._build_best_chain = lambda _: chain_result
    builder.canvas_builder.build = lambda frame_shapes, homographies: type(
        "Canvas",
        (),
        {"width": 12, "height": 12, "offset_matrix": np.eye(3, dtype=np.float64), "global_homographies": homographies},
    )()
    builder.warper.warp = lambda frame, homography, canvas: (frame.image, visible_mask.copy())
    builder.blender.blend = lambda frames, masks, sharpnesses: panorama

    result = builder.build_from_video("input.mp4", tmp_path / "out.png")

    assert result.image.shape[:2] == (10, 10)
    assert np.any(result.image == 0)


def test_build_applies_photometric_normalization_before_warp(tmp_path: Path) -> None:
    config = PanoramaConfig(save_debug_artifacts=False, crop_result=False, enable_final_sharpening=False)
    builder = PanoramaBuilder(config)
    dark = np.full((4, 4, 3), 60, dtype=np.uint8)
    bright = np.full((4, 4, 3), 180, dtype=np.uint8)
    selected = [
        SelectedFrame(
            frame=Frame(index=0, timestamp_seconds=0.0, image=dark),
            quality=FrameQuality(sharpness=10.0, difference_score=999.0, accepted=True, reason="selected"),
        ),
        SelectedFrame(
            frame=Frame(index=1, timestamp_seconds=1.0, image=bright),
            quality=FrameQuality(sharpness=20.0, difference_score=10.0, accepted=True, reason="selected"),
        ),
    ]
    chain_result = _ChainBuildResult(
        backend="sift",
        sampling_step=8,
        attempted_backends=["sift"],
        attempted_sampling_steps=[8],
        selected_frames=selected,
        rejected_frames=[],
        filtered_frames=selected,
        pairwise_homographies=[np.eye(3, dtype=np.float64)],
        pair_metrics=[],
    )
    metadata = VideoMetadata(path=Path("input.mp4"), fps=30.0, frame_count=2, width=4, height=4)
    warped_means: list[float] = []

    builder._read_metadata = lambda _: metadata
    builder._build_best_chain = lambda _: chain_result
    builder.canvas_builder.build = lambda frame_shapes, homographies: type(
        "Canvas",
        (),
        {"width": 4, "height": 4, "offset_matrix": np.eye(3, dtype=np.float64), "global_homographies": homographies},
    )()
    builder.warper.warp = lambda frame, homography, canvas: (
        warped_means.append(float(frame.image.mean())) or frame.image,
        np.ones((4, 4), dtype=np.uint8) * 255,
    )
    builder.blender.blend = lambda frames, masks, sharpnesses: frames[-1]

    builder.build_from_video("input.mp4", tmp_path / "out.png")

    assert len(warped_means) == 2
    assert warped_means[1] < 180.0


def test_build_applies_final_sharpening(tmp_path: Path) -> None:
    config = PanoramaConfig(save_debug_artifacts=False, crop_result=False, final_sharpen_strength=0.2, final_sharpen_sigma=1.0)
    builder = PanoramaBuilder(config)
    selected = [_selected_frame(0), _selected_frame(1)]
    chain_result = _ChainBuildResult(
        backend="sift",
        sampling_step=8,
        attempted_backends=["sift"],
        attempted_sampling_steps=[8],
        selected_frames=selected,
        rejected_frames=[],
        filtered_frames=selected,
        pairwise_homographies=[np.eye(3, dtype=np.float64)],
        pair_metrics=[],
    )
    metadata = VideoMetadata(path=Path("input.mp4"), fps=30.0, frame_count=2, width=16, height=16)
    panorama = np.zeros((16, 16, 3), dtype=np.uint8)
    panorama[4:12, 4:12] = 255
    panorama = np.clip((panorama.astype(np.float32) * 0.8), 0, 255).astype(np.uint8)

    builder._read_metadata = lambda _: metadata
    builder._build_best_chain = lambda _: chain_result
    builder.canvas_builder.build = lambda frame_shapes, homographies: type(
        "Canvas",
        (),
        {"width": 16, "height": 16, "offset_matrix": np.eye(3, dtype=np.float64), "global_homographies": homographies},
    )()
    builder.warper.warp = lambda frame, homography, canvas: (frame.image, np.ones((4, 4), dtype=np.uint8) * 255)
    builder.blender.blend = lambda frames, masks, sharpnesses: panorama

    result = builder.build_from_video("input.mp4", tmp_path / "out.png")

    assert float(result.image.mean()) >= float(panorama.mean())


def test_build_can_fallback_to_window_alternate_when_geometry_fails(tmp_path: Path) -> None:
    config = PanoramaConfig(save_debug_artifacts=False, crop_result=False, enable_final_sharpening=False)
    builder = PanoramaBuilder(config)
    primary = Frame(index=1, timestamp_seconds=1.0, image=np.ones((4, 4, 3), dtype=np.uint8) * 100)
    alternate = Frame(index=2, timestamp_seconds=1.0, image=np.ones((4, 4, 3), dtype=np.uint8) * 120)
    selected = [
        _selected_frame(0),
        SelectedFrame(
            frame=primary,
            quality=FrameQuality(sharpness=20.0, difference_score=10.0, accepted=True, reason="selected_windowed"),
            alternates=[alternate],
        ),
    ]

    class _Extractor:
        def extract(self, frame):
            return frame.index

    from panoramator.application import use_cases

    original_create_feature_extractor = use_cases.create_feature_extractor
    use_cases.create_feature_extractor = lambda backend_config: _Extractor()
    builder.matcher.match = lambda left, right: MatchSet(raw_count=12, good_matches=[object()] * 10, confidence=0.8)
    outcomes = iter(
        [
            type("Geometry", (), {"homography": None, "inliers": 0, "reprojection_error": float("inf"), "valid": False, "reason": "not_enough_matches"})(),
            type("Geometry", (), {"homography": np.eye(3, dtype=np.float64), "inliers": 10, "reprojection_error": 1.0, "valid": True, "reason": "ok"})(),
        ]
    )
    builder.geometry.estimate = lambda *args: next(outcomes)
    try:
        chain = builder._build_chain(selected, [], config, "sift", [config.sampling_step])
    finally:
        use_cases.create_feature_extractor = original_create_feature_extractor

    assert [item.frame.index for item in chain.filtered_frames] == [0, 2]
    assert chain.filtered_frames[1].quality.reason.endswith("geometry_fallback")
    assert chain.pair_metrics[0]["reason"] == "not_enough_matches"
    assert chain.pair_metrics[1]["reason"] == "ok_window_fallback"


def test_photometric_normalization_clamps_flat_frame_gain() -> None:
    from panoramator.postprocess.enhance import normalize_selected_frames

    frames = [
        SelectedFrame(
            frame=Frame(index=0, timestamp_seconds=0.0, image=np.full((4, 4, 3), 20, dtype=np.uint8)),
            quality=FrameQuality(sharpness=10.0, difference_score=999.0, accepted=True, reason="selected"),
        ),
        SelectedFrame(
            frame=Frame(index=1, timestamp_seconds=1.0, image=np.full((4, 4, 3), 200, dtype=np.uint8)),
            quality=FrameQuality(sharpness=10.0, difference_score=10.0, accepted=True, reason="selected"),
        ),
    ]

    normalized = normalize_selected_frames(frames, PanoramaConfig(photometric_smoothing=1.0))

    assert float(normalized[1].frame.image.mean()) < 200.0
    assert float(normalized[1].frame.image.mean()) > 150.0

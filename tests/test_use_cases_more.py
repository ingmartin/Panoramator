from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from panoramator.application.use_cases import PanoramaBuilder, _ChainBuildResult
from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import (
    FeatureSet,
    Frame,
    FrameQuality,
    MatchSet,
    PairGeometry,
    SelectedFrame,
    VideoMetadata,
)
from panoramator.projection.models import CylindricalProjection, PlanarProjection, SphericalProjection


def _selected_frame(index: int) -> SelectedFrame:
    return SelectedFrame(
        frame=Frame(index=index, timestamp_seconds=float(index), image=np.zeros((4, 4, 3), dtype=np.uint8)),
        quality=FrameQuality(sharpness=10.0 + index, difference_score=float(index), accepted=True, reason="selected"),
    )


def test_build_from_video_requires_two_selected_frames(tmp_path) -> None:
    builder = PanoramaBuilder(PanoramaConfig(save_debug_artifacts=False))
    builder._read_metadata = lambda _: VideoMetadata(path=Path("input.mp4"), fps=30.0, frame_count=1, width=4, height=4)
    builder._build_best_chain = lambda _: _ChainBuildResult(
        backend="orb",
        sampling_step=15,
        attempted_backends=["orb"],
        attempted_sampling_steps=[15],
        selected_frames=[_selected_frame(0)],
        rejected_frames=[],
        filtered_frames=[_selected_frame(0)],
        pairwise_homographies=[],
        pair_metrics=[],
    )

    with pytest.raises(RuntimeError, match="Not enough selected frames"):
        builder.build_from_video("input.mp4", tmp_path / "out.png")


def test_build_from_video_requires_valid_geometry_chain(tmp_path) -> None:
    builder = PanoramaBuilder(PanoramaConfig(save_debug_artifacts=False))
    selected = [_selected_frame(0), _selected_frame(1)]
    builder._read_metadata = lambda _: VideoMetadata(path=Path("input.mp4"), fps=30.0, frame_count=2, width=4, height=4)
    builder._build_best_chain = lambda _: _ChainBuildResult(
        backend="orb",
        sampling_step=15,
        attempted_backends=["orb"],
        attempted_sampling_steps=[15],
        selected_frames=selected,
        rejected_frames=[],
        filtered_frames=selected[:1],
        pairwise_homographies=[],
        pair_metrics=[],
    )

    with pytest.raises(RuntimeError, match="No valid frame chain"):
        builder.build_from_video("input.mp4", tmp_path / "out.png")


def test_combined_visible_mask_merges_all_masks() -> None:
    first = np.array([[0, 255], [0, 0]], dtype=np.uint8)
    second = np.array([[0, 0], [255, 0]], dtype=np.uint8)

    mask = PanoramaBuilder._combined_visible_mask([first, second])

    assert np.array_equal(mask, np.array([[0, 255], [255, 0]], dtype=np.uint8))


def test_fits_canvas_rejects_an_oversized_partial_chain() -> None:
    builder = PanoramaBuilder(PanoramaConfig(max_canvas_width=5, max_canvas_height=5))
    frames = [_selected_frame(0), _selected_frame(1)]
    pairwise_homographies = [np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])]

    assert builder._fits_canvas(frames, pairwise_homographies) is False
    assert PanoramaBuilder._combined_visible_mask([]) is None


def test_build_chain_rejects_pair_that_would_exceed_canvas_limit(monkeypatch) -> None:
    builder = PanoramaBuilder(
        PanoramaConfig(max_canvas_width=5, max_canvas_height=5, enable_feature_fallback=False)
    )
    selected = [_selected_frame(0), _selected_frame(1)]
    features = FeatureSet(keypoints=[], descriptors=None, backend="orb")
    oversized_transform = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    monkeypatch.setattr(
        "panoramator.application.use_cases.create_feature_extractor",
        lambda config: SimpleNamespace(extract=lambda frame: features),
    )
    builder.matcher = SimpleNamespace(match=lambda left, right: MatchSet(8, [], 1.0))
    builder.geometry = SimpleNamespace(
        estimate=lambda *args: PairGeometry(oversized_transform, 8, 0.0, True, "ok")
    )

    result = builder._build_chain(selected, [], builder.config, "orb", [builder.config.sampling_step])

    assert [item.frame.index for item in result.filtered_frames] == [0]
    assert result.pairwise_homographies == []
    assert result.pair_metrics[0]["valid"] is False
    assert result.pair_metrics[0]["reason"] == "canvas_limit"


def test_is_better_chain_prefers_longer_valid_chain() -> None:
    smaller = _ChainBuildResult("orb", 15, ["orb"], [15], [1, 2, 3], [], [1, 2], [], [])
    larger = _ChainBuildResult("orb", 15, ["orb"], [15], [1, 2, 3], [], [1, 2, 3], [], [])

    assert PanoramaBuilder._is_better_chain(larger, smaller) is True
    assert PanoramaBuilder._is_better_chain(smaller, larger) is False


def test_build_best_chain_prefers_candidate_with_more_validated_frames(monkeypatch) -> None:
    builder = PanoramaBuilder(PanoramaConfig(sampling_step=15, fallback_sampling_step=8, enable_sampling_fallback=True))
    selected_short = [_selected_frame(0), _selected_frame(1)]
    selected_long = [_selected_frame(0), _selected_frame(1), _selected_frame(2)]

    def _fake_select(video_path, config):
        return (selected_short if config.sampling_step == 15 else selected_long), []

    def _fake_build_chain_with_fallback(selected_frames, rejected_frames, config, attempted_sampling_steps):
        filtered = selected_frames[:1] if config.sampling_step == 15 else selected_frames
        return _ChainBuildResult(
            backend=config.feature_backend,
            sampling_step=config.sampling_step,
            attempted_backends=[config.feature_backend],
            attempted_sampling_steps=attempted_sampling_steps,
            selected_frames=selected_frames,
            rejected_frames=rejected_frames,
            filtered_frames=filtered,
            pairwise_homographies=[],
            pair_metrics=[],
        )

    monkeypatch.setattr(builder, "_select_frames", _fake_select)
    monkeypatch.setattr(builder, "_build_chain_with_fallback", _fake_build_chain_with_fallback)

    result = builder._build_best_chain("input.mp4")

    assert result.sampling_step == 8
    assert len(result.filtered_frames) == 3


def test_build_chain_with_fallback_uses_fallback_backend_when_primary_is_short(monkeypatch) -> None:
    builder = PanoramaBuilder(PanoramaConfig(feature_backend="orb", fallback_feature_backend="sift", fallback_min_chain_length=3))
    selected = [_selected_frame(0), _selected_frame(1)]

    def _fake_build_chain(selected_frames, rejected_frames, config, backend, attempted_sampling_steps):
        filtered = selected_frames[:1] if backend == "orb" else selected_frames
        return _ChainBuildResult(
            backend=backend,
            sampling_step=config.sampling_step,
            attempted_backends=[backend],
            attempted_sampling_steps=attempted_sampling_steps,
            selected_frames=selected_frames,
            rejected_frames=rejected_frames,
            filtered_frames=filtered,
            pairwise_homographies=[],
            pair_metrics=[],
        )

    monkeypatch.setattr(builder, "_build_chain", _fake_build_chain)

    result = builder._build_chain_with_fallback(selected, [], builder.config, [builder.config.sampling_step])

    assert result.backend == "sift"
    assert result.attempted_backends == ["orb", "sift"]


def test_read_metadata_and_select_frames_close_video_source(monkeypatch) -> None:
    events: list[str] = []

    class _FakeSource:
        def __init__(self, video_path: str, config: PanoramaConfig) -> None:
            assert video_path == "input.mp4"

        def open(self):
            events.append("open")
            return VideoMetadata(path=Path("input.mp4"), fps=30.0, frame_count=2, width=4, height=4)

        def iter_frames(self):
            events.append("iter")
            return [Frame(index=0, timestamp_seconds=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8))]

        def close(self):
            events.append("close")

    class _FakeSelector:
        def __init__(self, config: PanoramaConfig) -> None:
            self.config = config

        def select(self, frames):
            events.append("select")
            return frames, []

    from panoramator.application import use_cases

    monkeypatch.setattr(use_cases, "OpenCVVideoSource", _FakeSource)
    monkeypatch.setattr(use_cases, "FrameSelector", _FakeSelector)
    builder = PanoramaBuilder(PanoramaConfig())

    metadata = builder._read_metadata("input.mp4")
    selected, rejected = builder._select_frames("input.mp4", builder.config)

    assert metadata.frame_count == 2
    assert len(selected) == 1
    assert rejected == []
    assert events == ["open", "close", "open", "iter", "close", "select"]


def test_geometry_projection_respects_explicit_projection_and_rotation_mode() -> None:
    frame = Frame(index=0, timestamp_seconds=0.0, image=np.zeros((12, 16, 3), dtype=np.uint8))

    planar = PanoramaBuilder(PanoramaConfig(capture_mode="linear", projection="auto"))._geometry_projection(
        PanoramaConfig(capture_mode="linear", projection="auto"),
        frame,
    )
    cylindrical = PanoramaBuilder(
        PanoramaConfig(capture_mode="rotation", projection="auto")
    )._geometry_projection(PanoramaConfig(capture_mode="rotation", projection="auto"), frame)
    spherical = PanoramaBuilder(PanoramaConfig(projection="spherical"))._geometry_projection(
        PanoramaConfig(projection="spherical"),
        frame,
    )

    assert isinstance(planar, PlanarProjection)
    assert isinstance(cylindrical, CylindricalProjection)
    assert isinstance(spherical, SphericalProjection)


def test_sampling_steps_skip_non_improving_fallback_values() -> None:
    assert PanoramaBuilder(PanoramaConfig(sampling_step=8, fallback_sampling_step=8))._sampling_steps_to_try() == [8]
    assert PanoramaBuilder(PanoramaConfig(sampling_step=8, fallback_sampling_step=12))._sampling_steps_to_try() == [8]


def test_cylindrical_preview_returns_none_when_preview_chain_is_too_short(monkeypatch) -> None:
    builder = PanoramaBuilder(PanoramaConfig())
    chain = _ChainBuildResult(
        backend="orb",
        sampling_step=15,
        attempted_backends=["orb"],
        attempted_sampling_steps=[15],
        selected_frames=[_selected_frame(0), _selected_frame(1)],
        rejected_frames=[],
        filtered_frames=[_selected_frame(0), _selected_frame(1)],
        pairwise_homographies=[np.eye(3)],
        pair_metrics=[],
    )

    monkeypatch.setattr(
        builder,
        "_build_chain_with_fallback",
        lambda selected_frames, rejected_frames, config, attempted_sampling_steps: _ChainBuildResult(
            backend="orb",
            sampling_step=config.sampling_step,
            attempted_backends=["orb"],
            attempted_sampling_steps=attempted_sampling_steps,
            selected_frames=selected_frames,
            rejected_frames=rejected_frames,
            filtered_frames=selected_frames[:1],
            pairwise_homographies=[],
            pair_metrics=[],
        ),
    )

    assert builder._build_cylindrical_preview(chain) is None

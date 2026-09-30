from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest

from panoramator.application.use_cases import PanoramaBuilder, _ChainBuildResult
from panoramator.config.models import PanoramaConfig
from panoramator.domain.errors import PanoramaCancelled
from panoramator.domain.models import Frame, FrameQuality, SelectedFrame


def _selected(index: int, value: int = 100) -> SelectedFrame:
    return SelectedFrame(
        frame=Frame(index, float(index), np.full((4, 4, 3), value + index, dtype=np.uint8)),
        quality=FrameQuality(10.0, 10.0, True, "selected"),
    )


def _chain() -> _ChainBuildResult:
    selected = [_selected(0), _selected(1)]
    return _ChainBuildResult(
        backend="sift",
        sampling_step=1,
        attempted_backends=["sift"],
        attempted_sampling_steps=[1],
        selected_frames=selected,
        rejected_frames=[],
        filtered_frames=selected,
        pairwise_homographies=[np.eye(3, dtype=np.float64)],
        pair_metrics=[{"confidence": 0.8, "valid": True}],
    )


def _stub_builder(builder: PanoramaBuilder) -> None:
    cast(Any, builder)._build_best_chain_from_frames = lambda frames, rejected, callbacks: _chain()
    cast(Any, builder.canvas_builder).build = lambda frame_shapes, homographies: type(
        "Canvas",
        (),
        {
            "width": 4,
            "height": 4,
            "offset_matrix": np.eye(3),
            "global_homographies": homographies,
        },
    )()
    cast(Any, builder.warper).warp = lambda frame, homography, canvas: (
        frame.image,
        np.ones((4, 4), dtype=np.uint8) * 255,
    )
    cast(Any, builder.blender).blend = lambda frames, masks, sharpnesses: frames[0]


def test_build_from_frames_returns_metadata_progress_and_atomic_output(tmp_path: Path) -> None:
    builder = PanoramaBuilder(
        PanoramaConfig(
            capture_mode="linear",
            blur_threshold=0,
            min_difference=0,
            crop_result=False,
            save_debug_artifacts=False,
            enable_final_sharpening=False,
        )
    )
    _stub_builder(builder)
    progress = []

    result = builder.build_from_frames(
        [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)],
        tmp_path / "panorama.png",
        progress_callback=progress.append,
    )

    assert result.output_path == str(tmp_path / "panorama.png")
    assert (result.width, result.height) == (4, 4)
    assert result.used_frames == 2
    assert result.quality == pytest.approx(0.8)
    assert cv2.imread(str(tmp_path / "panorama.png")) is not None
    assert {item.stage for item in progress} >= {
        "reading",
        "selection",
        "homography",
        "rendering",
        "saving",
        "completed",
    }


def test_build_from_frames_cancellation_before_read_does_not_write(tmp_path: Path) -> None:
    builder = PanoramaBuilder(PanoramaConfig(capture_mode="linear", save_debug_artifacts=False))

    with pytest.raises(PanoramaCancelled):
        builder.build_from_frames(
            [np.zeros((4, 4, 3), dtype=np.uint8)],
            tmp_path / "panorama.png",
            cancel_callback=lambda: True,
        )

    assert not (tmp_path / "panorama.png").exists()


def test_build_from_frames_cancellation_during_read_stops_iteration(tmp_path: Path) -> None:
    builder = PanoramaBuilder(PanoramaConfig(capture_mode="linear", save_debug_artifacts=False))
    yielded = 0

    def _frames():
        nonlocal yielded
        for _ in range(3):
            yielded += 1
            yield np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(PanoramaCancelled):
        builder.build_from_frames(
            _frames(),
            tmp_path / "panorama.png",
            cancel_callback=lambda: yielded >= 1,
        )

    assert yielded == 1
    assert not (tmp_path / "panorama.png").exists()


def test_build_from_frames_decodes_only_bounded_uniform_sample(monkeypatch, tmp_path: Path) -> None:
    config = PanoramaConfig(
        capture_mode="linear",
        max_frames=2,
        blur_threshold=0,
        min_difference=0,
        crop_result=False,
        save_debug_artifacts=False,
        enable_final_sharpening=False,
    )
    builder = PanoramaBuilder(config)
    _stub_builder(builder)
    decoded: list[np.ndarray] = []

    def _decode(value):
        decoded.append(value)
        return value

    monkeypatch.setattr("panoramator.application.use_cases.decode_frame_input", _decode)
    frames = [np.full((4, 4, 3), index, dtype=np.uint8) for index in range(6)]

    result = builder.build_from_frames(frames, tmp_path / "sampled.png")

    assert len(decoded) == 2
    assert result.metadata.frame_count == 6
    assert result.used_frames == 2


def test_build_from_frames_rejects_malformed_input_without_aborting_batch(tmp_path: Path) -> None:
    builder = PanoramaBuilder(
        PanoramaConfig(
            capture_mode="linear",
            blur_threshold=0,
            min_difference=0,
            crop_result=False,
            save_debug_artifacts=False,
            enable_final_sharpening=False,
        )
    )
    _stub_builder(builder)

    def _keep_rejected(frames, rejected, callbacks):
        chain = _chain()
        chain.rejected_frames = rejected
        return chain

    cast(Any, builder)._build_best_chain_from_frames = _keep_rejected

    result = builder.build_from_frames(
        [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
        ],
        tmp_path / "panorama.png",
    )

    assert any(item["reason"] == "decode_error" for item in result.diagnostics.rejected_frames)
    assert result.metadata.frame_count == 3


def test_atomic_save_keeps_previous_output_when_encoding_fails(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "panorama.png"
    output.write_bytes(b"previous")
    builder = PanoramaBuilder(PanoramaConfig(capture_mode="linear"))
    monkeypatch.setattr("panoramator.application.use_cases.cv2.imwrite", lambda path, image: False)

    with pytest.raises(RuntimeError, match="Failed to write output image"):
        builder._write_panorama_image(output, np.zeros((2, 2, 3), dtype=np.uint8))

    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(f".{output.name}.*"))


def test_build_from_frames_stitches_synthetic_horizontal_translation(tmp_path: Path) -> None:
    base = np.zeros((120, 260, 3), dtype=np.uint8)
    rng = np.random.default_rng(4)
    base[:] = rng.integers(0, 255, base.shape, dtype=np.uint8)
    for x, y in rng.integers([10, 10], [250, 110], size=(80, 2)):
        cv2.circle(base, (int(x), int(y)), 3, (255, 255, 255), -1)
    frames = [base[:, start : start + 180].copy() for start in (0, 20, 40)]
    config = PanoramaConfig(
        capture_mode="linear",
        sampling_step=1,
        max_frames=5,
        feature_backend="sift",
        enable_feature_fallback=False,
        blur_threshold=0,
        min_difference=0,
        min_match_count=4,
        min_inlier_count=4,
        min_inlier_ratio=0.3,
        crop_result=False,
        save_debug_artifacts=False,
    )

    result = PanoramaBuilder(config).build_from_frames(frames, tmp_path / "synthetic.png")

    assert result.status == "ok"
    assert result.used_frames == 3
    assert result.width >= 200
    assert result.height >= 100
    assert cv2.imread(str(tmp_path / "synthetic.png")) is not None

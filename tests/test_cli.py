from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import panoramator.io.video as video_module
from panoramator.cli import main as cli_main
from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import PanoramaDiagnostics, PanoramaResult, VideoMetadata


def _build_args(**overrides) -> argparse.Namespace:
    values = {
        "video_path": "input.mp4",
        "output_path": "out.png",
        "config": None,
        "sampling_step": None,
        "max_frames": None,
        "downscale": None,
        "feature_downscale": None,
        "feature_backend": None,
        "blur_threshold": None,
        "adaptive_blur_threshold": False,
        "blur_rescue_sharpening": False,
        "no_blur_rescue_sharpening": False,
        "blur_rescue_sharpen_strength": None,
        "blur_rescue_sharpen_sigma": None,
        "frame_selection_window_size": None,
        "motion_model": None,
        "feather_blend_kernel": None,
        "seam_blur_kernel": None,
        "seam_band_width": None,
        "photometric_normalization": False,
        "no_photometric_normalization": False,
        "photometric_smoothing": None,
        "overlap_sharpness_weight": None,
        "narrow_gap_fill": False,
        "no_narrow_gap_fill": False,
        "max_narrow_gap_width": None,
        "photo_crop_margin_px": None,
        "photo_mode": False,
        "final_sharpening": False,
        "no_final_sharpening": False,
        "final_sharpen_strength": None,
        "final_sharpen_sigma": None,
        "feature_fallback": False,
        "no_feature_fallback": False,
        "fallback_feature_backend": None,
        "fallback_min_chain_length": None,
        "sampling_fallback": False,
        "no_sampling_fallback": False,
        "fallback_sampling_step": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_command_applies_overrides_and_prints_summary(monkeypatch, capsys) -> None:
    loaded_config = PanoramaConfig(
        sampling_step=3,
        max_frames=9,
        enable_blur_rescue_sharpening=False,
        enable_photometric_normalization=True,
        enable_final_sharpening=True,
        enable_feature_fallback=False,
        enable_sampling_fallback=False,
    )
    captured: dict[str, PanoramaConfig] = {}

    class _FakeBuilder:
        def __init__(self, config: PanoramaConfig) -> None:
            captured["config"] = config

        def build_from_video(self, video_path: str, output_path: str) -> PanoramaResult:
            diagnostics = PanoramaDiagnostics(
                selected_frames=[{"frame_index": 0}, {"frame_index": 1}],
                rejected_frames=[{"frame_index": 2}],
                feature_backend="sift",
                sampling_step=8,
                fallback_used=True,
            )
            metadata = VideoMetadata(path=Path("input.mp4"), fps=25.0, frame_count=30, width=1920, height=1080)
            return PanoramaResult(image=None, metadata=metadata, diagnostics=diagnostics)  # type: ignore[arg-type]

    monkeypatch.setattr(cli_main.PanoramaConfig, "from_json", classmethod(lambda cls, path: loaded_config))
    monkeypatch.setattr(cli_main, "PanoramaBuilder", _FakeBuilder)

    args = _build_args(
        config="config.json",
        sampling_step=8,
        max_frames=12,
        downscale=0.5,
        feature_downscale=0.75,
        feature_backend="sift",
        blur_threshold=123.0,
        adaptive_blur_threshold=True,
        blur_rescue_sharpening=True,
        no_blur_rescue_sharpening=True,
        blur_rescue_sharpen_strength=0.3,
        blur_rescue_sharpen_sigma=1.2,
        frame_selection_window_size=3,
        motion_model="homography",
        feather_blend_kernel=11,
        seam_blur_kernel=7,
        seam_band_width=9,
        photometric_normalization=True,
        no_photometric_normalization=True,
        photometric_smoothing=0.8,
        overlap_sharpness_weight=0.4,
        narrow_gap_fill=True,
        no_narrow_gap_fill=True,
        max_narrow_gap_width=3,
        photo_crop_margin_px=5,
        photo_mode=True,
        final_sharpening=True,
        no_final_sharpening=True,
        final_sharpen_strength=0.5,
        final_sharpen_sigma=1.5,
        feature_fallback=True,
        no_feature_fallback=True,
        fallback_feature_backend="orb",
        fallback_min_chain_length=5,
        sampling_fallback=True,
        no_sampling_fallback=True,
        fallback_sampling_step=4,
    )

    result = cli_main.build_command(args)

    assert result == 0
    config = captured["config"]
    assert config.sampling_step == 8
    assert config.max_frames == 12
    assert config.downscale == 0.5
    assert config.feature_downscale == 0.75
    assert config.feature_backend == "sift"
    assert config.blur_threshold == 123.0
    assert config.adaptive_blur_threshold is True
    assert config.enable_blur_rescue_sharpening is False
    assert config.blur_rescue_sharpen_strength == 0.3
    assert config.blur_rescue_sharpen_sigma == 1.2
    assert config.frame_selection_window_size == 3
    assert config.motion_model == "homography"
    assert config.feather_blend_kernel == 11
    assert config.seam_blur_kernel == 7
    assert config.seam_band_width == 9
    assert config.enable_photometric_normalization is False
    assert config.photometric_smoothing == 0.8
    assert config.overlap_sharpness_weight == 0.4
    assert config.enable_narrow_gap_fill is False
    assert config.max_narrow_gap_width == 3
    assert config.photo_crop_margin_px == 5
    assert config.photo_mode is True
    assert config.enable_final_sharpening is False
    assert config.final_sharpen_strength == 0.5
    assert config.final_sharpen_sigma == 1.5
    assert config.enable_feature_fallback is False
    assert config.fallback_feature_backend == "orb"
    assert config.fallback_min_chain_length == 5
    assert config.enable_sampling_fallback is False
    assert config.fallback_sampling_step == 4

    output = capsys.readouterr().out
    assert "Panorama saved to: out.png" in output
    assert "Selected frames: 2" in output
    assert "Rejected frames: 1" in output
    assert "Feature backend: sift" in output
    assert "Fallback used: True" in output
    assert "Video FPS: 25.0" in output


def test_build_command_validates_cli_overrides() -> None:
    with pytest.raises(ValueError, match="downscale must be between 0 and 1"):
        cli_main.build_command(_build_args(downscale=0.0))


def test_inspect_video_command_prints_metadata(monkeypatch, capsys) -> None:
    events: list[str] = []

    class _FakeSource:
        def __init__(self, video_path: str, config: PanoramaConfig) -> None:
            assert video_path == "input.mp4"
            assert isinstance(config, PanoramaConfig)

        def open(self) -> VideoMetadata:
            events.append("open")
            return VideoMetadata(path=Path("input.mp4"), fps=30.0, frame_count=50, width=1280, height=720)

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(video_module, "OpenCVVideoSource", _FakeSource)

    result = cli_main.inspect_video_command(argparse.Namespace(video_path="input.mp4"))

    assert result == 0
    assert events == ["open", "close"]
    output = capsys.readouterr().out
    assert "path=input.mp4" in output
    assert "fps=30.0" in output
    assert "frame_count=50" in output
    assert "width=1280" in output
    assert "height=720" in output


def test_export_config_command_saves_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "config.json"

    result = cli_main.export_config_command(argparse.Namespace(output_path=output_path))

    assert result == 0
    assert output_path.exists()
    assert "Config saved to:" in capsys.readouterr().out


def test_create_parser_routes_supported_subcommands() -> None:
    parser = cli_main.create_parser()

    build_args = parser.parse_args(
        [
            "build", "video.mp4", "out.png", "--photo-mode", "--sampling-step", "7",
            "--no-narrow-gap-fill", "--max-narrow-gap-width", "3", "--photo-crop-margin-px", "5",
        ]
    )
    inspect_args = parser.parse_args(["inspect-video", "video.mp4"])
    export_args = parser.parse_args(["export-config"])

    assert build_args.func is cli_main.build_command
    assert build_args.photo_mode is True
    assert build_args.sampling_step == 7
    assert build_args.no_narrow_gap_fill is True
    assert build_args.max_narrow_gap_width == 3
    assert build_args.photo_crop_margin_px == 5
    assert inspect_args.func is cli_main.inspect_video_command
    assert export_args.func is cli_main.export_config_command


def test_main_dispatches_selected_command(monkeypatch) -> None:
    parser = cli_main.create_parser()
    args = argparse.Namespace(func=lambda parsed: 17)

    monkeypatch.setattr(cli_main, "create_parser", lambda: parser)
    monkeypatch.setattr(parser, "parse_args", lambda: args)

    assert cli_main.main() == 17

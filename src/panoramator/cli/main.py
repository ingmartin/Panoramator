from __future__ import annotations

import argparse
from pathlib import Path

from panoramator.application.use_cases import PanoramaBuilder
from panoramator.cli import about as cli_about
from panoramator.cli.commands import (
    add_build_arguments,
    add_unwrap_arguments,
    apply_build_overrides,
    apply_unwrap_overrides,
)
from panoramator.config.models import PanoramaConfig
from panoramator.object_unwrap import (
    ObjectUnwrapper,
    UnwrapConfig,
    UnwrapStatus,
)


class PanoramatorHelpFormatter(argparse.RawTextHelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=100, width=200)


def format_version_screen() -> str:
    return cli_about.format_version_screen()


def format_root_help(parser: argparse.ArgumentParser) -> str:
    return "\n\n".join(
        [
            parser.format_help().rstrip(),
            "Examples:\n"
            "  panoramator build video.mp4 output.png\n"
            "  panoramator build video.mp4 output.png --capture-mode rotation --horizontal-fov-degrees 70\n"
            "  panoramator unwrap video.mp4 surface.png --surface auto --allow-partial\n"
            "  panoramator unwrap video.mp4 surface.webp --publish-profile coverage_first --photo-mode",
            "Command help:\n"
            "  panoramator build -h\n"
            "  panoramator unwrap -h\n"
            "  panoramator inspect-video -h\n"
            "  panoramator export-config -h",
        ]
    )


def build_command(args: argparse.Namespace) -> int:
    config = PanoramaConfig()
    if args.config:
        config = PanoramaConfig.from_json(args.config)

    apply_build_overrides(config, args)

    config.validate()

    result = PanoramaBuilder(config).build_from_video(args.video_path, args.output_path)
    if result.diagnostics.status != "orbit_not_supported_reliably":
        print(f"Panorama saved to: {args.output_path}")
    else:
        print("Panorama was not written: orbit capture is not supported for reliable scene panoramas; use unwrap for object-surface output")
    print(f"Selected frames: {len(result.diagnostics.selected_frames)}")
    print(f"Rejected frames: {len(result.diagnostics.rejected_frames)}")
    print(f"Feature backend: {result.diagnostics.feature_backend}")
    print(f"Sampling step: {result.diagnostics.sampling_step}")
    print(f"Fallback used: {result.diagnostics.fallback_used}")
    print(f"Capture mode: {result.diagnostics.capture_mode}")
    print(f"Projection: {result.diagnostics.projection}")
    print(f"Status: {result.diagnostics.status}")
    print(f"Video FPS: {result.metadata.fps}")
    return 0


def inspect_video_command(args: argparse.Namespace) -> int:
    from panoramator.io.video import OpenCVVideoSource

    source = OpenCVVideoSource(args.video_path, PanoramaConfig())
    metadata = source.open()
    source.close()
    print(f"path={metadata.path}")
    print(f"fps={metadata.fps}")
    print(f"frame_count={metadata.frame_count}")
    print(f"width={metadata.width}")
    print(f"height={metadata.height}")
    return 0


def unwrap_command(args: argparse.Namespace) -> int:
    config = UnwrapConfig()
    if getattr(args, "config", None):
        config = UnwrapConfig.from_json(args.config)

    apply_unwrap_overrides(config, args)
    config.validate()
    result = ObjectUnwrapper(config).unwrap_video(args.video_path, args.output_path)
    print(f"Status: {result.diagnostics.status.value}")
    print(f"Surface: {result.diagnostics.surface_kind.value}")
    if result.output_path is not None:
        print(f"Unwrap saved to: {result.output_path}")
    print(f"Selected frames: {len(result.diagnostics.selected_frames)}")
    print(f"Sampling step: {config.sampling_step}")
    print(result.diagnostics.message)
    if result.diagnostics.recommendation:
        print(f"Recommendation: {result.diagnostics.recommendation}")
    return 0 if result.output_path is not None and result.diagnostics.status in {UnwrapStatus.OK, UnwrapStatus.PARTIAL_SURFACE} else 2


def export_config_command(args: argparse.Namespace) -> int:
    PanoramaConfig().save(args.output_path)
    print(f"Config saved to: {args.output_path}")
    return 0


def _create_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
) -> argparse.ArgumentParser:
    return subparsers.add_parser(
        name,
        help=help_text,
        formatter_class=PanoramatorHelpFormatter,
    )


def _configure_build_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    build = _create_subparser(subparsers, "build", "Build panorama from video")
    add_build_arguments(build)
    build.set_defaults(func=build_command)
    return build


def _configure_unwrap_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    unwrap = _create_subparser(subparsers, "unwrap", "Build a surface map from video")
    add_unwrap_arguments(unwrap)
    unwrap.set_defaults(func=unwrap_command)
    return unwrap


def _configure_inspect_video_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    inspect_video = _create_subparser(subparsers, "inspect-video", "Inspect video metadata")
    inspect_video.add_argument("video_path", help="Input video file")
    inspect_video.set_defaults(func=inspect_video_command)
    return inspect_video


def _configure_export_config_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    export_config = _create_subparser(subparsers, "export-config", "Export default config")
    export_config.add_argument("output_path", nargs="?", default=str(Path("panoramator.config.json")), help="Destination JSON config path")
    export_config.set_defaults(func=export_config_command)
    return export_config


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panoramator",
        description="Build panoramas and object surface unwraps from video.",
        formatter_class=PanoramatorHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true", help="Show help for all major commands")
    parser.add_argument("--version", action="store_true", help="Show runtime version information")
    subparsers = parser.add_subparsers(dest="command")

    _configure_build_parser(subparsers)
    _configure_unwrap_parser(subparsers)
    _configure_inspect_video_parser(subparsers)
    _configure_export_config_parser(subparsers)
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    if getattr(args, "version", False):
        print(format_version_screen())
        return 0
    if getattr(args, "help", False) or not hasattr(args, "func"):
        print(format_root_help(parser))
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

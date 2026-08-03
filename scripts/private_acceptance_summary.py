from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panoramator.application.use_cases import PanoramaBuilder
from panoramator.config.models import PanoramaConfig
from panoramator.object_unwrap import ObjectUnwrapper, PublishProfile, UnwrapConfig

ROOT = Path("tests/private")


def _orbit_cases() -> list[dict[str, object]]:
    payload = json.loads((ROOT / "orbit" / "private_orbit_expectations.json").read_text(encoding="utf-8"))
    defaults = payload["defaults"]
    return [{**defaults, **case, "suite": "orbit"} for case in payload["cases"]]


def _panorama_cases(kind: str) -> list[dict[str, object]]:
    payload = json.loads((ROOT / kind / "private_panorama_expectations.json").read_text(encoding="utf-8"))
    defaults = payload["defaults"]
    return [{**defaults, **case, "suite": kind} for case in payload["cases"]]


def _print_header() -> None:
    print("suite\tpath\tresult\tstatus\tmode_or_surface\tprojection_or_frames\tselected_frames\tsampling_or_pairs")


def _run_orbit(case: dict[str, object], output_root: Path) -> None:
    video_path = ROOT / "orbit" / str(case["path"])
    output_path = output_root / "orbit" / f"{video_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = ObjectUnwrapper(
        UnwrapConfig(
            allow_partial=bool(case["allow_partial"]),
            save_debug_artifacts=bool(case["save_debug_artifacts"]),
            publish_profile=PublishProfile(str(case["publish_profile"])),
        )
    ).unwrap_video(video_path, output_path)
    measurements = result.diagnostics.measurements
    print(
        "\t".join(
            [
                "orbit",
                str(case["path"]),
                "ok",
                result.diagnostics.status.value,
                result.diagnostics.surface_kind.value,
                f"{float(measurements.get('surface_coverage_fraction', 0.0)):.6f}",
                str(len(result.diagnostics.selected_frames)),
                str(measurements.get("accepted_pose_pairs")),
            ]
        )
    )


def _run_panorama(kind: str, case: dict[str, object], output_root: Path) -> None:
    video_path = ROOT / kind / str(case["path"])
    output_path = output_root / kind / f"{video_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = PanoramaBuilder(PanoramaConfig(save_debug_artifacts=bool(case["save_debug_artifacts"]))).build_from_video(
            video_path, output_path
        )
    except RuntimeError as exc:  # pragma: no cover - reporting path
        print(
            "\t".join(
                [
                    kind,
                    str(case["path"]),
                    "error",
                    type(exc).__name__,
                    "-",
                    "-",
                    "-",
                    "-",
                ]
            )
        )
        return
    diagnostics = result.diagnostics
    print(
        "\t".join(
            [
                kind,
                str(case["path"]),
                "ok",
                diagnostics.status,
                diagnostics.capture_mode,
                diagnostics.projection,
                str(len(diagnostics.selected_frames)),
                str(diagnostics.sampling_step),
            ]
        )
    )


def main() -> int:
    if os.environ.get("PANORAMATOR_RUN_PRIVATE_VIDEO") != "1":
        print("Set PANORAMATOR_RUN_PRIVATE_VIDEO=1 to run private acceptance summary.")
        return 2
    output_root = Path("/tmp/panoramator-private-summary")
    _print_header()
    for case in _orbit_cases():
        _run_orbit(case, output_root)
    for kind in ("linear", "rotate"):
        for case in _panorama_cases(kind):
            _run_panorama(kind, case, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

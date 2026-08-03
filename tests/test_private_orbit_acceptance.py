from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from panoramator.object_unwrap import ObjectUnwrapper, PublishProfile, UnwrapConfig

pytestmark = pytest.mark.private_video

PRIVATE_ROOT = Path("tests/private/orbit")
EXPECTATIONS_PATH = PRIVATE_ROOT / "private_orbit_expectations.json"


def _load_cases() -> list[dict[str, object]]:
    if not EXPECTATIONS_PATH.exists():
        return []
    payload = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    defaults = payload["defaults"]
    return [{**defaults, **case} for case in payload["cases"]]


def _private_video_enabled() -> bool:
    return os.environ.get("PANORAMATOR_RUN_PRIVATE_VIDEO") == "1"


def _private_video_ready() -> bool:
    return PRIVATE_ROOT.exists() and any(PRIVATE_ROOT.rglob("*.mp4"))


def _case_id(case: dict[str, object]) -> str:
    return str(case["path"])


@pytest.mark.skipif(
    not EXPECTATIONS_PATH.exists(),
    reason="private orbit manifest is not present in this checkout",
)
def test_private_orbit_manifest_surface_layout_is_consistent() -> None:
    for case in _load_cases():
        relative_path = Path(str(case["path"]))
        assert relative_path.parts[0] in {"curved", "cylinder"}
        assert relative_path.parts[0] == str(case["fixture_surface_kind"])


@pytest.mark.skipif(
    not _private_video_enabled() or not _private_video_ready(),
    reason="private orbit acceptance is disabled; set PANORAMATOR_RUN_PRIVATE_VIDEO=1 and provide local videos",
)
@pytest.mark.parametrize("case", _load_cases(), ids=_case_id)
def test_private_orbit_unwrap_acceptance(case: dict[str, object], tmp_path: Path) -> None:
    video_path = PRIVATE_ROOT / str(case["path"])
    assert video_path.exists(), f"Missing private video fixture: {video_path}"

    config = UnwrapConfig(
        allow_partial=bool(case["allow_partial"]),
        save_debug_artifacts=bool(case["save_debug_artifacts"]),
        publish_profile=PublishProfile(str(case["publish_profile"])),
    )
    output_path = tmp_path / f"{video_path.stem}.png"
    result = ObjectUnwrapper(config).unwrap_video(video_path, output_path)
    measurements = result.diagnostics.measurements

    allowed_statuses = case.get("allowed_statuses")
    if allowed_statuses is None:
        assert result.diagnostics.status.value == case["expected_status"]
    else:
        assert result.diagnostics.status.value in allowed_statuses
    assert result.diagnostics.surface_kind.value == case["expected_surface_kind"]
    assert len(result.diagnostics.selected_frames) >= int(case["min_selected_frames"])

    coverage = float(measurements.get("surface_coverage_fraction", 0.0))
    assert coverage >= float(case["min_surface_coverage_fraction"])
    assert coverage <= float(case["max_surface_coverage_fraction"])

    expect_output = case.get("expect_output")
    if expect_output is True:
        assert result.output_path == output_path
        assert output_path.exists()
        assert result.coverage is not None
    elif expect_output is False:
        assert result.output_path is None
        assert not output_path.exists()

from __future__ import annotations

import os
from pathlib import Path

import pytest

from panoramator.object_unwrap import (
    ObjectUnwrapper,
    SurfaceKind,
    SurfaceOutputMode,
    UnwrapConfig,
)

pytestmark = pytest.mark.private_video

CURVED_CASES = (
    Path("tests/private/orbit/curved/VID20260728142100.mp4"),
    Path("tests/private/orbit/curved/VID20260728142150.mp4"),
    Path("tests/private/orbit/curved/VID20260729183902.mp4"),
    Path("tests/private/orbit/curved/VID20260729185124.mp4"),
    Path("tests/private/orbit/curved/VID20260731163059.mp4"),
)
CYLINDER_CASES = tuple(sorted(Path("tests/private/orbit/cylinder").glob("*.mp4")))
PUBLIC_CURVED_CASE = Path("docs/public-demo/VID20260805121156.mp4")


def _enabled() -> bool:
    return os.environ.get("PANORAMATOR_RUN_PRIVATE_VIDEO") == "1"


def _ready(paths: tuple[Path, ...]) -> bool:
    return all(path.exists() for path in paths)


@pytest.mark.skipif(
    not _enabled() or not _ready(CURVED_CASES),
    reason="observed-surface video acceptance is disabled; set PANORAMATOR_RUN_PRIVATE_VIDEO=1",
)
@pytest.mark.parametrize("video_path", CURVED_CASES, ids=lambda path: path.stem)
def test_curved_observed_surface_acceptance(video_path: Path, tmp_path: Path) -> None:
    config = UnwrapConfig(
        surface_kind=SurfaceKind.CURVED,
        surface_output_mode=SurfaceOutputMode.OBSERVED_SURFACE,
        allow_partial=True,
        save_debug_artifacts=True,
    )
    result = ObjectUnwrapper(config).unwrap_video(video_path, tmp_path / f"{video_path.stem}.png")
    measurements = result.diagnostics.measurements

    assert result.diagnostics.status.value in {"observed_surface", "partial_surface"}
    assert measurements["surface_builder"] == "curved_local_atlas"
    assert measurements["curved_surface_quality_gate_passed"] == 1
    synthetic_fraction = measurements["surface_synthetic_fraction"]
    assert isinstance(synthetic_fraction, (int, float))
    assert synthetic_fraction <= 0.05
    assert result.output_path is not None
    assert result.output_path.exists()


@pytest.mark.skipif(
    not _enabled() or not _ready(CYLINDER_CASES),
    reason="observed-surface video acceptance is disabled; set PANORAMATOR_RUN_PRIVATE_VIDEO=1",
)
@pytest.mark.parametrize("video_path", CYLINDER_CASES, ids=lambda path: path.stem)
def test_cylindrical_observed_surface_matrix(video_path: Path, tmp_path: Path) -> None:
    config = UnwrapConfig(
        surface_kind=SurfaceKind.CYLINDRICAL,
        surface_output_mode=SurfaceOutputMode.OBSERVED_SURFACE,
        allow_partial=True,
        save_debug_artifacts=True,
    )
    result = ObjectUnwrapper(config).unwrap_video(video_path, tmp_path / f"{video_path.stem}.png")
    measurements = result.diagnostics.measurements

    assert result.diagnostics.surface_kind is SurfaceKind.CYLINDRICAL
    assert "product_surface_quality_gate_passed" in measurements
    assert "surface_observed_coverage_fraction" in measurements
    assert "surface_synthetic_fraction" in measurements
    for key in ("surface_observed", "surface_synthetic", "surface_unknown", "surface_source"):
        assert any(path.endswith(f"/{key}.png") for path in result.diagnostics.output_files)
    assert result.diagnostics.status.value in {"observed_surface", "partial_surface", "unstable_camera_geometry"}
    if video_path.name == "VID20260731173355.mp4":
        assert measurements["product_surface_quality_gate_passed"] == 1
        assert measurements["observed_surface_selected_candidate"] == "product_surface"


@pytest.mark.skipif(
    not _enabled() or not PUBLIC_CURVED_CASE.exists(),
    reason="public observed-surface regression is disabled; set PANORAMATOR_RUN_PRIVATE_VIDEO=1",
)
def test_public_curved_observed_surface_regression(tmp_path: Path) -> None:
    config = UnwrapConfig(
        surface_kind=SurfaceKind.CURVED,
        surface_output_mode=SurfaceOutputMode.OBSERVED_SURFACE,
        allow_partial=True,
        save_debug_artifacts=True,
    )
    result = ObjectUnwrapper(config).unwrap_video(PUBLIC_CURVED_CASE, tmp_path / "public-curved.png")

    assert result.diagnostics.status.value in {"observed_surface", "partial_surface"}
    assert result.diagnostics.measurements["surface_builder"] == "curved_local_atlas"

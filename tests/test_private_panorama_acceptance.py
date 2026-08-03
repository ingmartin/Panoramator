from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from panoramator.application.use_cases import PanoramaBuilder
from panoramator.config.models import PanoramaConfig


def _private_video_enabled() -> bool:
    return os.environ.get("PANORAMATOR_RUN_PRIVATE_VIDEO") == "1"


def _private_root(kind: str) -> Path:
    return Path("tests/private") / kind


def _expectations_path(kind: str) -> Path:
    return _private_root(kind) / "private_panorama_expectations.json"


def _load_cases(kind: str) -> list[dict[str, object]]:
    path = _expectations_path(kind)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    defaults = payload["defaults"]
    return [{**defaults, **case, "suite_kind": kind} for case in payload["cases"]]


def _private_video_ready(kind: str) -> bool:
    root = _private_root(kind)
    return root.exists() and any(path.is_file() for path in root.iterdir())


def _case_id(case: dict[str, object]) -> str:
    return f"{case['suite_kind']}/{case['path']}"


def _suite_mark(kind: str):
    return getattr(pytest.mark, f"private_{kind}")


def _run_case(case: dict[str, object], tmp_path: Path) -> None:
    suite_kind = str(case["suite_kind"])
    video_path = _private_root(suite_kind) / str(case["path"])
    assert video_path.exists(), f"Missing private video fixture: {video_path}"

    output_path = tmp_path / f"{video_path.stem}.png"
    builder = PanoramaBuilder(PanoramaConfig(save_debug_artifacts=bool(case["save_debug_artifacts"])))

    if case["expected_result"] == "error":
        with pytest.raises(Exception) as exc_info:
            builder.build_from_video(video_path, output_path)
        assert type(exc_info.value).__name__ == case["expected_exception_type"]
        assert str(case["expected_exception_message"]) in str(exc_info.value)
        assert not output_path.exists()
        return

    result = builder.build_from_video(video_path, output_path)
    diagnostics = result.diagnostics
    assert diagnostics.status == case["expected_status"]
    assert diagnostics.capture_mode == case["expected_capture_mode"]
    assert diagnostics.projection == case["expected_projection"]
    assert len(diagnostics.selected_frames) >= int(case["min_selected_frames"])
    assert diagnostics.sampling_step == int(case["expected_sampling_step"])
    assert diagnostics.fallback_used is bool(case["expected_fallback_used"])
    assert result.image is not None
    assert output_path.exists()


@pytest.mark.private_video
@pytest.mark.private_linear
@pytest.mark.skipif(
    not _expectations_path("linear").exists(),
    reason="private linear manifest is not present in this checkout",
)
def test_private_linear_manifest_exists_and_has_cases() -> None:
    cases = _load_cases("linear")
    assert cases
    assert all(str(case["path"]) for case in cases)


@pytest.mark.private_video
@pytest.mark.private_rotate
@pytest.mark.skipif(
    not _expectations_path("rotate").exists(),
    reason="private rotate manifest is not present in this checkout",
)
def test_private_rotate_manifest_exists_and_has_cases() -> None:
    cases = _load_cases("rotate")
    assert cases
    assert all(str(case["path"]) for case in cases)


@_suite_mark("linear")
@pytest.mark.private_video
@pytest.mark.skipif(
    not _private_video_enabled() or not _private_video_ready("linear"),
    reason="private linear acceptance is disabled; set PANORAMATOR_RUN_PRIVATE_VIDEO=1 and provide local videos",
)
@pytest.mark.parametrize("case", _load_cases("linear"), ids=_case_id)
def test_private_linear_panorama_acceptance(case: dict[str, object], tmp_path: Path) -> None:
    _run_case(case, tmp_path)


@_suite_mark("rotate")
@pytest.mark.private_video
@pytest.mark.skipif(
    not _private_video_enabled() or not _private_video_ready("rotate"),
    reason="private rotate acceptance is disabled; set PANORAMATOR_RUN_PRIVATE_VIDEO=1 and provide local videos",
)
@pytest.mark.parametrize("case", _load_cases("rotate"), ids=_case_id)
def test_private_rotate_panorama_acceptance(case: dict[str, object], tmp_path: Path) -> None:
    _run_case(case, tmp_path)

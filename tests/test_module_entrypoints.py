from __future__ import annotations

import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from panoramator.cli import about as cli_about
from panoramator.domain import interfaces

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_domain_interfaces_module_loads_protocols() -> None:
    assert hasattr(interfaces, "FeatureExtractor")
    assert hasattr(interfaces, "FeatureMatcher")
    assert hasattr(interfaces, "GeometryEstimator")
    assert hasattr(interfaces, "CanvasBuilder")


def test_main_module_raises_system_exit(monkeypatch) -> None:
    monkeypatch.setattr("panoramator.cli.main.main", lambda: 0)

    with pytest.raises(SystemExit, match="0"):
        runpy.run_module("panoramator.__main__", run_name="__main__")


def test_python_module_version_smoke_from_source_tree() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    completed = subprocess.run(
        [sys.executable, "-m", "panoramator", "--version"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"Panoramator v{cli_about.get_pyproject_version()}" in completed.stdout

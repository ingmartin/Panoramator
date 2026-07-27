from __future__ import annotations

import runpy

import pytest

from panoramator.domain import interfaces


def test_domain_interfaces_module_loads_protocols() -> None:
    assert hasattr(interfaces, "FeatureExtractor")
    assert hasattr(interfaces, "FeatureMatcher")
    assert hasattr(interfaces, "GeometryEstimator")
    assert hasattr(interfaces, "CanvasBuilder")


def test_main_module_raises_system_exit(monkeypatch) -> None:
    monkeypatch.setattr("panoramator.cli.main.main", lambda: 0)

    with pytest.raises(SystemExit, match="0"):
        runpy.run_module("panoramator.__main__", run_name="__main__")

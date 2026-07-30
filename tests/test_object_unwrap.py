from __future__ import annotations

import json

import cv2
import numpy as np

from panoramator.domain.models import Frame
from panoramator.object_unwrap.analyzer import AnalyzedFrame
from panoramator.object_unwrap.coverage import least_covered_seam
from panoramator.object_unwrap.cylinder.builder import CylinderUnwrapBuilder
from panoramator.object_unwrap.diagnostics import write_artifacts
from panoramator.object_unwrap.models import SurfaceKind, UnwrapConfig, UnwrapDiagnostics, UnwrapStatus


def _frame(index: int, shift: int) -> AnalyzedFrame:
    image = np.zeros((100, 80, 3), np.uint8)
    image[10:90, 15:65] = (20 + shift, 120, 220)
    cv2.line(image, (25 + shift % 20, 10), (25 + shift % 20, 89), (255, 255, 255), 2)
    mask = np.zeros((100, 80), np.uint8)
    mask[10:90, 15:65] = 255
    return AnalyzedFrame(Frame(index, float(index), image), mask, 100.0, (15, 10, 50, 80))


def test_cylinder_builder_creates_alpha_coverage_and_low_coverage_seam() -> None:
    image, coverage, model, measurements = CylinderUnwrapBuilder().build(
        [_frame(0, 0), _frame(1, 8), _frame(2, 16)],
        UnwrapConfig(surface_kind=SurfaceKind.CYLINDRICAL, output_width=200, output_height=80),
    )

    assert image.shape == (80, 200, 3)
    assert coverage.shape == (80, 200)
    assert 0 < measurements["coverage_fraction"] <= 1
    assert model.kind is SurfaceKind.CYLINDRICAL
    assert least_covered_seam(coverage) == 0


def test_artifacts_keep_png_in_diagnostic_file_list(tmp_path) -> None:
    output = tmp_path / "unwrap.png"
    diagnostics = UnwrapDiagnostics(UnwrapStatus.OK, "ok", "", SurfaceKind.CYLINDRICAL, output_files=[str(output)])
    files = write_artifacts(output, diagnostics, np.full((4, 4), 255, np.uint8))

    report = json.loads((tmp_path / "unwrap_diagnostics.json").read_text())
    assert str(output) in files
    assert report["output_files"] == files

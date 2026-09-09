from __future__ import annotations

import numpy as np

from panoramator.domain.models import Frame
from panoramator.object_unwrap.analyzer import AnalyzedFrame
from panoramator.object_unwrap.cylinder.renderer import (
    _fill_internal_vertical_gaps,
    render_inverse_cylindrical_atlas,
)
from panoramator.object_unwrap.gap_fill import interpolate_surface_gaps


def _frame(image: np.ndarray, geometry: np.ndarray, publish: np.ndarray) -> AnalyzedFrame:
    return AnalyzedFrame(
        Frame(0, 0.0, image),
        geometry,
        publish,
        100.0,
        (0, 0, image.shape[1], image.shape[0]),
    )


def test_cylindrical_renderer_keeps_geometry_details_outside_publish_band() -> None:
    image = np.full((48, 96, 3), (30, 80, 150), np.uint8)
    image[:12] = (220, 40, 30)
    geometry = np.full((48, 96), 255, np.uint8)
    publish = np.zeros_like(geometry)
    publish[12:36] = 255

    rendered, coverage, _source, _error, _measurements = render_inverse_cylindrical_atlas(
        [_frame(image, geometry, publish)] * 3,
        [0.0, 0.4, 0.8],
        output_height=48,
        output_width=120,
    )

    assert np.count_nonzero(coverage[:8]) > 0
    assert np.all(rendered[:8][coverage[:8] > 0] == (220, 40, 30))
    # The upper rows are outside the strict publish band but inside a selected
    # cylindrical wall column; they must remain published texture.
    assert np.all(coverage[:, 40] > 0)


def test_cylindrical_renderer_interpolates_short_internal_vertical_gaps() -> None:
    image = np.zeros((48, 1, 3), np.uint8)
    image[:20, 0] = (20, 40, 200)
    image[26:, 0] = (220, 180, 30)
    coverage = np.zeros((48, 1), np.uint8)
    coverage[:20, 0] = 255
    coverage[26:, 0] = 255
    source = np.ones((48, 1), np.uint16)

    filled = _fill_internal_vertical_gaps(image, coverage, source, max_gap=24)

    assert filled == 6
    assert np.all(coverage[20:26, 0] > 0)
    assert np.all((image[20:26, 0] > (20, 40, 30)).all(axis=1))


def test_interpolate_surface_gaps_fills_bounded_transparent_stripe() -> None:
    image = np.zeros((8, 20, 3), np.uint8)
    image[:, :6] = (10, 20, 30)
    image[:, 14:] = (110, 120, 130)
    coverage = np.zeros((8, 20), np.uint8)
    coverage[:, :6] = 255
    coverage[:, 14:] = 255
    source = np.zeros((8, 20), np.uint16)
    source[:, :6] = 1
    source[:, 14:] = 2

    filled = interpolate_surface_gaps(image, coverage, source, max_gap=8)

    assert filled == 8 * 8
    assert np.all(coverage[:, 6:14] > 0)
    assert np.all(image[:, 6] == (21, 31, 41))

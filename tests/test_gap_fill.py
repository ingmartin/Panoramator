from __future__ import annotations

import numpy as np

from panoramator.postprocess.gaps import fill_narrow_mask_gaps


def test_fill_narrow_mask_gaps_interpolates_an_enclosed_vertical_strip() -> None:
    image = np.zeros((2, 5, 3), dtype=np.uint8)
    image[:, 0] = 20
    image[:, 3] = 110
    mask = np.zeros((2, 5), dtype=np.uint8)
    mask[:, 0] = 255
    mask[:, 3] = 255

    filled, filled_mask, metrics = fill_narrow_mask_gaps(image, mask, max_width=2)

    assert np.all(filled_mask[:, :4] == 255)
    assert np.all(filled[:, 1] == 50)
    assert np.all(filled[:, 2] == 80)
    assert metrics == {"filled_runs": 2.0, "filled_pixels": 4.0}


def test_fill_narrow_mask_gaps_does_not_modify_external_or_wide_background() -> None:
    image = np.full((1, 8, 3), 17, dtype=np.uint8)
    mask = np.zeros((1, 8), dtype=np.uint8)
    mask[:, 2] = 255
    mask[:, 6] = 255

    filled, filled_mask, metrics = fill_narrow_mask_gaps(image, mask, max_width=2)

    assert np.array_equal(filled, image)
    assert np.array_equal(filled_mask, mask)
    assert metrics == {"filled_runs": 0.0, "filled_pixels": 0.0}

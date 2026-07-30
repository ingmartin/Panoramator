from __future__ import annotations

import cv2
import numpy as np

from panoramator.object_unwrap.cylinder.builder import CylinderUnwrapBuilder


def _reference_texture() -> np.ndarray:
    """A high-contrast continuous drawing exposes a seam or a doubled contour."""
    image = np.full((64, 200, 3), (38, 92, 160), np.uint8)
    cv2.line(image, (0, 43), (199, 16), (245, 245, 245), 3)
    cv2.ellipse(image, (98, 32), (30, 20), 0, 0, 360, (25, 35, 230), 3)
    cv2.putText(image, "UV", (72, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 220, 40), 2, cv2.LINE_AA)
    return image


def test_feature_mosaic_preserves_continuous_reference_across_source_boundaries() -> None:
    reference = _reference_texture()
    # These crops are the same physical surface seen in three overlapping
    # observations. Their centres correspond exactly to one global atlas.
    fragments = [
        (reference[:, 0:100].copy(), np.full((64, 100), 255, np.uint8)),
        (reference[:, 50:150].copy(), np.full((64, 100), 255, np.uint8)),
        (reference[:, 100:200].copy(), np.full((64, 100), 255, np.uint8)),
    ]

    mosaic, coverage, source, error = CylinderUnwrapBuilder._feature_mosaic(
        fragments, [50 / 199, 100 / 199, 150 / 199], min_angle=0.0, angle_span=1.0, atlas_width=200
    )

    assert np.array_equal(coverage, np.full((64, 200), 255, np.uint8))
    assert np.max(error) == 0
    # Feather blending of equal surface samples must preserve the drawing. A
    # one-level tolerance accounts for uint8 conversion after accumulation.
    assert int(np.max(np.abs(mosaic.astype(np.int16) - reference.astype(np.int16)))) <= 1
    # Ownership must change in overlap regions, but those changes must not add
    # a visible contour: this is the automated counterpart of checking
    # source.png beside pano2.png during manual acceptance.
    boundaries = np.flatnonzero(np.any(source[:, 1:] != source[:, :-1], axis=0)) + 1
    assert len(boundaries) >= 2
    for boundary in boundaries:
        assert float(np.mean(np.abs(mosaic[:, boundary].astype(np.int16) - mosaic[:, boundary - 1].astype(np.int16)))) < 35

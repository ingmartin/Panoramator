from __future__ import annotations

import itertools

import cv2
import numpy as np
import pytest

from panoramator.domain.models import Frame
from panoramator.object_unwrap.analyzer import Analysis, AnalyzedFrame
from panoramator.object_unwrap.models import SurfaceKind, SurfaceModel, UnwrapConfig
from panoramator.object_unwrap.product_surface import (
    ProductSurfaceBuilder,
    _compose,
    _owner_reentry_count,
    _phase_cell_bounds,
    _Registration,
    _registration_gap_map,
    _Strip,
    _vertical_measurements,
    _warp_strip_with_registration,
    cylindrical_surface_support_mask,
    extract_central_strip,
    find_seam,
    monotonic_offsets,
    photometric_normalize,
    register_strips,
    repack_registered_offsets,
)
from panoramator.object_unwrap.service import _SurfaceBuild


def _frame(index: int, colour: tuple[int, int, int] = (80, 100, 120)) -> AnalyzedFrame:
    image = np.full((64, 80, 3), colour, np.uint8)
    cv2.line(image, (30 + index, 8), (30 + index, 56), (255, 255, 255), 2)
    mask = np.zeros((64, 80), np.uint8)
    mask[8:57, 16 + index : 64 + index] = 255
    return AnalyzedFrame(Frame(index, float(index), image), mask, mask.copy(), 100.0, (16 + index, 8, 48, 49))


def _baseline() -> _SurfaceBuild:
    return _SurfaceBuild(
        np.full((64, 192, 3), 100, np.uint8),
        np.full((64, 192), 255, np.uint8),
        SurfaceModel(SurfaceKind.CURVED),
        {"surface_coverage_fraction": 0.2},
        {},
        True,
    )


def test_central_strip_extraction_does_not_use_bbox_edges() -> None:
    item = _frame(0)
    strip = extract_central_strip(item, item.publish_mask, 128)

    assert strip is not None
    assert strip.image.shape[:2] == strip.mask.shape
    assert strip.width < round(48 * 128 / 49)


def test_monotonic_offsets_keep_one_orbit_direction() -> None:
    strips = [extract_central_strip(_frame(index), _frame(index).publish_mask, 128) for index in range(5)]
    valid = [strip for strip in strips if strip is not None]

    offsets = monotonic_offsets(valid, 128)

    assert offsets[0] == 0.0
    assert all(right > left for left, right in itertools.pairwise(offsets))


def test_seam_search_prefers_low_conflict_column() -> None:
    current = np.zeros((32, 40, 3), np.uint8)
    candidate = np.full_like(current, 255)
    candidate[:, 18:22] = 0
    overlap = np.ones((32, 40), np.bool_)

    result = find_seam(current, candidate, overlap)

    assert result is not None
    path, score = result
    assert 15 <= float(np.median(path)) <= 24
    assert score < 0.8


def test_photometric_normalization_clamps_gain() -> None:
    current = np.full((16, 16, 3), 250, np.uint8)
    candidate = np.full_like(current, 5)
    normalized, gain, bias = photometric_normalize(current, candidate, np.ones((16, 16), np.bool_))

    assert 0.85 <= gain <= 1.18
    assert -24.0 <= bias <= 24.0
    assert int(normalized.max()) <= 255


def test_product_builder_publishes_quality_contract() -> None:
    frames = [_frame(index) for index in range(6)]
    result = ProductSurfaceBuilder().build(
        Analysis(frames, SurfaceKind.CURVED),
        UnwrapConfig(output_height=64, output_width=192),
        _baseline(),
    )

    assert "product_surface_coverage_fraction" in result.measurements
    assert "product_surface_rejected_reason" in result.measurements
    assert "product_surface_metrics" in result.artifacts
    assert "product_surface_candidate" in result.artifacts


def test_cylindrical_builder_keeps_strip_ratio_fixed() -> None:
    frames = [_frame(index) for index in range(6)]
    result = ProductSurfaceBuilder().build(
        Analysis(frames, SurfaceKind.CYLINDRICAL),
        UnwrapConfig(
            output_height=64,
            output_width=192,
            surface_kind=SurfaceKind.CYLINDRICAL,
        ),
        _baseline(),
    )

    selected = [
        item for item in result.artifacts["product_surface_vertical_frames"] if item.get("selected") == 1
    ]
    assert selected
    assert {item["selected_strip_ratio"] for item in selected} == {0.14}


def test_vertical_reference_rejects_half_and_double_height_frames() -> None:
    items: list[AnalyzedFrame] = []
    masks: list[np.ndarray] = []
    for index, height in enumerate((49, 49, 24, 98, 49)):
        image = np.full((128, 80, 3), 100, np.uint8)
        mask = np.zeros((128, 80), np.uint8)
        mask[16 : 16 + height, 16:64] = 255
        items.append(AnalyzedFrame(Frame(index, float(index), image), mask, mask.copy(), 100.0, (16, 16, 48, height)))
        masks.append(mask)

    reference, measurements = _vertical_measurements(items, masks, 128)

    assert reference.height == 49.0
    assert [item["vertical_height_valid"] for item in measurements] == [1, 1, 0, 0, 1]
    assert measurements[2]["vertical_scale"] == pytest.approx(24 / 49)
    assert measurements[3]["vertical_scale"] == pytest.approx(98 / 49)


def test_reference_strip_preserves_common_vertical_scale() -> None:
    items = []
    masks = []
    for index, height in enumerate((49, 49, 49)):
        image = np.full((96, 80, 3), 100, np.uint8)
        mask = np.zeros((96, 80), np.uint8)
        mask[12 : 12 + height, 16:64] = 255
        item = AnalyzedFrame(Frame(index, float(index), image), mask, mask.copy(), 100.0, (16, 12, 48, height))
        items.append(item)
        masks.append(mask)
    reference, _ = _vertical_measurements(items, masks, 128)

    strip = extract_central_strip(items[0], masks[0], 128, reference=reference)

    assert strip is not None
    assert strip.image.shape[:2] == (128, strip.width)
    assert strip.vertical_scale == pytest.approx(1.0)
    assert strip.frame_top_residual_px == pytest.approx(0.0)
    assert strip.frame_bottom_residual_px == pytest.approx(0.0)


def test_reference_strip_aligns_vertical_translation_without_rescaling_it() -> None:
    items = []
    masks = []
    for index, top in enumerate((10, 30)):
        image = np.full((96, 80, 3), 100, np.uint8)
        mask = np.zeros((96, 80), np.uint8)
        mask[top : top + 49, 16:64] = 255
        item = AnalyzedFrame(Frame(index, float(index), image), mask, mask.copy(), 100.0, (16, top, 48, 49))
        items.append(item)
        masks.append(mask)
    reference, _ = _vertical_measurements(items, masks, 128)

    first = extract_central_strip(items[0], masks[0], 128, reference=reference)
    second = extract_central_strip(items[1], masks[1], 128, reference=reference)

    assert first is not None and second is not None
    assert np.array_equal(first.mask, second.mask)
    assert first.vertical_scale == pytest.approx(1.0)
    assert second.vertical_scale == pytest.approx(1.0)
    assert second.frame_top_residual_px > 0.0


def test_registered_offsets_clamp_large_unobservable_gaps() -> None:
    first = extract_central_strip(_frame(0), _frame(0).publish_mask, 128, ratio=0.14)
    second = extract_central_strip(_frame(1), _frame(1).publish_mask, 128, ratio=0.14)

    assert first is not None and second is not None
    offsets, clamped = repack_registered_offsets([first, second], [0.0, 10_000.0])

    assert clamped == 1
    assert offsets[1] <= 0.88 * min(first.width, second.width)


def test_registration_affine_changes_local_strip_shape_without_double_translation() -> None:
    strip = extract_central_strip(_frame(0), _frame(0).publish_mask, 128, ratio=0.20)

    assert strip is not None
    identity = _warp_strip_with_registration(strip, (1, 0, 0, 0, 1, 0), 49.0, 128)
    scaled = _warp_strip_with_registration(strip, (1.12, 0, 0, 0, 1.12, 0), 49.0, 128)

    assert np.array_equal(identity.mask, strip.mask)
    assert identity.image.shape == strip.image.shape
    assert scaled.registration_affine is not None
    assert scaled.registration_affine[0] == pytest.approx(1.12)
    assert np.count_nonzero(scaled.mask) > 0


def test_registration_gap_map_keeps_unobserved_interval_explicit() -> None:
    first = extract_central_strip(_frame(0), _frame(0).publish_mask, 64, ratio=0.14)
    second = extract_central_strip(_frame(1), _frame(1).publish_mask, 64, ratio=0.14)

    assert first is not None and second is not None
    gaps = _registration_gap_map([first, second], [0.0, float(first.width + 10)], 64, first.width * 2 + 10)

    assert np.count_nonzero(gaps[:, first.width : first.width + 10]) == 64 * 10


def test_register_strips_builds_dense_monotonic_phase(monkeypatch) -> None:
    strips = [extract_central_strip(_frame(index), _frame(index).publish_mask, 128, ratio=0.14) for index in range(5)]
    valid = [strip for strip in strips if strip is not None]
    estimates = [
        _Registration(True, dx, abs(dx) * 128 / 49.0, 0.9, 20, 0.5, "", (1, 0, 0, 0, 1, 0))
        for dx in (4.0, 4.0, 16.0, 4.0)
    ]

    def fake_registration(left, right, output_height, reference_height):
        return estimates.pop(0)

    monkeypatch.setattr("panoramator.object_unwrap.product_surface.estimate_pairwise_registration", fake_registration)
    selected, offsets, records = register_strips(valid, 128, 49.0)

    assert len(selected) == len(valid)
    assert all(right > left for left, right in itertools.pairwise(offsets))
    phases = [float(record["phase_atlas_px"]) for record in records]
    assert all(right >= left for left, right in itertools.pairwise(phases))
    assert all(record["keyframe_selected"] == 1 for record in records)


def test_composer_does_not_allow_non_neighbour_owner_reentry() -> None:
    strips = []
    for index in range(3):
        image = np.full((16, 8, 3), 50 + index * 50, np.uint8)
        mask = np.full((16, 8), 255, np.uint8)
        strips.append(_Strip(image, mask, _frame(index), 8))

    _, _, owner, _, _, _, _, _ = _compose(strips, [0.0, 3.0, 6.0], 16, 14, np.zeros((16, 14, 3), np.uint8))

    assert _owner_reentry_count(owner) == 0


def test_composer_can_interpolate_transparent_registration_stripes() -> None:
    strips = []
    for index, colour in enumerate(((20, 40, 80), (120, 140, 180))):
        image = np.full((16, 6, 3), colour, np.uint8)
        mask = np.full((16, 6), 255, np.uint8)
        strips.append(_Strip(image, mask, _frame(index), 6))

    _, coverage, _, _, _, _, _, _ = _compose(
        strips,
        [0.0, 14.0],
        16,
        20,
        np.zeros((16, 20, 3), np.uint8),
        interpolate_gaps=True,
        max_interpolation_gap_px=8,
    )

    assert np.all(coverage > 0)


def test_phase_cells_are_disjoint_and_cover_the_atlas_span() -> None:
    strips = [extract_central_strip(_frame(index), _frame(index).publish_mask, 64, ratio=0.14) for index in range(3)]
    valid = [strip for strip in strips if strip is not None]
    cells = _phase_cell_bounds(valid, [0.0, 12.0, 24.0], 36)

    assert cells[0][0] == 0
    assert cells[-1][1] == 36
    assert all(left <= right for left, right in cells)
    assert all(right <= next_left for (_, right), (next_left, _) in itertools.pairwise(cells))


def test_cylindrical_support_mask_recovers_bright_pixels_inside_component_hull() -> None:
    image = np.zeros((64, 80, 3), np.uint8)
    image[28:36, 34:42] = 220
    geometry = np.zeros((64, 80), np.uint8)
    geometry[10:54, 16:64] = 255
    geometry[28:36, 34:42] = 0
    item = AnalyzedFrame(
        Frame(0, 0.0, image),
        geometry,
        geometry.copy(),
        100.0,
        (16, 10, 48, 44),
    )

    support, recovered = cylindrical_surface_support_mask(item)

    assert np.count_nonzero(recovered[28:36, 34:42]) > 0
    assert np.all(support[28:36, 34:42] > 0)


def test_photometric_normalization_preserves_near_white_highlights() -> None:
    current = np.full((16, 16, 3), 40, np.uint8)
    candidate = np.full_like(current, 200)

    normalized, _, _ = photometric_normalize(current, candidate, np.ones((16, 16), np.bool_))

    assert int(normalized.min()) >= 200

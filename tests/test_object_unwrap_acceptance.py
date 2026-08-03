from __future__ import annotations

import cv2
import numpy as np

from panoramator.domain.models import Frame
from panoramator.object_unwrap.analyzer import AnalyzedFrame
from panoramator.object_unwrap.cylinder.builder import CylinderUnwrapBuilder
from panoramator.object_unwrap.models import PublishProfile, UnwrapConfig
from panoramator.object_unwrap.planar_mosaic import build_planar_mosaic
from panoramator.object_unwrap.rectification import evaluate_mosaic_quality


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


def test_publish_profiles_keep_anchor_details_hard_while_softening_smooth_overlap() -> None:
    smooth_left = np.full((12, 12, 3), 80, np.uint8)
    smooth_left[:, 5:7] = 160
    smooth_left = cv2.GaussianBlur(smooth_left, (5, 5), 0)
    smooth_right = np.full((12, 12, 3), 200, np.uint8)
    smooth_mask = np.full((12, 12), 255, np.uint8)
    smooth_frames = [
        AnalyzedFrame(Frame(0, 0.0, smooth_left), smooth_mask, smooth_mask.copy(), 1.0, (0, 0, 12, 12)),
        AnalyzedFrame(Frame(1, 1.0, smooth_right), smooth_mask, smooth_mask.copy(), 4.0, (0, 0, 12, 12)),
    ]
    anchor_left = np.full((18, 18, 3), 230, np.uint8)
    anchor_right = anchor_left.copy()
    anchor_left[:, 8:10] = 0
    anchor_right[:, 7:9] = 255
    anchor_mask = np.full((18, 18), 255, np.uint8)
    anchor_frames = [
        AnalyzedFrame(Frame(0, 0.0, anchor_left), anchor_mask, anchor_mask.copy(), 1.0, (0, 0, 18, 18)),
        AnalyzedFrame(Frame(1, 1.0, anchor_right), anchor_mask, anchor_mask.copy(), 1.04, (0, 0, 18, 18)),
    ]
    shared_edges = [
        {
            "left_frame": 0,
            "right_frame": 1,
            "reason": "ok",
            "a00": 1.0,
            "a01": 0.0,
            "a02": 0.0,
            "a10": 0.0,
            "a11": 1.0,
            "a12": 0.0,
        }
    ]

    smooth_conservative = build_planar_mosaic(
        smooth_frames,
        shared_edges,
        output_height=12,
        publish_profile=PublishProfile.CONSERVATIVE,
    )
    smooth_coverage_first = build_planar_mosaic(
        smooth_frames,
        shared_edges,
        output_height=12,
        publish_profile=PublishProfile.COVERAGE_FIRST,
    )
    anchor_conservative = build_planar_mosaic(
        anchor_frames,
        shared_edges,
        output_height=18,
        publish_profile=PublishProfile.CONSERVATIVE,
    )
    anchor_coverage_first = build_planar_mosaic(
        anchor_frames,
        shared_edges,
        output_height=18,
        publish_profile=PublishProfile.COVERAGE_FIRST,
    )

    assert smooth_conservative is not None
    assert smooth_coverage_first is not None
    assert anchor_conservative is not None
    assert anchor_coverage_first is not None
    smooth_conservative_image, smooth_conservative_coverage, smooth_conservative_owner, _ = smooth_conservative
    smooth_coverage_first_image, smooth_coverage_first_coverage, smooth_coverage_first_owner, _ = smooth_coverage_first
    anchor_conservative_image, anchor_conservative_coverage, anchor_conservative_owner, _ = anchor_conservative
    anchor_coverage_first_image, anchor_coverage_first_coverage, anchor_coverage_first_owner, _ = anchor_coverage_first

    assert np.array_equal(smooth_conservative_coverage, smooth_coverage_first_coverage)
    assert np.count_nonzero(smooth_conservative_owner == 2) == smooth_conservative_owner.size
    assert np.count_nonzero(smooth_coverage_first_owner == 2) == smooth_coverage_first_owner.size
    assert float(np.mean(smooth_conservative_image[:, 2])) >= 199.0
    assert float(np.mean(smooth_coverage_first_image[:, 2])) > 80.0
    assert float(np.mean(smooth_coverage_first_image[:, 2])) < 200.0

    assert np.array_equal(anchor_conservative_coverage, anchor_coverage_first_coverage)
    assert np.count_nonzero(anchor_conservative_owner[:, 9] == 1) >= anchor_conservative_owner.shape[0] - 1
    assert np.count_nonzero(anchor_coverage_first_owner[:, 9] == 1) >= anchor_coverage_first_owner.shape[0] - 1
    assert float(np.mean(anchor_conservative_image[:, 9])) <= 1.0
    assert float(np.mean(anchor_coverage_first_image[:, 9])) <= 1.0


def test_coverage_first_profile_still_rejects_strong_anchor_conflicts() -> None:
    image = np.full((40, 64, 3), 235, np.uint8)
    image[8:32, 8:56:8] = 0
    coverage = np.full((40, 64), 255, np.uint8)
    source = np.tile(np.array([1, 2], np.uint16), (40, 32))
    error = np.zeros((40, 64), np.uint8)
    error[:, 1:] = np.where(source[:, 1:] != source[:, :-1], 96, 0)

    conservative_config = UnwrapConfig(publish_profile=PublishProfile.CONSERVATIVE)
    coverage_first_config = UnwrapConfig(publish_profile=PublishProfile.COVERAGE_FIRST)

    conservative_gate = evaluate_mosaic_quality(
        image,
        coverage,
        source,
        error,
        conservative_config.max_mosaic_boundary_mean_error,
        conservative_config.max_mosaic_boundary_severe_fraction,
        conservative_config.mosaic_boundary_severe_error,
        conservative_config.max_mosaic_boundary_severe_footprint
        * conservative_config.publish_profile_settings()["severe_footprint_multiplier"],
        max_anchor_conflict_footprint=conservative_config.max_mosaic_anchor_conflict_footprint
        * conservative_config.publish_profile_settings()["anchor_conflict_multiplier"],
        max_owner_instability=conservative_config.max_mosaic_owner_instability
        * conservative_config.publish_profile_settings()["owner_instability_multiplier"],
    )
    coverage_first_gate = evaluate_mosaic_quality(
        image,
        coverage,
        source,
        error,
        coverage_first_config.max_mosaic_boundary_mean_error,
        coverage_first_config.max_mosaic_boundary_severe_fraction,
        coverage_first_config.mosaic_boundary_severe_error,
        coverage_first_config.max_mosaic_boundary_severe_footprint
        * coverage_first_config.publish_profile_settings()["severe_footprint_multiplier"],
        max_anchor_conflict_footprint=coverage_first_config.max_mosaic_anchor_conflict_footprint
        * coverage_first_config.publish_profile_settings()["anchor_conflict_multiplier"],
        max_owner_instability=coverage_first_config.max_mosaic_owner_instability
        * coverage_first_config.publish_profile_settings()["owner_instability_multiplier"],
    )

    assert conservative_gate.passed is False
    assert coverage_first_gate.passed is False
    assert conservative_gate.anchor_conflict_footprint > 0.0
    assert coverage_first_gate.anchor_conflict_footprint > 0.0

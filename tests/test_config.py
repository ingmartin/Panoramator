import pytest

from panoramator.config.models import PanoramaConfig


def test_config_to_dict_contains_feature_backend() -> None:
    config = PanoramaConfig(feature_backend="sift", sampling_step=10)
    data = config.to_dict()
    assert data["feature_backend"] == "sift"
    assert data["sampling_step"] == 10


def test_config_disables_adaptive_blur_threshold_by_default() -> None:
    assert PanoramaConfig().adaptive_blur_threshold is False


def test_config_rejects_non_positive_downscale() -> None:
    with pytest.raises(ValueError, match="downscale must be between 0 and 1"):
        PanoramaConfig(downscale=0.0)


def test_config_rejects_non_positive_feature_downscale() -> None:
    with pytest.raises(ValueError, match="feature_downscale must be between 0 and 1"):
        PanoramaConfig(feature_downscale=0.0)


def test_config_normalizes_backend_names() -> None:
    config = PanoramaConfig(feature_backend="SIFT", fallback_feature_backend="ORB", motion_model="AFFINE")

    assert config.feature_backend == "sift"
    assert config.fallback_feature_backend == "orb"
    assert config.motion_model == "affine"


def test_config_rejects_negative_blur_rescue_sharpen_strength() -> None:
    with pytest.raises(ValueError, match="blur_rescue_sharpen_strength must be >= 0"):
        PanoramaConfig(blur_rescue_sharpen_strength=-0.1)


def test_config_rejects_non_positive_blur_rescue_sharpen_sigma() -> None:
    with pytest.raises(ValueError, match="blur_rescue_sharpen_sigma must be > 0"):
        PanoramaConfig(blur_rescue_sharpen_sigma=0.0)


def test_config_rejects_invalid_frame_selection_window_size() -> None:
    with pytest.raises(ValueError, match="frame_selection_window_size must be >= 1"):
        PanoramaConfig(frame_selection_window_size=0)


def test_config_rejects_negative_final_sharpen_strength() -> None:
    with pytest.raises(ValueError, match="final_sharpen_strength must be >= 0"):
        PanoramaConfig(final_sharpen_strength=-0.1)


def test_config_json_round_trip_preserves_user_settings(tmp_path) -> None:
    config = PanoramaConfig(
        sampling_step=7,
        feature_backend="sift",
        motion_model="homography",
        crop_result=False,
    )
    config_path = tmp_path / "panoramator.json"

    config.save(config_path)

    assert PanoramaConfig.from_json(config_path) == config


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"sampling_step": 0}, "sampling_step must be >= 1"),
        ({"max_frames": 0}, "max_frames must be >= 1"),
        ({"downscale": 1.1}, "downscale must be between 0 and 1"),
        ({"feature_downscale": 1.1}, "feature_downscale must be between 0 and 1"),
        ({"blur_threshold": -1.0}, "blur_threshold must be >= 0"),
        ({"adaptive_blur_percentile": 1.1}, "adaptive_blur_percentile must be between 0.0 and 1.0"),
        ({"min_difference": -1.0}, "min_difference must be >= 0"),
        ({"feature_backend": "akaze"}, "feature_backend must be one of: orb, sift"),
        ({"fallback_feature_backend": "akaze"}, "fallback_feature_backend must be one of: orb, sift"),
        ({"fallback_min_chain_length": 1}, "fallback_min_chain_length must be >= 2"),
        ({"fallback_sampling_step": 0}, "fallback_sampling_step must be >= 1"),
        ({"max_features": 0}, "max_features must be >= 1"),
        ({"ratio_test": 1.0}, "ratio_test must be between 0.0 and 1.0"),
        ({"min_match_count": 0}, "min_match_count must be >= 1"),
        ({"min_inlier_count": 0}, "min_inlier_count must be >= 1"),
        ({"min_inlier_ratio": 0.0}, "min_inlier_ratio must be between 0.0"),
        ({"motion_model": "rigid"}, "motion_model must be one of: translation, partial_affine, affine, homography"),
        ({"ransac_threshold": 0.0}, "ransac_threshold must be > 0"),
        ({"max_reprojection_error": 0.0}, "max_reprojection_error must be > 0"),
        ({"max_scale_deviation": -0.1}, "max_scale_deviation must be >= 0"),
        ({"max_rotation_degrees": -0.1}, "max_rotation_degrees must be >= 0"),
        ({"max_homography_corner_scale": 0.9}, "max_homography_corner_scale must be >= 1.0"),
        ({"max_canvas_width": 0}, "max_canvas_width and max_canvas_height must be >= 1"),
        ({"feather_blend_kernel": 0}, "feather_blend_kernel must be >= 1"),
        ({"seam_blur_kernel": 0}, "seam_blur_kernel must be >= 1"),
        ({"seam_band_width": 0}, "seam_band_width must be >= 1"),
        ({"photometric_smoothing": -0.1}, "photometric_smoothing must be between 0.0 and 1.0"),
        ({"overlap_sharpness_weight": -0.1}, "overlap_sharpness_weight must be >= 0"),
        ({"final_sharpen_sigma": 0.0}, "final_sharpen_sigma must be > 0"),
    ],
)
def test_config_rejects_invalid_pipeline_settings(settings: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PanoramaConfig(**settings)

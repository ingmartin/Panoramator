from panoramator.application.use_cases import PanoramaBuilder, _ChainBuildResult
from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import Frame, FrameQuality, SelectedFrame


def _selected_frame(index: int) -> SelectedFrame:
    return SelectedFrame(
        frame=Frame(index=index, timestamp_seconds=float(index), image=[]),  # type: ignore[arg-type]
        quality=FrameQuality(sharpness=1.0, difference_score=1.0, accepted=True, reason="ok"),
    )


def test_should_try_fallback_for_short_orb_chain() -> None:
    builder = PanoramaBuilder(PanoramaConfig(feature_backend="orb", enable_feature_fallback=True, fallback_min_chain_length=8))
    result = _ChainBuildResult(
        backend="orb",
        sampling_step=15,
        attempted_backends=["orb"],
        attempted_sampling_steps=[15],
        selected_frames=[],
        rejected_frames=[],
        filtered_frames=[_selected_frame(index) for index in range(7)],
        pairwise_homographies=[],
        pair_metrics=[],
    )

    assert builder._should_try_fallback(result) is True


def test_should_not_try_fallback_for_sufficient_chain() -> None:
    builder = PanoramaBuilder(PanoramaConfig(feature_backend="orb", enable_feature_fallback=True, fallback_min_chain_length=8))
    result = _ChainBuildResult(
        backend="orb",
        sampling_step=15,
        attempted_backends=["orb"],
        attempted_sampling_steps=[15],
        selected_frames=[],
        rejected_frames=[],
        filtered_frames=[_selected_frame(index) for index in range(8)],
        pairwise_homographies=[],
        pair_metrics=[],
    )

    assert builder._should_try_fallback(result) is False


def test_sampling_steps_include_denser_fallback() -> None:
    builder = PanoramaBuilder(PanoramaConfig(sampling_step=15, enable_sampling_fallback=True, fallback_sampling_step=8))
    assert builder._sampling_steps_to_try() == [15, 8]


def test_sampling_steps_skip_non_denser_fallback() -> None:
    builder = PanoramaBuilder(PanoramaConfig(sampling_step=8, enable_sampling_fallback=True, fallback_sampling_step=15))
    assert builder._sampling_steps_to_try() == [8]

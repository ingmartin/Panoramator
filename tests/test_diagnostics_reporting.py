from __future__ import annotations

import json

from panoramator.config.models import PanoramaConfig
from panoramator.diagnostics.reporting import write_diagnostics
from panoramator.domain.models import PanoramaDiagnostics


def test_write_diagnostics_creates_debug_files(tmp_path) -> None:
    diagnostics = PanoramaDiagnostics(
        selected_frames=[{"frame_index": 0}],
        validated_frames=[{"frame_index": 1}],
        rejected_frames=[{"frame_index": 2}],
        pair_metrics=[{"valid": True}],
        feature_backend="sift",
        sampling_step=8,
        fallback_used=True,
        fallback_attempted=True,
        attempted_backends=["orb", "sift"],
        attempted_sampling_steps=[15, 8],
        output_files=["out.png"],
    )

    output_files = write_diagnostics(tmp_path / "result.png", PanoramaConfig(), diagnostics)

    assert len(output_files) == 2
    config_path, _ = output_files
    assert "result_debug" in config_path
    report = json.loads((tmp_path / "result_debug" / "run.json").read_text(encoding="utf-8"))
    assert report["feature_backend"] == "sift"
    assert report["fallback_used"] is True
    assert report["fallback_attempted"] is True
    assert report["attempted_sampling_steps"] == [15, 8]

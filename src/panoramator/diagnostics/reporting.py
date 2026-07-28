from __future__ import annotations

import json
from pathlib import Path

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import PanoramaDiagnostics


def write_diagnostics(
    output_path: str | Path,
    config: PanoramaConfig,
    diagnostics: PanoramaDiagnostics,
) -> list[str]:
    output = Path(output_path)
    debug_dir = output.with_suffix("")
    debug_dir = debug_dir.parent / f"{debug_dir.name}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    config_path = debug_dir / "effective_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    report_path = debug_dir / "run.json"
    report = {
        "selected_frames": diagnostics.selected_frames,
        "validated_frames": diagnostics.validated_frames,
        "rejected_frames": diagnostics.rejected_frames,
        "pair_metrics": diagnostics.pair_metrics,
        "feature_backend": diagnostics.feature_backend,
        "sampling_step": diagnostics.sampling_step,
        "fallback_used": diagnostics.fallback_used,
        "fallback_attempted": diagnostics.fallback_attempted,
        "attempted_backends": diagnostics.attempted_backends,
        "attempted_sampling_steps": diagnostics.attempted_sampling_steps,
        "output_files": diagnostics.output_files,
        "capture_mode": diagnostics.capture_mode,
        "projection": diagnostics.projection,
        "strategy_confidence": diagnostics.strategy_confidence,
        "strategy_reason": diagnostics.strategy_reason,
        "strategy_measurements": diagnostics.strategy_measurements,
        "crop_policy": diagnostics.crop_policy,
        "crop_before_size": diagnostics.crop_before_size,
        "crop_after_size": diagnostics.crop_after_size,
        "crop_lost_area_fraction": diagnostics.crop_lost_area_fraction,
        "trajectory": diagnostics.trajectory,
        "seam_metrics": diagnostics.seam_metrics,
        "status": diagnostics.status,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return [str(config_path), str(report_path)]

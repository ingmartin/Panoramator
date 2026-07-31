from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .models import UnwrapConfig
from .models import UnwrapDiagnostics


def write_artifacts(
    output: str | Path,
    config: UnwrapConfig,
    diagnostics: UnwrapDiagnostics,
    coverage: np.ndarray | None,
    artifacts: dict[str, object] | None = None,
) -> list[str]:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = target.with_suffix("")
    debug_dir = debug_dir.parent / f"{debug_dir.name}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    config_path = debug_dir / "effective_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    report_path = debug_dir / "run.json"
    files = list(diagnostics.output_files)
    files.append(str(config_path))
    files.append(str(report_path))
    if coverage is not None:
        coverage_path = debug_dir / "coverage.png"
        if not cv2.imwrite(str(coverage_path), coverage):
            raise RuntimeError(f"Failed to write unwrap coverage: {coverage_path}")
        files.append(str(coverage_path))
    for name, artifact in (artifacts or {}).items():
        if isinstance(artifact, np.ndarray):
            artifact_path = debug_dir / f"{name}.png"
            if not cv2.imwrite(str(artifact_path), artifact):
                raise RuntimeError(f"Failed to write unwrap artifact: {artifact_path}")
        else:
            artifact_path = debug_dir / f"{name}.json"
            artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        files.append(str(artifact_path))
    diagnostics.output_files = files
    report_path.write_text(json.dumps(diagnostics.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return files

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .models import UnwrapDiagnostics


def write_artifacts(
    output: str | Path,
    diagnostics: UnwrapDiagnostics,
    coverage: np.ndarray | None,
    artifacts: dict[str, object] | None = None,
) -> list[str]:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path = target.with_name(f"{target.stem}_diagnostics.json")
    coverage_path = target.with_name(f"{target.stem}_coverage.png")
    files = list(diagnostics.output_files)
    files.append(str(diagnostics_path))
    if coverage is not None:
        cv2.imwrite(str(coverage_path), coverage)
        files.append(str(coverage_path))
    for name, artifact in (artifacts or {}).items():
        if isinstance(artifact, np.ndarray):
            artifact_path = target.with_name(f"{target.stem}_{name}.png")
            if not cv2.imwrite(str(artifact_path), artifact):
                raise RuntimeError(f"Failed to write unwrap artifact: {artifact_path}")
        else:
            artifact_path = target.with_name(f"{target.stem}_{name}.json")
            artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        files.append(str(artifact_path))
    diagnostics.output_files = files
    diagnostics_path.write_text(json.dumps(diagnostics.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return files

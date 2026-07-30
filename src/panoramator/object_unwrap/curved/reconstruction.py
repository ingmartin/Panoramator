from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .template import CurvedSurfaceTemplate


@dataclass(slots=True)
class MeshReconstruction:
    vertices: np.ndarray | None
    faces: np.ndarray | None
    confidence: float


def reconstruct_from_silhouettes(silhouettes: list[np.ndarray]) -> MeshReconstruction:
    """Refuse an ungrounded mesh rather than emit invented unobserved geometry."""
    confidence = CurvedSurfaceTemplate().compatibility(silhouettes) * min(1.0, len(silhouettes) / 16.0)
    # Free 3D fitting requires multi-view constraints not guaranteed by input.
    # Returning no mesh directs callers to the explicitly labelled side fallback.
    return MeshReconstruction(None, None, confidence)

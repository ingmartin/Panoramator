from __future__ import annotations

import numpy as np


def unwrap_mesh(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Produce a simple cylindrical UV initialization for a validated mesh."""
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must have shape (N, 3)")
    angles = np.arctan2(vertices[:, 2], vertices[:, 0])
    u = (angles + np.pi) / (2 * np.pi)
    height = vertices[:, 1]
    v = (height - height.min()) / max(float(height.max() - height.min()), 1e-9)
    return np.column_stack((u, v)).astype(np.float32)

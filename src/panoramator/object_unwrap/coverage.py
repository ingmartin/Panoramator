from __future__ import annotations

import numpy as np


def coverage_fraction(coverage: np.ndarray) -> float:
    return float(np.count_nonzero(coverage) / max(coverage.size, 1))


def least_covered_seam(coverage: np.ndarray) -> int:
    """Choose the centre of the least-observed angular column as the UV seam."""
    columns = np.count_nonzero(coverage, axis=0)
    return int(np.argmin(columns)) if columns.size else 0

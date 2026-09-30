"""Public errors raised by panorama builds."""

from __future__ import annotations


class PanoramaError(RuntimeError):
    """Base class for expected panorama-build failures."""


class PanoramaCancelled(PanoramaError):
    """Raised when a build is cancelled by its caller."""


class CanvasLimitExceeded(PanoramaError):
    """Raised when the requested panorama cannot fit the configured canvas."""

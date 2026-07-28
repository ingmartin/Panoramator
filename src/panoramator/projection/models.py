from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from panoramator.camera.models import CameraParameters


class Projection(ABC):
    name: str

    @abstractmethod
    def project_points(self, points: np.ndarray) -> np.ndarray:
        """Map reference-plane points to panorama surface coordinates."""

    @abstractmethod
    def unproject_points(self, points: np.ndarray) -> np.ndarray:
        """Map panorama surface coordinates back to reference-plane points."""

    @abstractmethod
    def valid_surface_points(self, points: np.ndarray) -> np.ndarray:
        """Return points that map to the front-facing source-camera plane."""


class PlanarProjection(Projection):
    name = "planar"

    def project_points(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64).copy()

    def unproject_points(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64).copy()

    def valid_surface_points(self, points: np.ndarray) -> np.ndarray:
        return np.ones(np.asarray(points).shape[:-1], dtype=bool)


class CylindricalProjection(Projection):
    name = "cylindrical"

    def __init__(self, camera: CameraParameters) -> None:
        self.camera = camera

    def project_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        x = points[..., 0] - self.camera.center_x
        y = points[..., 1] - self.camera.center_y
        f = self.camera.focal_length_px
        theta = np.arctan2(x, f)
        scale = np.sqrt(x * x + f * f)
        return np.stack((f * theta + self.camera.center_x, f * y / scale + self.camera.center_y), axis=-1)

    def unproject_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        f = self.camera.focal_length_px
        theta = (points[..., 0] - self.camera.center_x) / f
        x = f * np.tan(theta)
        y = (points[..., 1] - self.camera.center_y) / np.cos(theta)
        return np.stack((x + self.camera.center_x, y + self.camera.center_y), axis=-1)

    def valid_surface_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        theta = (points[..., 0] - self.camera.center_x) / self.camera.focal_length_px
        # ``tan(theta)`` repeats every pi.  Only the front-facing half-plane is
        # physically visible by the original pinhole camera; accepting the next
        # branch folds the first frame back into the far end of a panorama.
        return np.isfinite(theta) & (np.cos(theta) > 1e-8)


class SphericalProjection(CylindricalProjection):
    """A vertical spherical mapping; retained as an explicit supported surface."""

    name = "spherical"

    def project_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        x = points[..., 0] - self.camera.center_x
        y = points[..., 1] - self.camera.center_y
        f = self.camera.focal_length_px
        theta = np.arctan2(x, f)
        phi = np.arctan2(y, np.sqrt(x * x + f * f))
        return np.stack((f * theta + self.camera.center_x, f * phi + self.camera.center_y), axis=-1)

    def unproject_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        f = self.camera.focal_length_px
        theta = (points[..., 0] - self.camera.center_x) / f
        phi = (points[..., 1] - self.camera.center_y) / f
        x = f * np.tan(theta)
        y = np.tan(phi) * np.sqrt(x * x + f * f)
        return np.stack((x + self.camera.center_x, y + self.camera.center_y), axis=-1)


def create_projection(name: str, camera: CameraParameters) -> Projection:
    if name == "planar":
        return PlanarProjection()
    if name == "cylindrical":
        return CylindricalProjection(camera)
    if name == "spherical":
        return SphericalProjection(camera)
    raise ValueError(f"Unsupported projection: {name}")

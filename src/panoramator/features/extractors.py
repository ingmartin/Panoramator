from __future__ import annotations

import cv2
import numpy as np

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import FeatureSet, Frame


class ORBFeatureExtractor:
    backend_name = "orb"

    def __init__(self, config: PanoramaConfig) -> None:
        self.extractor = cv2.ORB_create(nfeatures=config.max_features)

    def extract(self, frame: Frame) -> FeatureSet:
        image = frame.feature_image if frame.feature_image is not None else frame.image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.extractor.detectAndCompute(gray, None)
        return FeatureSet(
            keypoints=_rescale_keypoints(keypoints or [], image.shape, frame.image.shape),
            descriptors=descriptors,
            backend=self.backend_name,
        )


class SIFTFeatureExtractor:
    backend_name = "sift"

    def __init__(self, config: PanoramaConfig) -> None:
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT is not available in this OpenCV build")
        self.extractor = cv2.SIFT_create(nfeatures=config.max_features)

    def extract(self, frame: Frame) -> FeatureSet:
        image = frame.feature_image if frame.feature_image is not None else frame.image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.extractor.detectAndCompute(gray, None)
        return FeatureSet(
            keypoints=_rescale_keypoints(keypoints or [], image.shape, frame.image.shape),
            descriptors=descriptors,
            backend=self.backend_name,
        )


def create_feature_extractor(config: PanoramaConfig):
    backend = config.feature_backend.lower()
    if backend == "orb":
        return ORBFeatureExtractor(config)
    if backend == "sift":
        return SIFTFeatureExtractor(config)
    raise ValueError(f"Unsupported feature backend: {config.feature_backend}")


def _rescale_keypoints(keypoints: list, source_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> list:
    if not keypoints:
        return []
    if source_shape[:2] == target_shape[:2]:
        return keypoints

    scale_x = target_shape[1] / max(source_shape[1], 1)
    scale_y = target_shape[0] / max(source_shape[0], 1)
    scaled = []
    for keypoint in keypoints:
        scaled.append(
            cv2.KeyPoint(
                x=float(keypoint.pt[0] * scale_x),
                y=float(keypoint.pt[1] * scale_y),
                size=float(keypoint.size * np.sqrt(scale_x * scale_y)),
                angle=float(keypoint.angle),
                response=float(keypoint.response),
                octave=int(keypoint.octave),
                class_id=int(keypoint.class_id),
            )
        )
    return scaled

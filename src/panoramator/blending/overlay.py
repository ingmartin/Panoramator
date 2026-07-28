from __future__ import annotations

import cv2
import numpy as np

from panoramator.config.models import PanoramaConfig


class AverageBlender:
    def __init__(self, config: PanoramaConfig) -> None:
        self.config = config

    def blend(
        self,
        warped_frames: list[np.ndarray],
        warped_masks: list[np.ndarray],
        frame_sharpnesses: list[float] | None = None,
        prefer_sharp_source: bool = False,
    ) -> np.ndarray:
        if not warped_frames:
            raise RuntimeError("No warped frames provided to blender")
        acc = np.zeros_like(warped_frames[0], dtype=np.float64)
        weight = np.zeros(warped_masks[0].shape, dtype=np.float64)
        sharpness_mean = 0.0
        if frame_sharpnesses:
            sharpness_mean = float(np.mean(np.asarray(frame_sharpnesses, dtype=np.float64)))
        best_frame = np.zeros_like(warped_frames[0]) if prefer_sharp_source else None
        best_score = np.zeros(warped_masks[0].shape, dtype=np.float64) if prefer_sharp_source else None
        for index, (frame, mask) in enumerate(zip(warped_frames, warped_masks, strict=True)):
            edge_weight = self._weight_map(mask)
            normalized = edge_weight * self._detail_weight(frame, mask)
            if frame_sharpnesses:
                normalized *= self._global_sharpness_weight(frame_sharpnesses[index], sharpness_mean)
            acc += frame.astype(np.float64) * normalized[..., None]
            weight += normalized
            if best_frame is not None and best_score is not None:
                selection_score = edge_weight
                if frame_sharpnesses:
                    selection_score = selection_score * self._global_sharpness_weight(
                        frame_sharpnesses[index], sharpness_mean
                    )
                replace = selection_score > best_score
                best_frame[replace] = frame[replace]
                best_score[replace] = selection_score[replace]
        if best_frame is not None:
            # A global surface cannot align different scene depths perfectly.  In
            # those overlap zones a single sharp source is preferable to averaging
            # two shifted edges into a permanently blurred double contour.
            return best_frame
        weight = np.maximum(weight, 1.0)
        blended = acc / weight[..., None]
        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
        return self._smooth_overlap_seams(blended_uint8, warped_masks)

    def _weight_map(self, mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8) * 255
        kernel = _odd_kernel(self.config.feather_blend_kernel)
        if kernel <= 1:
            return (binary > 0).astype(np.float64)
        radius = max(1.0, kernel / 2.0)
        distance = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)
        normalized = np.clip(distance / radius, 0.0, 1.0)
        return np.clip(normalized, 1e-6, 1.0) * (binary > 0)

    def _detail_weight(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        strength = float(self.config.overlap_sharpness_weight)
        if strength <= 0:
            return np.ones(mask.shape, dtype=np.float64)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detail = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        detail = cv2.GaussianBlur(detail, (0, 0), sigmaX=1.0, sigmaY=1.0)
        active = mask > 0
        if not np.any(active):
            return np.ones(mask.shape, dtype=np.float64)

        baseline = float(np.mean(detail[active]))
        if baseline <= 1e-6:
            return np.ones(mask.shape, dtype=np.float64)

        normalized = np.clip(detail / baseline, 0.5, 2.0)
        return 1.0 + (normalized - 1.0) * strength

    def _global_sharpness_weight(self, sharpness: float, mean_sharpness: float) -> float:
        strength = float(self.config.overlap_sharpness_weight)
        if strength <= 0 or mean_sharpness <= 1e-6:
            return 1.0
        normalized = np.clip(sharpness / mean_sharpness, 0.75, 1.5)
        return float(1.0 + (normalized - 1.0) * strength)

    def _smooth_overlap_seams(self, image: np.ndarray, warped_masks: list[np.ndarray]) -> np.ndarray:
        kernel = _odd_kernel(self.config.seam_blur_kernel)
        if kernel <= 1 or len(warped_masks) < 2:
            return image

        coverage = np.zeros(warped_masks[0].shape, dtype=np.uint16)
        for mask in warped_masks:
            coverage += (mask > 0).astype(np.uint16)

        overlap_mask = (coverage > 1).astype(np.uint8) * 255
        seam_mask = self._seam_boundary_mask(warped_masks, overlap_mask)
        if not np.any(seam_mask):
            return image

        seam_mask = cv2.GaussianBlur(seam_mask, (kernel, kernel), 0)
        seam_alpha = seam_mask.astype(np.float64) / 255.0
        blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)
        mixed = image.astype(np.float64) * (1.0 - seam_alpha[..., None]) + blurred.astype(np.float64) * seam_alpha[..., None]
        return np.clip(mixed, 0, 255).astype(np.uint8)

    def _seam_boundary_mask(self, warped_masks: list[np.ndarray], overlap_mask: np.ndarray) -> np.ndarray:
        band_width = max(1, int(self.config.seam_band_width))
        seam_mask: np.ndarray = np.zeros_like(overlap_mask, dtype=np.uint8)
        band_kernel = np.ones((band_width, band_width), dtype=np.uint8)

        for mask in warped_masks:
            binary = (mask > 0).astype(np.uint8) * 255
            if not np.any(binary):
                continue
            eroded = cv2.erode(binary, np.ones((3, 3), dtype=np.uint8), iterations=1)
            boundary = cv2.subtract(binary, eroded)
            boundary_band = cv2.dilate(boundary, band_kernel, iterations=1)
            seam_mask = cv2.bitwise_or(seam_mask, cv2.bitwise_and(boundary_band, overlap_mask))

        return seam_mask


def _odd_kernel(value: int) -> int:
    if value <= 1:
        return 1
    return value if value % 2 == 1 else value + 1

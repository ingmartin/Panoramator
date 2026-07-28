from __future__ import annotations

import cv2
import numpy as np

from panoramator.config.models import PanoramaConfig


class AverageBlender:
    def __init__(self, config: PanoramaConfig) -> None:
        self.config = config
        self.last_seam_metrics: list[dict[str, float]] = []
        self.last_photometric_metrics: list[dict[str, float]] = []
        self.last_global_photometric_metrics: list[dict[str, float]] = []

    def blend(
        self,
        warped_frames: list[np.ndarray],
        warped_masks: list[np.ndarray],
        frame_sharpnesses: list[float] | None = None,
        prefer_sharp_source: bool = False,
    ) -> np.ndarray:
        if not warped_frames:
            raise RuntimeError("No warped frames provided to blender")
        self.last_seam_metrics = []
        self.last_photometric_metrics = []
        self.last_global_photometric_metrics = []
        if prefer_sharp_source:
            return self._blend_with_seams(warped_frames, warped_masks, frame_sharpnesses)
        acc = np.zeros_like(warped_frames[0], dtype=np.float64)
        weight = np.zeros(warped_masks[0].shape, dtype=np.float64)
        sharpness_mean = 0.0
        if frame_sharpnesses:
            sharpness_mean = float(np.mean(np.asarray(frame_sharpnesses, dtype=np.float64)))
        for index, (frame, mask) in enumerate(zip(warped_frames, warped_masks, strict=True)):
            edge_weight = self._weight_map(mask)
            normalized = edge_weight * self._detail_weight(frame, mask)
            if frame_sharpnesses:
                normalized *= self._global_sharpness_weight(frame_sharpnesses[index], sharpness_mean)
            acc += frame.astype(np.float64) * normalized[..., None]
            weight += normalized
        weight = np.maximum(weight, 1.0)
        blended = acc / weight[..., None]
        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
        return self._smooth_overlap_seams(blended_uint8, warped_masks)

    def _blend_with_seams(
        self, frames: list[np.ndarray], masks: list[np.ndarray], sharpnesses: list[float] | None) -> np.ndarray:
        """Compose frames through a low-cost dynamic-programming seam.

        It selects one source in a conflict area instead of averaging shifted
        details; for the usual horizontal panorama the seam is a top-to-bottom
        path.  Non-overlapping pixels are copied unchanged.
        """
        if self.config.enable_global_photometric_normalization:
            frames, self.last_global_photometric_metrics = self._globally_align_frames(frames, masks)
        result = frames[0].copy()
        coverage = masks[0] > 0
        current_sharpness = sharpnesses[0] if sharpnesses else 1.0
        for index in range(1, len(frames)):
            incoming = frames[index]
            incoming_mask = masks[index] > 0
            overlap = coverage & incoming_mask
            only_incoming = incoming_mask & ~coverage
            new_coverage_ratio = float(int(only_incoming.sum()) / max(1, int(incoming_mask.sum())))
            incoming_sharpness = sharpnesses[index] if sharpnesses else 1.0
            if not np.any(only_incoming) and np.array_equal(incoming_mask, coverage):
                if incoming_sharpness > current_sharpness:
                    result[overlap] = incoming[overlap]
                continue
            if new_coverage_ratio < self.config.rotation_min_new_coverage_ratio and np.any(coverage):
                self.last_seam_metrics.append(
                    {"frame_index": float(index), "new_coverage_ratio": new_coverage_ratio, "decision": -1.0}
                )
                # Do not turn a deliberately seam-free sliver into a black hole.
                # The existing owner remains in the overlap; only genuinely new
                # pixels extend the visible mask.
                result[only_incoming] = incoming[only_incoming]
                coverage |= incoming_mask
                continue
            if np.any(overlap) and self.config.enable_photometric_normalization:
                incoming, metric = self._photometrically_align(result, incoming, overlap, index)
                self.last_photometric_metrics.append(metric)
            result[only_incoming] = incoming[only_incoming]
            if np.any(overlap):
                if not np.any(only_incoming):
                    # A fully covered frame cannot extend the panorama.  Replacing
                    # an arbitrary half of it causes a sequence of visible seams.
                    if np.array_equal(incoming_mask, coverage) and incoming_sharpness > current_sharpness:
                        result[overlap] = incoming[overlap]
                    continue
                use_incoming, cost, orientation = self._find_seam(result, incoming, overlap, only_incoming)
                result[use_incoming] = incoming[use_incoming]
                self.last_seam_metrics.append(
                    {
                        "frame_index": float(index),
                        "mean_conflict": cost,
                        "overlap_pixels": float(overlap.sum()),
                        "new_coverage_ratio": new_coverage_ratio,
                        "orientation": 1.0 if orientation == "vertical" else 0.0,
                        "decision": 1.0,
                    }
                )
            coverage |= incoming_mask
            current_sharpness = max(current_sharpness, sharpnesses[index] if sharpnesses else 1.0)
        return result

    def _globally_align_frames(
        self, frames: list[np.ndarray], masks: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[dict[str, float]]]:
        """Solve overlap colour offsets for the entire chain with frame zero fixed.

        Local seam correction remains useful for residual exposure changes, but
        this anchored least-squares pass prevents its corrections from drifting
        along a long sequence of frames.
        """
        count = len(frames)
        if count < 2:
            return frames, []
        equations: list[tuple[int, int, np.ndarray]] = []
        for left in range(count - 1):
            for right in range(left + 1, count):
                overlap = (masks[left] > 0) & (masks[right] > 0)
                if int(overlap.sum()) < 64:
                    continue
                left_gray = cv2.cvtColor(frames[left], cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(frames[right], cv2.COLOR_BGR2GRAY)
                calm = overlap & (left_gray > 8) & (left_gray < 247) & (right_gray > 8) & (right_gray < 247)
                if int(calm.sum()) < 64:
                    continue
                difference = np.median(
                    frames[left][calm].astype(np.float32) - frames[right][calm].astype(np.float32), axis=0
                )
                equations.append((left, right, difference))
        if not equations:
            return frames, []
        matrix = np.zeros((len(equations), count - 1), dtype=np.float64)
        for row, (left, right, _) in enumerate(equations):
            if left:
                matrix[row, left - 1] = -1.0
            if right:
                matrix[row, right - 1] = 1.0
        offsets = np.zeros((count, 3), dtype=np.float64)
        for channel in range(3):
            target = np.asarray([difference[channel] for _, _, difference in equations], dtype=np.float64)
            solved, *_ = np.linalg.lstsq(matrix, target, rcond=None)
            offsets[1:, channel] = np.clip(solved, -self.config.photometric_offset_limit, self.config.photometric_offset_limit)
        corrected = [
            np.clip(frame.astype(np.float32) + offsets[index], 0, 255).astype(np.uint8)
            for index, frame in enumerate(frames)
        ]
        metrics = [
            {
                "frame_index": float(index),
                "offset_b": float(offset[0]),
                "offset_g": float(offset[1]),
                "offset_r": float(offset[2]),
                "constraint_count": float(len(equations)),
            }
            for index, offset in enumerate(offsets)
        ]
        return corrected, metrics

    def _photometrically_align(
        self, base: np.ndarray, incoming: np.ndarray, overlap: np.ndarray, index: int
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Apply a robust overlap-only colour correction before selecting a seam."""
        base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        incoming_gray = cv2.cvtColor(incoming, cv2.COLOR_BGR2GRAY)
        gradient = np.abs(cv2.Sobel(incoming_gray, cv2.CV_32F, 1, 0)) + np.abs(
            cv2.Sobel(incoming_gray, cv2.CV_32F, 0, 1)
        )
        candidates = overlap & (base_gray > 8) & (base_gray < 247) & (incoming_gray > 8) & (incoming_gray < 247)
        if np.any(candidates):
            threshold = float(np.percentile(gradient[candidates], 60))
            candidates &= gradient <= threshold
        if int(candidates.sum()) < 32:
            return incoming, {"frame_index": float(index), "sample_count": float(candidates.sum()), "applied": 0.0}
        source = incoming[candidates].astype(np.float32)
        target = base[candidates].astype(np.float32)
        source_std = np.maximum(source.std(axis=0), 1.0)
        gain = np.clip(target.std(axis=0) / source_std, 1.0 - self.config.photometric_gain_limit, 1.0 + self.config.photometric_gain_limit)
        offset = np.clip(
            np.median(target - source * gain, axis=0),
            -self.config.photometric_offset_limit,
            self.config.photometric_offset_limit,
        )
        corrected = np.clip(incoming.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
        before = float(np.mean(np.abs(target - source)))
        after = float(np.mean(np.abs(base[candidates].astype(np.float32) - corrected[candidates].astype(np.float32))))
        return corrected, {
            "frame_index": float(index),
            "sample_count": float(candidates.sum()),
            "applied": 1.0,
            "error_before": before,
            "error_after": after,
            "gain_b": float(gain[0]),
            "gain_g": float(gain[1]),
            "gain_r": float(gain[2]),
            "offset_b": float(offset[0]),
            "offset_g": float(offset[1]),
            "offset_r": float(offset[2]),
        }

    @classmethod
    def _find_seam(
        cls, base: np.ndarray, incoming: np.ndarray, overlap: np.ndarray, only_incoming: np.ndarray
    ) -> tuple[np.ndarray, float, str]:
        """Choose seam orientation from the direction in which coverage grows."""
        overlap_y, overlap_x = np.where(overlap)
        new_y, new_x = np.where(only_incoming)
        # For handheld footage the newly visible area is often L-shaped.  Its
        # centroid tracks the actual camera drift better than its bounding box:
        # using the latter forces a vertical seam through small vertical shakes
        # and produces a dense sawtooth pattern on straight lines.
        horizontal_growth = abs(float(new_x.mean() - overlap_x.mean())) >= abs(float(new_y.mean() - overlap_y.mean()))
        if horizontal_growth:
            selected, cost = cls._vertical_seam(base, incoming, overlap, only_incoming)
            return selected, cost, "vertical"
        # A left-to-right path is the same dynamic programme in transposed
        # coordinates and avoids forcing a vertical seam on tilt panoramas.
        selected, cost = cls._vertical_seam(
            np.swapaxes(base, 0, 1),
            np.swapaxes(incoming, 0, 1),
            overlap.T,
            only_incoming.T,
        )
        return selected.T, cost, "horizontal"

    @staticmethod
    def _vertical_seam(
        base: np.ndarray, incoming: np.ndarray, overlap: np.ndarray, only_incoming: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Find a smooth top-to-bottom seam joining the incoming frame's new edge."""
        difference = np.mean(np.abs(base.astype(np.float32) - incoming.astype(np.float32)), axis=2)
        gray = cv2.cvtColor(incoming, cv2.COLOR_BGR2GRAY)
        salience = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        cost = difference + 0.15 * salience
        ys, xs = np.where(overlap)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        # Dynamic programming limits a seam's per-row movement.  The old
        # independent row minimum was the direct source of the visible sawtooth.
        local_cost = cost[y0 : y1 + 1, x0 : x1 + 1].copy()
        local_overlap = overlap[y0 : y1 + 1, x0 : x1 + 1]
        local_cost[~local_overlap] = 1e9
        cumulative = local_cost.copy()
        parents = np.zeros_like(cumulative, dtype=np.int32)
        max_step = 4
        for row in range(1, cumulative.shape[0]):
            previous = cumulative[row - 1]
            for raw_column in np.flatnonzero(local_overlap[row]):
                column = int(raw_column)
                start, end = max(0, column - max_step), min(cumulative.shape[1], column + max_step + 1)
                relative = int(np.argmin(previous[start:end]))
                parents[row, column] = start + relative
                cumulative[row, column] += previous[start + relative]
        end_x = int(np.argmin(cumulative[-1]))
        seam = np.empty(cumulative.shape[0], dtype=np.int32)
        seam[-1] = end_x
        for row in range(len(seam) - 1, 0, -1):
            seam[row - 1] = parents[row, seam[row]]
        _, unique_x = np.where(only_incoming)
        incoming_on_right = float(np.mean(unique_x)) >= (x0 + x1) / 2.0
        selected = np.zeros_like(overlap)
        for row, seam_x in enumerate(seam, start=y0):
            x = x0 + int(seam_x)
            if incoming_on_right:
                selected[row, x:x1 + 1] = overlap[row, x:x1 + 1]
            else:
                selected[row, x0:x] = overlap[row, x0:x]
        return selected, float(np.mean(difference[overlap]))

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

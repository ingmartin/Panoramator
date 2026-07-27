from __future__ import annotations

import cv2

from panoramator.config.models import PanoramaConfig
from panoramator.domain.models import FeatureSet, MatchSet


class BFMatcherAdapter:
    def __init__(self, config: PanoramaConfig) -> None:
        self.config = config

    def match(self, left: FeatureSet, right: FeatureSet) -> MatchSet:
        if left.descriptors is None or right.descriptors is None:
            return MatchSet(raw_count=0, good_matches=[], confidence=0.0)

        norm_type = cv2.NORM_HAMMING if left.backend == "orb" else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        raw_matches = matcher.knnMatch(left.descriptors, right.descriptors, k=2)
        good_matches = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.config.ratio_test * second.distance:
                good_matches.append(first)
        confidence = len(good_matches) / max(len(raw_matches), 1)
        return MatchSet(raw_count=len(raw_matches), good_matches=good_matches, confidence=confidence)

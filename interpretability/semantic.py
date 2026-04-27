from __future__ import annotations

"""SemanticInterpreter: intensity-band auto-labeling for decoded features."""

import dataclasses
from typing import Iterable, List

from core.hierarchy import Feature


class SemanticInterpreter:
    """Intensity-band semantic interpreter with configurable high/medium thresholds."""

    def __init__(self, high: float = 0.8, medium: float = 0.4) -> None:
        self._high = high
        self._medium = medium

    def annotate(self, features: Iterable[Feature]) -> List[Feature]:
        enriched: List[Feature] = []
        for feature in features:
            if not feature.label:
                band = self._intensity_band(feature.score)
                feature = dataclasses.replace(feature, label=f"{band} concept around activation index {feature.source_index}")
            enriched.append(feature)
        return enriched

    def _intensity_band(self, score: float) -> str:
        if score >= self._high:
            return "high-intensity"
        if score >= self._medium:
            return "medium-intensity"
        return "low-intensity"

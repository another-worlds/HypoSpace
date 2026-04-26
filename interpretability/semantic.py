from __future__ import annotations

"""SemanticInterpreter: intensity-band auto-labeling for decoded features."""

import dataclasses
from typing import Iterable, List

from core.hierarchy import Feature


class SemanticInterpreter:
    """Intensity-band semantic interpreter (high ≥ 0.8, medium ≥ 0.4, low < 0.4)."""

    def annotate(self, features: Iterable[Feature]) -> List[Feature]:
        enriched: List[Feature] = []
        for feature in features:
            if not feature.label:
                band = self._intensity_band(feature.score)
                feature = dataclasses.replace(feature, label=f"{band} concept around activation index {feature.source_index}")
            enriched.append(feature)
        return enriched

    @staticmethod
    def _intensity_band(score: float) -> str:
        if score >= 0.8:
            return "high-intensity"
        if score >= 0.4:
            return "medium-intensity"
        return "low-intensity"

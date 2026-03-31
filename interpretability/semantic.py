from __future__ import annotations

from typing import Iterable, List

from core.hierarchy import Feature


class SemanticInterpreter:
    """Template-first semantic interpreter."""

    def annotate(self, features: Iterable[Feature]) -> List[Feature]:
        enriched: List[Feature] = []
        for feature in features:
            if not feature.label:
                band = self._intensity_band(feature.score)
                feature.label = f"{band} concept around activation index {feature.source_index}"
            enriched.append(feature)
        return enriched

    @staticmethod
    def _intensity_band(score: float) -> str:
        if score >= 0.8:
            return "high-intensity"
        if score >= 0.4:
            return "medium-intensity"
        return "low-intensity"

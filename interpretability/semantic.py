from __future__ import annotations

from typing import Iterable, List

from core.hierarchy import Feature


# ISSUE-L01: "template-first" is inaccurate — uses hardcoded thresholds (0.8, 0.4); no template system exists
class SemanticInterpreter:
    """Template-first semantic interpreter."""

    def annotate(self, features: Iterable[Feature]) -> List[Feature]:
        enriched: List[Feature] = []
        for feature in features:
            if not feature.label:
                band = self._intensity_band(feature.score)
                # ISSUE-M02: mutates Feature.label in-place; callers holding the original object see this change
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

from __future__ import annotations

from typing import Iterable, List, Tuple

from core.hierarchy import Feature


class SemanticCanvas:
    """Lightweight data prep for concept graph visualization."""

    def to_points(self, features: Iterable[Feature]) -> List[Tuple[str, float]]:
        return [(feature.id, feature.score) for feature in features]

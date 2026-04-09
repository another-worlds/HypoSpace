from __future__ import annotations

from typing import Iterable, List, Tuple

from core.hierarchy import Feature


class SemanticCanvas:
    """Lightweight data prep for concept graph visualization."""

    def to_points(self, features: Iterable[Feature]) -> List[Tuple[str, float]]:
        ordered = sorted(features, key=lambda feature: feature.score, reverse=True)
        return [(feature.id, feature.score) for feature in ordered]

    def to_edges(self, features: Iterable[Feature]) -> List[Tuple[str, str, float]]:
        """Create simple nearest-neighbor links by score order.

        Edges are `(source_id, target_id, strength)` where strength is the
        average absolute score of the two connected features.
        """
        ordered = sorted(features, key=lambda feature: feature.score, reverse=True)
        if len(ordered) < 2:
            return []

        edges: List[Tuple[str, str, float]] = []
        for source, target in zip(ordered, ordered[1:]):
            strength = (abs(source.score) + abs(target.score)) / 2.0
            edges.append((source.id, target.id, strength))
        return edges

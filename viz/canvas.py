from __future__ import annotations

"""SemanticCanvas: feature-point and nearest-neighbour-edge visualization helpers."""

from typing import Dict, Iterable, List, Tuple

from core.hierarchy import Feature


class SemanticCanvas:
    """Lightweight data prep for concept graph visualization."""

    def to_points(self, features: Iterable[Feature]) -> List[Tuple[str, float]]:
        ordered = sorted(features, key=lambda feature: feature.score, reverse=True)
        return [(feature.id, feature.score) for feature in ordered]

    def to_layout(self, features: Iterable[Feature]) -> List[Dict[str, object]]:
        """Return features as positioned points for scatter visualization.

        x = normalized rank (highest-scoring feature at x=1.0, lowest at x=0.0).
        y = score normalized to [0, 1] relative to the maximum absolute score.
        Single-feature input returns x=0.5, y=0.5. Empty input returns [].
        """
        ordered = sorted(features, key=lambda f: f.score, reverse=True)
        n = len(ordered)
        if n == 0:
            return []
        if n == 1:
            return [{"id": ordered[0].id, "score": ordered[0].score, "x": 0.5, "y": 0.5, "label": ordered[0].label}]
        max_abs = max(abs(f.score) for f in ordered) or 1.0
        result: List[Dict[str, object]] = []
        for rank, f in enumerate(ordered):
            x = 1.0 - rank / (n - 1)
            y = max(0.0, min(1.0, (f.score + max_abs) / (2 * max_abs)))
            result.append({"id": f.id, "score": f.score, "x": x, "y": y, "label": f.label})
        return result

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

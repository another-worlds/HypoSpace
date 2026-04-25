from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(slots=True)
class Feature:
    """Single extracted feature in latent concept space."""

    id: str
    layer: str
    score: float
    source_index: int
    label: str | None = None


class HierarchyEngine:
    """Thin wrapper over hierarchical/universal SAE backends.

    This class intentionally keeps logic minimal and delegates heavy lifting
    to external libraries.
    """

    def __init__(self, backend: str = "matryoshka") -> None:
        self.backend = backend

    def extract_features(self, activations: Iterable[float], layer: str, top_k: int = 8) -> List[Feature]:
        """Select top-k magnitudes as MVP feature candidates."""
        values = list(activations)
        if not values:
            return []

        ranked = sorted(enumerate(values), key=lambda x: abs(x[1]), reverse=True)
        # ISSUE-M04: top_k ≤ 0 silently becomes 1; no warning emitted to the caller
        selected = ranked[: max(1, top_k)]
        return [
            Feature(
                id=f"{layer}:feature:{rank}",
                layer=layer,
                score=abs(value),
                source_index=index,
            )
            for rank, (index, value) in enumerate(selected)
        ]

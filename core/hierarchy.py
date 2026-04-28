from __future__ import annotations

"""HierarchyEngine and Feature dataclass: top-k feature extraction by activation magnitude."""

from dataclasses import dataclass
from typing import Iterable, List, Protocol, runtime_checkable


@dataclass(slots=True)
class Feature:
    """Single extracted feature in latent concept space."""

    id: str
    layer: str
    score: float
    source_index: int
    label: str | None = None


@runtime_checkable
class FeatureBackend(Protocol):
    """Extraction contract satisfied by MagnitudeBackend and MatryoshkaBackend.

    Receives a flat list of activation floats and returns Feature objects ranked
    by score descending. top_k clamping and the ValueError guard for non-positive
    top_k are handled by HierarchyEngine before calling extract().
    """

    def extract(self, activations: list[float], layer: str, top_k: int) -> List[Feature]: ...


def _magnitude_top_k(values: list[float], layer: str, top_k: int) -> List[Feature]:
    """Magnitude-based top-k selection — the stdlib-only fallback path."""
    ranked = sorted(enumerate(values), key=lambda x: abs(x[1]), reverse=True)
    selected = ranked[:top_k]
    return [
        Feature(
            id=f"{layer}:feature:{rank}",
            layer=layer,
            score=abs(value),
            source_index=index,
        )
        for rank, (index, value) in enumerate(selected)
    ]


class HierarchyEngine:
    """Thin dispatcher over SAE or magnitude backends.

    When feature_backend is None (default) the engine falls back to magnitude
    top-k, preserving full backward compatibility. Pass a FeatureBackend
    (e.g. MatryoshkaBackend from data/sae_backend.py) to use a real SAE.
    """

    def __init__(self, backend: str = "matryoshka", feature_backend: FeatureBackend | None = None) -> None:
        self.backend = backend
        self._feature_backend = feature_backend

    def extract_features(self, activations: Iterable[float], layer: str, top_k: int = 8) -> List[Feature]:
        """Dispatch to the injected backend, or fall back to magnitude top-k."""
        values = list(activations)
        if not values:
            return []
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
        if self._feature_backend is not None:
            return self._feature_backend.extract(values, layer, top_k)
        return _magnitude_top_k(values, layer, top_k)

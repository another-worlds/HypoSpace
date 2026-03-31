from __future__ import annotations

from typing import Iterable, List


class ActivationExtractor:
    """nnsight-facing extractor interface (MVP stub)."""

    def extract(self, raw_activations: Iterable[float]) -> List[float]:
        return [float(item) for item in raw_activations]

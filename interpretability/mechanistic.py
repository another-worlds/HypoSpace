from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from core.hierarchy import Feature


@dataclass(slots=True)
class InterventionResult:
    feature_id: str
    baseline: float
    ablated: float
    effect_size: float


class MechanisticAnalyzer:
    """CPU-only 50% ablation stub. Use PyVeneInterventionRunner for real zero-ablation when torch is available."""

    def run_interventions(self, features: Iterable[Feature]) -> List[InterventionResult]:
        """Return synthetic intervention results (ablated = baseline * 0.5, effect_size always baseline / 2)."""
        rows: List[InterventionResult] = []
        for feature in features:
            baseline = feature.score
            ablated = baseline * 0.5
            effect_size = max(0.0, baseline - ablated)
            rows.append(
                InterventionResult(
                    feature_id=feature.id,
                    baseline=baseline,
                    ablated=ablated,
                    effect_size=effect_size,
                )
            )
        return rows

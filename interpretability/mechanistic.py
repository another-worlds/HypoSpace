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

    # ISSUE-M03: result fields match real pyvene output but values are synthetic (baseline * 0.5); no provenance flag
    def run_interventions(self, features: Iterable[Feature]) -> List[InterventionResult]:
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

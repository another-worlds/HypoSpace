from __future__ import annotations

"""FaithfulnessChecker, GovernanceScorecard, and GovernanceThresholdError for governance enforcement."""

from dataclasses import dataclass
from typing import Iterable

from core.config import GovernanceConfig
from interpretability.mechanistic import InterventionResult


class GovernanceThresholdError(RuntimeError):
    """Raised when governance thresholds are configured as strict and not met."""


@dataclass(slots=True)
class GovernanceScorecard:
    faithfulness_score: float
    stability_score: float
    risk_flag: str
    passes_thresholds: bool
    intervention_method: str = "unknown"

    def to_dict(self) -> dict[str, object]:
        return {
            "faithfulness_score": self.faithfulness_score,
            "stability_score": self.stability_score,
            "risk_flag": self.risk_flag,
            "passes_thresholds": self.passes_thresholds,
            "intervention_method": self.intervention_method,
        }


class FaithfulnessChecker:
    """Compute lightweight governance metrics for MVP."""

    def __init__(self, config: GovernanceConfig | None = None) -> None:
        self.config = config or GovernanceConfig()

    def evaluate(
        self,
        interventions: Iterable[InterventionResult],
        intervention_method: str = "unknown",
    ) -> GovernanceScorecard:
        rows = list(interventions)
        if not rows:
            scorecard = GovernanceScorecard(
                faithfulness_score=0.0,
                stability_score=0.0,
                risk_flag="no_data",
                passes_thresholds=False,
                intervention_method=intervention_method,
            )
            self._enforce_thresholds(scorecard)
            return scorecard

        mean_effect = sum(row.effect_size for row in rows) / len(rows)
        stability = self._stability(rows)
        passes = mean_effect >= self.config.min_faithfulness_score and stability >= self.config.min_stability_score
        risk_flag = "ok" if passes else "low_faithfulness"

        scorecard = GovernanceScorecard(
            faithfulness_score=mean_effect,
            stability_score=stability,
            risk_flag=risk_flag,
            passes_thresholds=passes,
            intervention_method=intervention_method,
        )
        self._enforce_thresholds(scorecard)
        return scorecard

    def _enforce_thresholds(self, scorecard: GovernanceScorecard) -> None:
        if self.config.fail_on_low_confidence and not scorecard.passes_thresholds:
            raise GovernanceThresholdError(
                "Governance thresholds failed: "
                f"faithfulness={scorecard.faithfulness_score:.3f}, "
                f"stability={scorecard.stability_score:.3f}"
            )

    @staticmethod
    def _stability(rows: list[InterventionResult]) -> float:
        baselines = [row.baseline for row in rows]
        if not baselines:
            return 0.0
        mean = sum(baselines) / len(baselines)
        # Guard against near-zero mean before dividing: avoids denominator collapsing to 1e-6
        if abs(mean) < 1e-4:
            return 1.0
        variance = sum((value - mean) ** 2 for value in baselines) / len(baselines)
        normalized_std = (variance ** 0.5) / (abs(mean) + 1e-6)
        return max(0.0, 1.0 - min(1.0, normalized_std))

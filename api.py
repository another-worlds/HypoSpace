from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.config import DecoderConfig
from core.decoder import DecodeResult, RealityDecoder
from data.extractor import ActivationExtractor
from data.preprocessor import ActivationPreprocessor
from interpretability.faithfulness import FaithfulnessChecker, GovernanceScorecard
from interpretability.mechanistic import MechanisticAnalyzer
from interpretability.semantic import SemanticInterpreter


@dataclass(slots=True)
class HypoSpaceResult:
    decode: DecodeResult
    scorecard: GovernanceScorecard


class HypoSpaceAPI:
    """Public API façade for one-call model introspection workflows."""

    def __init__(self, config: DecoderConfig | None = None) -> None:
        self.config = config or DecoderConfig()
        self.extractor = ActivationExtractor()
        self.preprocessor = ActivationPreprocessor()
        self.decoder = RealityDecoder(config=self.config)
        self.semantic = SemanticInterpreter()
        self.mechanistic = MechanisticAnalyzer()
        self.faithfulness = FaithfulnessChecker(config=self.config.governance)

    def decode(
        self,
        model_name: str,
        layer: str,
        raw_activations: Iterable[float],
        version: str = "0.1.0",
        reference_kernel_id: str | None = None,
    ) -> DecodeResult:
        values = self.preprocessor.normalize(self.extractor.extract(raw_activations))
        result = self.decoder.decode(
            model_name=model_name,
            layer=layer,
            activations=values,
            version=version,
            reference_kernel_id=reference_kernel_id,
        )
        result.features = self.semantic.annotate(result.features)
        return result

    def scorecard(self, result: DecodeResult) -> GovernanceScorecard:
        interventions = self.mechanistic.run_interventions(result.features)
        return self.faithfulness.evaluate(interventions)

    def decode_and_score(
        self,
        model_name: str,
        layer: str,
        raw_activations: Iterable[float],
        version: str = "0.1.0",
        reference_kernel_id: str | None = None,
    ) -> HypoSpaceResult:
        decode_result = self.decode(
            model_name=model_name,
            layer=layer,
            raw_activations=raw_activations,
            version=version,
            reference_kernel_id=reference_kernel_id,
        )
        scorecard = self.scorecard(decode_result)
        return HypoSpaceResult(decode=decode_result, scorecard=scorecard)

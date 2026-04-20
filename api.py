from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

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
        self.extractor = ActivationExtractor(cache_dir=self.config.runtime.cache_dir)
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
    ) -> DecodeResult:
        values = self.preprocessor.normalize(
            self.extractor.extract(
                raw_activations,
                model_name=model_name,
                layer=layer,
            )
        )
        result = self.decoder.decode(model_name=model_name, layer=layer, activations=values, version=version)
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
    ) -> HypoSpaceResult:
        decode_result = self.decode(model_name=model_name, layer=layer, raw_activations=raw_activations, version=version)
        scorecard = self.scorecard(decode_result)
        return HypoSpaceResult(decode=decode_result, scorecard=scorecard)

    def decode_from_model(
        self,
        model_name: str,
        layer: str,
        layer_path: str,
        inputs: Union[str, Iterable[int]],
        token_index: int = -1,
        version: str = "0.1.0",
    ) -> DecodeResult:
        """Extract live activations via nnsight and decode.

        Args:
            model_name: HuggingFace model identifier (e.g. "gpt2").
            layer: HypoSpace layer name used for artifact naming.
            layer_path: nnsight attribute path (e.g. "transformer.h.0").
            inputs: Text string or token id sequence passed to the model.
            token_index: Token position to extract; -1 means last token.
            version: Kernel version tag.

        Returns:
            DecodeResult with features extracted from the live model.
        """
        from data.nnsight_extractor import NNSightExtractor

        nnsight_ex = NNSightExtractor(
            model_name=model_name,
            device=self.config.runtime.device,
            cache_dir=self.config.runtime.cache_dir,
        )
        raw_activations = nnsight_ex.extract(
            inputs=inputs,
            layer_path=layer_path,
            token_index=token_index,
        )
        values = self.preprocessor.normalize(raw_activations)
        result = self.decoder.decode(model_name=model_name, layer=layer, activations=values, version=version)
        result.features = self.semantic.annotate(result.features)
        return result

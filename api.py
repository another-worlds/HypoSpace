from __future__ import annotations

"""Public one-call façade: HypoSpaceAPI orchestrates decode, score, and diagnostics workflows."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Union

from core.config import DecoderConfig
from core.decoder import DecodeResult, RealityDecoder
from core.hierarchy import Feature
from data.extractor import ActivationExtractor
from data.preprocessor import ActivationPreprocessor
from interpretability.faithfulness import FaithfulnessChecker, GovernanceScorecard
from interpretability.mechanistic import InterventionResult, MechanisticAnalyzer
from interpretability.semantic import SemanticInterpreter

if TYPE_CHECKING:
    from data.nnsight_extractor import NNSightExtractor
    from diagnostics import DiagnosticsReport


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
        self._nnsight: NNSightExtractor | None = None

    # ------------------------------------------------------------------
    # Path A — raw activations
    # ------------------------------------------------------------------

    @staticmethod
    def _check_finite(values: list[float]) -> None:
        bad = [v for v in values if v != v or v == float("inf") or v == float("-inf")]
        if bad:
            raise ValueError(f"raw_activations contains non-finite values: {bad[:3]!r}")

    def decode(
        self,
        model_name: str,
        layer: str,
        raw_activations: Iterable[float],
        version: str = "0.1.0",
    ) -> DecodeResult:
        floats = [float(v) for v in raw_activations]
        self._check_finite(floats)
        values = self.preprocessor.normalize(
            self.extractor.extract(
                floats,
                model_name=model_name,
                layer=layer,
            )
        )
        result = self.decoder.decode(model_name=model_name, layer=layer, activations=values, version=version)
        result.features = self.semantic.annotate(result.features)
        return result

    def scorecard(
        self,
        result: DecodeResult,
        interventions: list[InterventionResult] | None = None,
        intervention_method: str = "unknown",
    ) -> GovernanceScorecard:
        if interventions is None:
            interventions = self.mechanistic.run_interventions(result.features)
            intervention_method = "stub-50pct"
        return self.faithfulness.evaluate(interventions, intervention_method=intervention_method)

    def decode_and_score(
        self,
        model_name: str,
        layer: str,
        raw_activations: Iterable[float],
        version: str = "0.1.0",
    ) -> HypoSpaceResult:
        decode_result = self.decode(model_name=model_name, layer=layer, raw_activations=raw_activations, version=version)
        return HypoSpaceResult(decode=decode_result, scorecard=self.scorecard(decode_result))

    # ------------------------------------------------------------------
    # Path B — live model extraction via nnsight
    # ------------------------------------------------------------------

    def _nnsight_extractor(self, model_name: str) -> NNSightExtractor:
        """Return a cached NNSightExtractor, reloading only on model change."""
        from data.nnsight_extractor import NNSightExtractor

        if self._nnsight is None or self._nnsight.model_name != model_name:
            self._nnsight = NNSightExtractor(
                model_name=model_name,
                device=self.config.runtime.device,
                cache_dir=self.config.runtime.cache_dir,
            )
        return self._nnsight

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
        raw_activations = self._nnsight_extractor(model_name).extract(
            inputs=inputs,
            layer_path=layer_path,
            token_index=token_index,
        )
        values = self.preprocessor.normalize(raw_activations)
        result = self.decoder.decode(model_name=model_name, layer=layer, activations=values, version=version)
        result.features = self.semantic.annotate(result.features)
        return result

    def decode_and_score_from_model(
        self,
        model_name: str,
        layer: str,
        layer_path: str,
        inputs: Union[str, Iterable[int]],
        token_index: int = -1,
        version: str = "0.1.0",
    ) -> HypoSpaceResult:
        """Extract live activations via nnsight, decode, and score governance.

        Single-call equivalent of decode_from_model() + scorecard() for the
        live-model path. Uses real zero-ablation interventions when torch is
        available; falls back to the stub otherwise.
        """
        decode_result = self.decode_from_model(
            model_name=model_name,
            layer=layer,
            layer_path=layer_path,
            inputs=inputs,
            token_index=token_index,
            version=version,
        )
        interventions, method = self._run_real_interventions(
            features=decode_result.features,
            model_name=model_name,
            layer_path=layer_path,
            inputs=inputs,
            token_index=token_index,
        )
        return HypoSpaceResult(
            decode=decode_result,
            scorecard=self.scorecard(decode_result, interventions=interventions, intervention_method=method),
        )

    def _run_real_interventions(
        self,
        features: list[Feature],
        model_name: str,
        layer_path: str,
        inputs: Union[str, Iterable[int]],
        token_index: int,
    ) -> tuple[list[InterventionResult], str]:
        """Attempt real interventions via PyVeneInterventionRunner; fall back to stub.

        Returns (interventions, intervention_method) so callers can record provenance.
        """
        try:
            from data.pyvene_runner import PyVeneInterventionRunner
            import torch  # noqa: F401
            nnsight_ex = self._nnsight_extractor(model_name)
            runner = PyVeneInterventionRunner(
                lm=nnsight_ex._lm,
                layer_path=layer_path,
                device=self.config.runtime.device,
            )
            return runner.run_interventions(features, inputs, token_index), "pyvene-zero-ablation"
        except ImportError:
            return self.mechanistic.run_interventions(features), "stub-50pct"

    def diagnostics(self) -> DiagnosticsReport:
        from diagnostics import run_diagnostics
        return run_diagnostics()

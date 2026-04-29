from __future__ import annotations

"""Public one-call façade: HypoSpaceAPI orchestrates decode, score, and diagnostics workflows."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Union

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

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
    from data.sae_backend import MatryoshkaBackend
    from diagnostics import DiagnosticsReport


@dataclass(slots=True)
class HypoSpaceResult:
    decode: DecodeResult
    scorecard: GovernanceScorecard

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.decode.model_name,
            "layer": self.decode.layer,
            "features": [
                {"id": f.id, "score": f.score, "label": f.label}
                for f in self.decode.features
            ],
            "scorecard": self.scorecard.to_dict(),
        }


class HypoSpaceAPI:
    """Public API façade for one-call model introspection workflows."""

    def __init__(self, config: DecoderConfig | None = None) -> None:
        self.config = config or DecoderConfig()
        self.extractor = ActivationExtractor(cache_dir=self.config.runtime.cache_dir)
        self.preprocessor = ActivationPreprocessor()
        self.decoder = RealityDecoder(config=self.config, feature_backend=self._build_sae_backend())
        self.semantic = SemanticInterpreter(
            high=self.config.high_intensity_threshold,
            medium=self.config.medium_intensity_threshold,
        )
        self.mechanistic = MechanisticAnalyzer()
        self.faithfulness = FaithfulnessChecker(config=self.config.governance)
        self._nnsight: NNSightExtractor | None = None

    def _build_sae_backend(self):
        """Construct SAE backend from config; return None on any failure (magnitude fallback)."""
        try:
            from data.sae_backend import build_backend
            return build_backend(
                backend=self.config.backend,
                sae_path=self.config.sae_path,
                device=self.config.runtime.device,
            )
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Path A — raw activations
    # ------------------------------------------------------------------

    @staticmethod
    def _check_finite(values: list[float]) -> None:
        bad = [v for v in values if v != v or v == float("inf") or v == float("-inf")]
        if bad:
            raise ValueError(f"raw_activations contains non-finite values: {bad[:3]!r}")

    def _validate_decode_args(self, model_name: str, layer: str, version: str, values: list[float]) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not layer.strip():
            raise ValueError("layer must be a non-empty string")
        if not _SEMVER_RE.match(version):
            raise ValueError(f"version must be semver (X.Y.Z), got {version!r}")
        max_features = self.config.runtime.max_features
        if len(values) > max_features:
            raise ValueError(f"activations length {len(values)} exceeds max_features={max_features}")

    def decode(
        self,
        model_name: str,
        layer: str,
        raw_activations: Iterable[float],
        version: str = "0.1.0",
    ) -> DecodeResult:
        floats = [float(v) for v in raw_activations]
        self._validate_decode_args(model_name, layer, version, floats)
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
            # No real interventions available on Path A — always falls back to stub.
            interventions = self.mechanistic.run_interventions(result.features)
            intervention_method = "stub-50pct"
        return self.faithfulness.evaluate(interventions, intervention_method=intervention_method)

    def decode_and_score(
        self,
        model_name: str,
        layer: str,
        raw_activations: Iterable[float],
        version: str = "0.1.0",
        *,
        layer_path: str | None = None,
        inputs: Union[str, Iterable[int]] | None = None,
        token_index: int = -1,
    ) -> HypoSpaceResult:
        """Decode raw activations and compute governance scorecard.

        When layer_path and inputs are both provided and torch is available,
        real zero-ablation interventions are used (same as Path B). Otherwise
        falls back to the MechanisticAnalyzer stub (intervention_method="stub-50pct").
        For unconditional real interventions use decode_and_score_from_model().
        """
        decode_result = self.decode(model_name=model_name, layer=layer, raw_activations=raw_activations, version=version)
        if layer_path is not None and inputs is not None:
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
        return HypoSpaceResult(decode=decode_result, scorecard=self.scorecard(decode_result))

    def decode_batch(
        self,
        model_name: str,
        layer: str,
        activation_matrix: Iterable[Iterable[float]],
        version: str = "0.1.0",
    ) -> list[DecodeResult]:
        """Decode multiple activation vectors for the same model and layer."""
        return [self.decode(model_name, layer, row, version) for row in activation_matrix]

    def decode_and_score_batch(
        self,
        model_name: str,
        layer: str,
        activation_matrix: Iterable[Iterable[float]],
        version: str = "0.1.0",
    ) -> list[HypoSpaceResult]:
        """Decode and score multiple activation vectors for the same model and layer."""
        return [self.decode_and_score(model_name, layer, row, version) for row in activation_matrix]

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
            nnsight_ex.ensure_loaded()
            runner = PyVeneInterventionRunner(
                lm=nnsight_ex.loaded_model,
                layer_path=layer_path,
                device=self.config.runtime.device,
                ablation_mode="hooks",
            )
            return runner.run_interventions(features, inputs, token_index), runner.intervention_method
        except ImportError:
            return self.mechanistic.run_interventions(features), "stub-50pct"

    def diagnostics(self) -> DiagnosticsReport:
        from diagnostics import run_diagnostics
        return run_diagnostics()

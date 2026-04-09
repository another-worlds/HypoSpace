from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from core.config import DecoderConfig
from core.hierarchy import Feature, HierarchyEngine
from core.kernel_library import KernelLibrary, KernelTemplate
from data.utils import utc_timestamp


@dataclass(slots=True)
class DecodeResult:
    model_name: str
    layer: str
    features: List[Feature] = field(default_factory=list)
    kernel_path: str | None = None
    metadata: Dict[str, str] = field(default_factory=dict)


class RealityDecoder:
    """Main orchestration entrypoint."""

    def __init__(self, config: DecoderConfig | None = None, kernel_library: KernelLibrary | None = None) -> None:
        self.config = config or DecoderConfig()
        self.hierarchy = HierarchyEngine(backend=self.config.backend)
        self.kernels = kernel_library or KernelLibrary(root=self.config.runtime.cache_dir)

    def decode(
        self,
        model_name: str,
        layer: str,
        activations: Iterable[float],
        version: str = "0.1.0",
    ) -> DecodeResult:
        features = self.hierarchy.extract_features(
            activations=activations,
            layer=layer,
            top_k=min(self.config.top_k, self.config.runtime.max_features),
        )
        kernel_id = f"{model_name}-{layer}"
        cross_run_match_rate = 0.0
        try:
            match_scores = self.kernels.match(kernel_id=kernel_id, features=features)
            if match_scores:
                positive = [score for score in match_scores.values() if score >= 0.5]
                cross_run_match_rate = len(positive) / len(match_scores)
        except KeyError:
            cross_run_match_rate = 0.0

        template = KernelTemplate(
            kernel_id=kernel_id,
            version=version,
            model_name=model_name,
            layer=layer,
            features=features,
            metadata={
                "created_at_utc": utc_timestamp(),
                "backend": self.config.backend,
                "device": self.config.runtime.device,
                "cross_run_match_rate": f"{cross_run_match_rate:.3f}",
            },
        )
        kernel_path = self.kernels.save(template)
        return DecodeResult(
            model_name=model_name,
            layer=layer,
            features=features,
            kernel_path=str(kernel_path),
            metadata={
                "backend": self.config.backend,
                "device": self.config.runtime.device,
                "top_k": str(self.config.top_k),
                "cross_run_match_rate": f"{cross_run_match_rate:.3f}",
            },
        )

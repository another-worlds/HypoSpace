from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from core.config import DecoderConfig
from core.hierarchy import Feature, HierarchyEngine
from core.kernel_library import KernelLibrary, KernelTemplate


@dataclass(slots=True)
class DecodeResult:
    model_name: str
    layer: str
    features: List[Feature] = field(default_factory=list)
    kernel_path: str | None = None
    match_score: float | None = None
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
        reference_kernel_id: str | None = None,
    ) -> DecodeResult:
        features = self.hierarchy.extract_features(
            activations=activations,
            layer=layer,
            top_k=min(self.config.top_k, self.config.runtime.max_features),
        )
        template = KernelTemplate(
            kernel_id=f"{model_name}-{layer}",
            version=version,
            model_name=model_name,
            layer=layer,
            features=features,
        )
        kernel_path = self.kernels.save(template)

        match_score: float | None = None
        if reference_kernel_id:
            match_score = self.kernels.match_latest(reference_kernel_id, features)

        return DecodeResult(
            model_name=model_name,
            layer=layer,
            features=features,
            kernel_path=str(kernel_path),
            match_score=match_score,
            metadata={
                "backend": self.config.backend,
                "device": self.config.runtime.device,
                "top_k": str(self.config.top_k),
            },
        )

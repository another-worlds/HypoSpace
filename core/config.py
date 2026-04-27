from __future__ import annotations

"""Configuration dataclasses: DecoderConfig, RuntimeConfig, GovernanceConfig."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class GovernanceConfig:
    """Configuration for faithfulness and governance checks."""

    min_faithfulness_score: float = 0.65
    min_stability_score: float = 0.60
    fail_on_low_confidence: bool = False


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime controls for CPU/GPU execution profiles."""

    device: str = "cpu"
    max_features: int = 2048
    cache_dir: str = ".hypo_cache"


@dataclass(slots=True)
class DecoderConfig:
    """Top-level decoder settings."""

    backend: str = "matryoshka"
    top_k: int = 8
    sae_path: str | None = None
    high_intensity_threshold: float = 0.8
    medium_intensity_threshold: float = 0.4
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)

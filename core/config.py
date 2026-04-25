from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


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
    batch_size: int = 1  # ISSUE-M07: never read by any component — wire into NNSightExtractor or remove
    max_features: int = 2048
    cache_dir: str = ".hypo_cache"


@dataclass(slots=True)
class DecoderConfig:
    """Top-level decoder settings."""

    backend: str = "matryoshka"
    top_k: int = 8
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)

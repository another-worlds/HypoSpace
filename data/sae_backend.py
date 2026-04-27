from __future__ import annotations

"""SAE feature extraction backends (torch-optional).

MagnitudeBackend  — stdlib-only magnitude top-k, mirrors HierarchyEngine fallback.
MatryoshkaBackend — Matryoshka/TopK/JumpReLU SAE encoder (requires torch).

Both satisfy the FeatureBackend protocol from core.hierarchy.

Usage (via api.py composition root):
    from data.sae_backend import build_backend
    backend = build_backend(config.backend, config.sae_path, config.runtime.device)
    # Returns None when torch is absent or sae_path is None → magnitude fallback.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

from core.hierarchy import Feature, _magnitude_top_k


def sae_available() -> bool:
    """Return True when torch is importable (minimum requirement for SAE backends)."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# MagnitudeBackend — stdlib-only, always available
# ---------------------------------------------------------------------------

class MagnitudeBackend:
    """Stdlib-only magnitude top-k backend. Satisfies FeatureBackend protocol."""

    def extract(self, activations: list[float], layer: str, top_k: int) -> List[Feature]:
        return _magnitude_top_k(activations, layer, top_k)


# ---------------------------------------------------------------------------
# SAEEncodeResult — intermediate result, independently testable
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SAEEncodeResult:
    """Raw sparse output of one SAE forward pass before Feature construction."""

    feature_indices: List[int]
    feature_scores: List[float]


# ---------------------------------------------------------------------------
# MatryoshkaBackend — torch-required
# ---------------------------------------------------------------------------

class MatryoshkaBackend:
    """Matryoshka / TopK / JumpReLU SAE encoder (requires torch).

    Loads a trained Linear encoder from a checkpoint and uses
    Linear → ReLU → sparse (index, score) pairs to produce Features.

    Feature.id uses the ":sae:" infix for machine-readable provenance:
        "{layer}:sae:{rank}:{sae_dictionary_index}"
    Feature.source_index is the SAE dictionary index (not the raw activation index).
    Feature.score is the post-ReLU activation magnitude.

    Checkpoint layouts supported:
    - Directory: sae_path/encoder.pt  (bare Linear state dict)
    - Single file: sae_path           (full checkpoint with optional "encoder" sub-key)

    Args:
        sae_path: Path to the SAE checkpoint directory or .pt file.
        device:   Torch device string, e.g. "cpu" or "cuda".
    """

    def __init__(self, sae_path: str, device: str = "cpu") -> None:
        import torch  # ImportError propagates to build_backend → graceful fallback
        self._device = device
        self._encoder: torch.nn.Module = self._load_encoder(Path(sae_path), device)

    @staticmethod
    def _load_encoder(path: Path, device: str) -> "torch.nn.Module":
        import torch

        ckpt_path = path / "encoder.pt" if path.is_dir() else path
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAE checkpoint not found at {ckpt_path}. "
                "Check DecoderConfig.sae_path."
            )

        state = torch.load(ckpt_path, map_location=device, weights_only=True)

        # Full checkpoint may nest the Linear state dict under an "encoder" key.
        if "encoder" in state and not isinstance(state["encoder"], torch.Tensor):
            state = state["encoder"]

        # Infer dimensions from weight matrix shape: (dict_size, d_in).
        weight = state["weight"]
        dict_size, d_in = weight.shape
        encoder = torch.nn.Linear(d_in, dict_size, bias="bias" in state)
        encoder.load_state_dict(state)
        encoder.to(device)
        encoder.eval()
        return encoder

    def _encode(self, activations: list[float]) -> SAEEncodeResult:
        """Run the encoder forward pass; return the non-zero (index, score) pairs."""
        import torch

        with torch.no_grad():
            x = torch.tensor(activations, dtype=torch.float32, device=self._device)
            activated = torch.relu(self._encoder(x))
            mask = activated > 0
            indices = mask.nonzero(as_tuple=True)[0].tolist()
            scores = activated[mask].tolist()

        return SAEEncodeResult(feature_indices=indices, feature_scores=scores)

    def extract(self, activations: list[float], layer: str, top_k: int) -> List[Feature]:
        """Encode activations and return top-k Features by SAE activation score."""
        result = self._encode(activations)
        paired = sorted(zip(result.feature_scores, result.feature_indices), reverse=True)
        return [
            Feature(
                id=f"{layer}:sae:{rank}:{sae_idx}",
                layer=layer,
                score=float(score),
                source_index=sae_idx,
            )
            for rank, (score, sae_idx) in enumerate(paired[:top_k])
        ]


# ---------------------------------------------------------------------------
# Factory — used exclusively by api.py
# ---------------------------------------------------------------------------

def build_backend(backend: str, sae_path: str | None, device: str = "cpu"):
    """Construct the appropriate FeatureBackend, or return None for magnitude fallback.

    Returns None when:
    - backend == "magnitude"
    - sae_path is None (no checkpoint configured)
    - torch is not installed
    - checkpoint is missing or corrupt (emits a UserWarning)
    - backend string is unrecognised

    matryoshka, topk, and jumprelu all share the same Linear → ReLU inference
    shape and route to MatryoshkaBackend.
    """
    if backend == "magnitude" or sae_path is None:
        return None
    if not sae_available():
        return None
    try:
        if backend in ("matryoshka", "topk", "jumprelu"):
            return MatryoshkaBackend(sae_path=sae_path, device=device)
    except Exception as exc:
        warnings.warn(
            f"SAE backend '{backend}' failed to load from '{sae_path}' ({exc}); "
            "falling back to magnitude top-k.",
            stacklevel=3,
        )
    return None

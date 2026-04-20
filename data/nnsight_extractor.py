from __future__ import annotations

from typing import Iterable, List, Union

from data.extractor import ActivationExtractor


def nnsight_available() -> bool:
    """Return True if nnsight and torch are importable."""
    try:
        import nnsight  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class NNSightExtractor:
    """Live activation extractor using nnsight model tracing.

    Requires nnsight and torch. Results are written through
    ActivationExtractor's disk cache for cross-run reuse.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        cache_dir: str = ".hypo_cache",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._cache = ActivationExtractor(cache_dir=cache_dir)
        self._lm: object | None = None

    @classmethod
    def require(cls) -> None:
        """Raise ImportError with install instructions if nnsight is missing."""
        if not nnsight_available():
            raise ImportError(
                "nnsight and torch are required for live extraction. "
                "Install with: pip install nnsight"
            )

    def _load_model(self) -> object:
        self.require()
        if self._lm is None:
            from nnsight import LanguageModel
            self._lm = LanguageModel(self.model_name, device_map=self.device)
        return self._lm

    def extract(
        self,
        inputs: Union[str, Iterable[int]],
        layer_path: str,
        token_index: int = -1,
    ) -> List[float]:
        """Trace a forward pass and return activations at layer_path.

        Args:
            inputs: Text string or token id sequence.
            layer_path: Dot-notation attribute path on the model
                (e.g. "transformer.h.0", "model.layers.3").
            token_index: Token position to extract; -1 means last token.

        Returns:
            Flat list of float activation values for the given token.
        """
        import torch

        lm = self._load_model()

        with lm.trace(inputs):
            layer = _resolve_layer(lm, layer_path)
            # In nnsight >= 0.3, .save() returns the tensor directly.
            raw = layer.output.save()

        # Some layers return (hidden_states, ...) tuples; take the first element.
        if isinstance(raw, tuple):
            raw = raw[0]

        if isinstance(raw, torch.Tensor):
            raw = raw.detach()
            # Shape: (batch, seq_len, hidden) or (seq_len, hidden)
            if raw.dim() == 3:
                raw = raw[0]  # drop batch dim
            values: List[float] = raw[token_index].tolist()
        else:
            values = list(raw)

        return self._cache.extract(
            raw_activations=values,
            model_name=self.model_name,
            layer=layer_path,
        )


def _resolve_layer(model: object, layer_path: str) -> object:
    """Navigate a model's attribute tree by dot-separated path.

    Integer segments are treated as sequence indices.
    Example: "transformer.h.0" → model.transformer.h[0]
    """
    node = model
    for part in layer_path.split("."):
        if part.isdigit():
            node = node[int(part)]
        else:
            node = getattr(node, part)
    return node

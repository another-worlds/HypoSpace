from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Union


def nnsight_available() -> bool:
    """Return True if nnsight and torch are importable."""
    try:
        import nnsight  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class _NNSightInputCache:
    """Input-keyed disk cache for live activation tensors.

    Keys on (model_name, layer_path, inputs, token_index) so cache hits
    skip the forward pass entirely.
    """

    def __init__(self, cache_dir: str) -> None:
        self.root = Path(cache_dir) / "nnsight"
        self.root.mkdir(parents=True, exist_ok=True)

    def key(
        self,
        model_name: str,
        layer_path: str,
        inputs: str | Iterable[int],
        token_index: int,
    ) -> str:
        inputs_repr: str | list[int] = inputs if isinstance(inputs, str) else list(inputs)
        raw = json.dumps(
            {
                "model": model_name,
                "layer_path": layer_path,
                "inputs": inputs_repr,
                "token_index": token_index,
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get(self, key: str) -> List[float] | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("values")

    def put(self, key: str, values: List[float]) -> None:
        path = self.root / f"{key}.json"
        path.write_text(json.dumps({"values": values}, ensure_ascii=False), encoding="utf-8")


class NNSightExtractor:
    """Live activation extractor using nnsight model tracing.

    Requires nnsight and torch. Caches by input key so repeated calls
    with the same (model, layer, text, token_index) skip the forward pass.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        cache_dir: str = ".hypo_cache",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._cache = _NNSightInputCache(cache_dir)
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
        inputs: str | Iterable[int],
        layer_path: str,
        token_index: int = -1,
    ) -> List[float]:
        """Trace a forward pass and return activations at layer_path.

        Cache is checked before the forward pass; on a hit the model is
        not called at all.

        Args:
            inputs: Text string or token id sequence.
            layer_path: Dot-notation attribute path on the model
                (e.g. "transformer.h.0", "model.layers.3").
            token_index: Token position to extract; -1 means last token.

        Returns:
            Flat list of float activation values for the given token.
        """
        return self.extract_layers(inputs, [layer_path], token_index)[layer_path]

    def extract_layers(
        self,
        inputs: str | Iterable[int],
        layer_paths: list[str],
        token_index: int = -1,
    ) -> Dict[str, List[float]]:
        """Extract activations from multiple layers in a single forward pass.

        Layers already in the cache are returned immediately without running
        the model. Only uncached layers trigger a trace, and all uncached
        layers are captured together in one forward pass.

        Args:
            inputs: Text string or token id sequence.
            layer_paths: Dot-notation layer paths to extract.
            token_index: Token position to extract; -1 means last token.

        Returns:
            Mapping from layer_path to flat float activation list.
        """
        result: Dict[str, List[float]] = {}
        uncached: List[str] = []

        for lp in layer_paths:
            k = self._cache.key(self.model_name, lp, inputs, token_index)
            cached = self._cache.get(k)
            if cached is not None:
                result[lp] = cached
            else:
                uncached.append(lp)

        if uncached:
            raw_tensors = self._trace(inputs, uncached)
            for lp, tensor in raw_tensors.items():
                values = _tensor_to_floats(tensor, token_index)
                self._cache.put(
                    self._cache.key(self.model_name, lp, inputs, token_index),
                    values,
                )
                result[lp] = values

        return result

    def _trace(
        self,
        inputs: str | Iterable[int],
        layer_paths: list[str],
    ) -> Dict[str, object]:
        """Run one nnsight forward pass and return raw tensors per layer."""
        lm = self._load_model()
        saved: Dict[str, object] = {}
        with lm.trace(inputs):
            for lp in layer_paths:
                saved[lp] = _resolve_layer(lm, lp).output.save()
        return saved


def _tensor_to_floats(raw: object, token_index: int) -> List[float]:
    """Convert a layer output (tensor or tuple) to a flat float list."""
    import torch

    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, torch.Tensor):
        raw = raw.detach()
        if raw.dim() == 3:
            raw = raw[0]  # drop batch dim
        return raw[token_index].tolist()
    return list(raw)  # type: ignore[arg-type]


def _resolve_layer(model: object, layer_path: str) -> object:
    """Navigate a model's attribute tree by dot-separated path.

    Integer segments are treated as sequence indices.
    Example: "transformer.h.0" → model.transformer.h[0]
    """
    node = model
    for part in layer_path.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node

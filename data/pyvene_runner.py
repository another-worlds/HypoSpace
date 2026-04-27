from __future__ import annotations

"""PyVeneInterventionRunner: real zero-ablation interventions via pyvene or PyTorch hooks."""

import warnings
from typing import Any, Iterable, List, Union

from core.hierarchy import Feature
from data.utils import resolve_layer
from interpretability.mechanistic import InterventionResult


def pyvene_available() -> bool:
    try:
        import pyvene  # noqa: F401
        import torch   # noqa: F401
        return True
    except ImportError:
        return False


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class PyVeneInterventionRunner:
    """Real zero-ablation mechanistic interventions via pyvene or PyTorch hooks.

    Borrows an already-loaded nnsight LanguageModel — does not create or own it.
    When pyvene is installed, uses its ZeroIntervention API; otherwise falls back
    to raw PyTorch forward hooks, which achieve the same effect.
    """

    def __init__(self, lm: Any, layer_path: str, device: str = "cpu") -> None:
        self.lm = lm
        self.layer_path = layer_path
        self.device = device

    @classmethod
    def require(cls) -> None:
        """Raise ImportError with install hint if torch is missing."""
        if not _torch_available():
            raise ImportError(
                "torch is required for real interventions. "
                "Install with: pip install torch"
            )

    def run_interventions(
        self,
        features: List[Feature],
        inputs: Union[str, Iterable[int]],
        token_index: int = -1,
    ) -> List[InterventionResult]:
        """Zero-ablate each feature's activation dimension and measure logit effect.

        For each feature, runs one ablated forward pass (zero-ing the dimension
        corresponding to feature.source_index at self.layer_path) and computes
        effect_size as the mean absolute logit difference from baseline.

        Args:
            features: Top-k features from a DecodeResult.
            inputs: Text or token ids passed to the model.
            token_index: Token position used for ablation hook targeting.

        Returns:
            One InterventionResult per feature with a real effect_size.
        """
        import torch

        baseline_logits = self._get_baseline_logits(inputs)
        results: List[InterventionResult] = []

        for feature in features:
            if pyvene_available():
                ablated_logits = self._ablate_with_pyvene(inputs, feature.source_index)
            else:
                ablated_logits = self._ablate_with_hooks(inputs, feature.source_index)

            effect_size = torch.mean(
                torch.abs(baseline_logits - ablated_logits)
            ).item()

            results.append(
                InterventionResult(
                    feature_id=feature.id,
                    baseline=feature.score,
                    ablated=0.0,
                    effect_size=float(effect_size),
                )
            )

        return results

    def _get_baseline_logits(self, inputs: Union[str, Iterable[int]]) -> Any:
        """Run a plain forward pass and return the output logits tensor."""
        import torch

        with self.lm.trace(inputs):
            out = self.lm.output.logits.save()

        logits = out if isinstance(out, torch.Tensor) else out[0]
        if logits.dim() == 3:
            logits = logits[0]  # drop batch dim
        return logits[-1]  # last token position

    def _ablate_with_pyvene(
        self, inputs: Union[str, Iterable[int]], source_index: int
    ) -> Any:
        """Zero-ablate source_index using pyvene's ZeroIntervention."""
        try:
            return self._ablate_with_pyvene_impl(inputs, source_index)
        except Exception as exc:
            warnings.warn(
                f"pyvene intervention failed ({exc!r}); falling back to hooks",
                stacklevel=2,
            )
            return self._ablate_with_hooks(inputs, source_index)

    def _ablate_with_pyvene_impl(
        self, inputs: Union[str, Iterable[int]], source_index: int
    ) -> Any:
        import pyvene as pv
        import torch

        hf_model = self.lm._module
        layer_module = resolve_layer(hf_model, self.layer_path)
        layer_index = _layer_index(self.layer_path)

        config = pv.IntervenableConfig(
            representations=[
                {
                    "layer": layer_index,
                    "component": "block_output",
                    "intervention_type": pv.ZeroIntervention,
                }
            ]
        )
        intervenable = pv.IntervenableModel(config, hf_model)

        if isinstance(inputs, str):
            tokenizer = self.lm.tokenizer
            encoded = tokenizer(inputs, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
        else:
            input_ids = torch.tensor([list(inputs)], device=self.device)

        with torch.no_grad():
            _, outputs = intervenable(
                {"input_ids": input_ids},
                unit_locations={"sources->base": (None, [[[source_index]]])},
            )

        logits = outputs.logits
        if logits.dim() == 3:
            logits = logits[0]
        return logits[-1]

    def _ablate_with_hooks(
        self, inputs: Union[str, Iterable[int]], source_index: int
    ) -> Any:
        """Zero-ablate source_index at the target layer via a PyTorch forward hook.

        Runs directly on the underlying HuggingFace model (self.lm._module) to
        avoid conflicts between registered hooks and nnsight's trace context.
        """
        import torch

        hf_model = self.lm._module
        layer_module = resolve_layer(hf_model, self.layer_path)

        def _zero_dim(module: Any, input: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                lst = list(output)
                lst[0] = lst[0].clone()
                lst[0][..., source_index] = 0.0
                return tuple(lst)
            cloned = output.clone()
            cloned[..., source_index] = 0.0
            return cloned

        if isinstance(inputs, str):
            encoded = self.lm.tokenizer(inputs, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
        else:
            input_ids = torch.tensor([list(inputs)], device=self.device)

        handle = layer_module.register_forward_hook(_zero_dim)
        try:
            with torch.no_grad():
                out = hf_model(input_ids)
        finally:
            handle.remove()

        logits = out.logits if hasattr(out, "logits") else out[0]
        if logits.dim() == 3:
            logits = logits[0]
        return logits[-1]


def _layer_index(layer_path: str) -> int:
    """Extract the first integer segment from a layer path as the layer index."""
    for part in layer_path.split("."):
        if part.isdigit():
            return int(part)
    warnings.warn(
        f"No integer segment found in layer_path '{layer_path}'; defaulting layer index to 0",
        stacklevel=2,
    )
    return 0

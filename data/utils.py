from __future__ import annotations

"""Shared utilities: utc_timestamp() and resolve_layer() for dot-notation model traversal."""

from datetime import datetime, timezone
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_layer(model: Any, layer_path: str) -> Any:
    """Navigate a model's attribute tree by dot-separated path (integers → list index).

    Example: "transformer.h.0" → model.transformer.h[0]
    """
    node = model
    for part in layer_path.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node

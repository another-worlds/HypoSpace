from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List


def diskcache_available() -> bool:
    try:
        import diskcache  # noqa: F401
        return True
    except ImportError:
        return False


class ActivationExtractor:
    """nnsight-facing extractor interface with lightweight disk cache."""

    def __init__(self, cache_dir: str = ".hypo_cache") -> None:
        self.cache_root = Path(cache_dir) / "activations"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._dc: object | None = None
        if diskcache_available():
            import diskcache
            self._dc = diskcache.Cache(str(self.cache_root))

    def extract(
        self,
        raw_activations: Iterable[float],
        model_name: str = "unknown-model",
        layer: str = "unknown-layer",
    ) -> List[float]:
        values = [float(item) for item in raw_activations]
        cache_key = self._cache_key(values=values, model_name=model_name, layer=layer)

        if self._dc is not None:
            cached = self._dc.get(cache_key)
            if cached is not None:
                return cached
            self._dc.set(cache_key, values)
            return values

        cache_path = self.cache_root / f"{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return [float(item) for item in payload.get("values", [])]

        cache_path.write_text(json.dumps({"values": values}, ensure_ascii=False), encoding="utf-8")
        return values

    @staticmethod
    def _cache_key(values: List[float], model_name: str, layer: str) -> str:
        raw = json.dumps({"model": model_name, "layer": layer, "values": values}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

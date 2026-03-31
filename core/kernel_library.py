from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from core.hierarchy import Feature


@dataclass(slots=True)
class KernelTemplate:
    """Persistable concept kernel artifact."""

    kernel_id: str
    version: str
    model_name: str
    layer: str
    features: List[Feature] = field(default_factory=list)


class KernelLibrary:
    """Persistent storage for kernel templates."""

    MANIFEST_NAME = "manifest.json"

    def __init__(self, root: str | Path = ".hypo_kernels") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / self.MANIFEST_NAME

    def save(self, template: KernelTemplate) -> Path:
        output = self.root / f"{template.kernel_id}-{template.version}.json"
        payload = asdict(template)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._update_manifest(template, output.name)
        return output

    def load(self, kernel_id: str, version: str) -> KernelTemplate:
        input_path = self.root / f"{kernel_id}-{version}.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Kernel artifact not found: {input_path}")
        data = json.loads(input_path.read_text(encoding="utf-8"))
        features = [Feature(**item) for item in data.pop("features", [])]
        return KernelTemplate(features=features, **data)

    def list_kernels(self) -> Dict[str, List[str]]:
        manifest = self._read_manifest()
        result: Dict[str, List[str]] = {}
        for kernel_id, payload in manifest.items():
            versions = list(payload.get("versions", []))
            result[kernel_id] = self._sorted_versions(versions)
        return result

    def load_latest(self, kernel_id: str) -> KernelTemplate:
        manifest = self._read_manifest()
        if kernel_id not in manifest:
            raise KeyError(f"Kernel id not found in manifest: {kernel_id}")
        payload = manifest[kernel_id]
        latest_version = payload["latest"]
        return self.load(kernel_id, latest_version)

    def match(self, features: Iterable[Feature], threshold: float = 0.8) -> Dict[str, float]:
        """MVP placeholder matcher returning normalized score by feature id."""
        return {feature.id: min(1.0, feature.score / max(threshold, 1e-6)) for feature in features}

    def _read_manifest(self) -> Dict[str, Dict[str, object]]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _update_manifest(self, template: KernelTemplate, filename: str) -> None:
        manifest = self._read_manifest()
        payload = manifest.setdefault(template.kernel_id, {"latest": template.version, "versions": [], "files": {}})

        versions = payload["versions"]
        if template.version not in versions:
            versions.append(template.version)
        sorted_versions = self._sorted_versions(versions)
        payload["versions"] = sorted_versions
        payload["latest"] = sorted_versions[-1]
        payload["files"][template.version] = filename

        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _sorted_versions(versions: List[str]) -> List[str]:
        return sorted(versions, key=KernelLibrary._version_key)

    @staticmethod
    def _version_key(version: str) -> Tuple[int, ...]:
        chunks = version.split(".")
        out: List[int] = []
        for chunk in chunks:
            try:
                out.append(int(chunk))
            except ValueError:
                out.append(0)
        while len(out) < 3:
            out.append(0)
        return tuple(out)

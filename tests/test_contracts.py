import json
from pathlib import Path

from api import HypoSpaceAPI
from core.config import DecoderConfig, RuntimeConfig
from core.hierarchy import Feature
from viz.canvas import SemanticCanvas


def test_kernel_artifact_contract(tmp_path: Path) -> None:
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path)), top_k=3))
    result = api.decode(model_name="demo", layer="layer_3", raw_activations=[0.1, 0.5, -0.9], version="0.1.0")

    assert result.kernel_path is not None
    payload = json.loads(Path(result.kernel_path).read_text(encoding="utf-8"))

    assert payload["kernel_id"] == "demo-layer_3"
    assert payload["version"] == "0.1.0"
    assert payload["model_name"] == "demo"
    assert payload["layer"] == "layer_3"
    assert isinstance(payload["features"], list)
    assert {"id", "layer", "score", "source_index", "label"}.issubset(payload["features"][0].keys())


def test_semantic_canvas_contract() -> None:
    canvas = SemanticCanvas()
    features = [
        Feature(id="a", layer="L0", score=0.2, source_index=0),
        Feature(id="b", layer="L0", score=0.7, source_index=1),
        Feature(id="c", layer="L0", score=0.5, source_index=2),
    ]

    points = canvas.to_points(features)
    edges = canvas.to_edges(features)

    assert points == [("b", 0.7), ("c", 0.5), ("a", 0.2)]
    assert edges == [("b", "c", 0.6), ("c", "a", 0.35)]

import pytest

from api import HypoSpaceAPI
from core.config import DecoderConfig, GovernanceConfig, RuntimeConfig
from core.hierarchy import Feature
from core.kernel_library import KernelLibrary, KernelTemplate
from interpretability.faithfulness import GovernanceThresholdError


def test_api_smoke_flow(tmp_path) -> None:
    config = DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path)), top_k=2)
    api = HypoSpaceAPI(config=config)
    run = api.decode_and_score("demo", "layer_0", [0.1, -0.9, 0.3], version="0.1.1")

    assert len(run.decode.features) == 2
    assert run.decode.kernel_path
    assert run.scorecard.risk_flag in {"ok", "low_faithfulness"}


def test_kernel_manifest_semver_and_latest_load(tmp_path) -> None:
    config = DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path)), top_k=1)
    api = HypoSpaceAPI(config=config)

    api.decode("demo", "layer_1", [0.1, 0.2, 0.3], version="0.2.0")
    api.decode("demo", "layer_1", [0.7, 0.2, 0.1], version="0.10.0")
    api.decode("demo", "layer_1", [0.9, 0.2, 0.1], version="0.3.0")

    lib = KernelLibrary(root=str(tmp_path))
    versions = lib.list_kernels()["demo-layer_1"]
    latest = lib.load_latest("demo-layer_1")

    assert versions == ["0.2.0", "0.3.0", "0.10.0"]
    assert latest.version == "0.10.0"
    assert latest.layer == "layer_1"


def test_kernel_match_and_merge(tmp_path) -> None:
    config = DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path)), top_k=2)
    api = HypoSpaceAPI(config=config)

    first = api.decode("demo", "layer_4", [0.1, 0.9, 0.2], version="0.1.0")
    second = api.decode("demo", "layer_4", [0.2, 0.85, 0.1], version="0.2.0")

    lib = KernelLibrary(root=str(tmp_path))
    match = lib.match(kernel_id="demo-layer_4", features=second.features)
    merged = lib.merge("demo-layer_4", base_version="0.1.0", candidate_version="0.2.0", merged_version="0.3.0")

    assert all(0.0 <= score <= 1.0 for score in match.values())
    assert merged.version == "0.3.0"
    assert merged.metadata["merge_base_version"] == "0.1.0"
    assert merged.metadata["merge_candidate_version"] == "0.2.0"
    assert len(merged.features) >= len(first.features)


def test_kernel_merge_semantics(tmp_path) -> None:
    """merge() keeps the higher-scoring feature per source_index and includes candidate-only indices."""
    lib = KernelLibrary(root=tmp_path / "lib")

    base = KernelTemplate(
        kernel_id="model-layer",
        version="1.0.0",
        model_name="model",
        layer="layer",
        features=[
            Feature(id="layer:feature:0", layer="layer", score=0.9, source_index=0),  # shared, base wins
            Feature(id="layer:feature:1", layer="layer", score=0.3, source_index=1),  # shared, candidate wins
        ],
    )
    candidate = KernelTemplate(
        kernel_id="model-layer",
        version="2.0.0",
        model_name="model",
        layer="layer",
        features=[
            Feature(id="layer:feature:0", layer="layer", score=0.5, source_index=0),  # lower → base kept
            Feature(id="layer:feature:1", layer="layer", score=0.8, source_index=1),  # higher → candidate wins
            Feature(id="layer:feature:2", layer="layer", score=0.6, source_index=2),  # candidate-only → added
        ],
    )
    lib.save(base)
    lib.save(candidate)

    merged = lib.merge("model-layer", base_version="1.0.0", candidate_version="2.0.0", merged_version="3.0.0")

    by_index = {f.source_index: f for f in merged.features}
    assert by_index[0].score == pytest.approx(0.9)   # base score was higher
    assert by_index[1].score == pytest.approx(0.8)   # candidate score was higher
    assert 2 in by_index                              # candidate-only index included
    scores = [f.score for f in merged.features]
    assert scores == sorted(scores, reverse=True)     # sorted descending


def test_fail_on_low_confidence_raises(tmp_path) -> None:
    config = DecoderConfig(
        runtime=RuntimeConfig(cache_dir=str(tmp_path)),
        governance=GovernanceConfig(
            min_faithfulness_score=0.99,
            min_stability_score=0.99,
            fail_on_low_confidence=True,
        ),
    )
    api = HypoSpaceAPI(config=config)

    with pytest.raises(GovernanceThresholdError):
        api.decode_and_score("demo", "layer_2", [0.2, 0.1, 0.05])


def test_api_sae_path_none_falls_back_gracefully(tmp_path) -> None:
    """DecoderConfig.sae_path=None causes silent magnitude fallback; pipeline still works."""
    config = DecoderConfig(
        backend="matryoshka",
        sae_path=None,
        top_k=2,
        runtime=RuntimeConfig(cache_dir=str(tmp_path)),
    )
    api = HypoSpaceAPI(config=config)
    result = api.decode_and_score("demo", "layer_0", [0.1, -0.9, 0.3])
    assert len(result.decode.features) == 2
    assert all(":feature:" in f.id for f in result.decode.features)

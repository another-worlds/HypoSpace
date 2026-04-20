from __future__ import annotations

import pytest

from api import HypoSpaceResult
from data.nnsight_extractor import NNSightExtractor, nnsight_available

# All tests in this module are skipped when nnsight/torch are not installed.
pytestmark = pytest.mark.skipif(
    not nnsight_available(),
    reason="nnsight and torch are required for these tests",
)

_MODEL = "gpt2"
_LAYER = "transformer.h.0"
_HIDDEN_DIM = 768  # GPT-2 small hidden size


def test_nnsight_available() -> None:
    assert nnsight_available()


def test_extract_returns_float_list(tmp_path) -> None:
    extractor = NNSightExtractor(_MODEL, device="cpu", cache_dir=str(tmp_path))
    values = extractor.extract("Hello world", layer_path=_LAYER, token_index=-1)

    assert isinstance(values, list)
    assert len(values) == _HIDDEN_DIM
    assert all(isinstance(v, float) for v in values)


def test_extract_last_vs_first_token_differ(tmp_path) -> None:
    extractor = NNSightExtractor(_MODEL, device="cpu", cache_dir=str(tmp_path))
    last = extractor.extract("Hello world", layer_path=_LAYER, token_index=-1)
    first = extractor.extract("Hello world", layer_path=_LAYER, token_index=0)

    assert last != first


def test_extract_caches_result(tmp_path) -> None:
    extractor = NNSightExtractor(_MODEL, device="cpu", cache_dir=str(tmp_path))
    first = extractor.extract("Hello world", layer_path=_LAYER)
    second = extractor.extract("Hello world", layer_path=_LAYER)

    assert first == second


def test_extract_different_layers_differ(tmp_path) -> None:
    extractor = NNSightExtractor(_MODEL, device="cpu", cache_dir=str(tmp_path))
    layer0 = extractor.extract("Hello world", layer_path="transformer.h.0")
    layer1 = extractor.extract("Hello world", layer_path="transformer.h.1")

    assert layer0 != layer1


def test_require_raises_when_unavailable(monkeypatch) -> None:
    # Patch nnsight_available to pretend nnsight is missing.
    import data.nnsight_extractor as mod
    monkeypatch.setattr(mod, "nnsight_available", lambda: False)

    with pytest.raises(ImportError, match="pip install nnsight"):
        NNSightExtractor.require()


def test_api_decode_from_model(tmp_path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig

    api = HypoSpaceAPI(
        config=DecoderConfig(
            runtime=RuntimeConfig(cache_dir=str(tmp_path)),
            top_k=5,
        )
    )
    result = api.decode_from_model(
        model_name=_MODEL,
        layer="layer_0",
        layer_path=_LAYER,
        inputs="The quick brown fox",
        version="0.1.0",
    )

    assert len(result.features) == 5
    assert result.kernel_path is not None
    assert all(f.label is not None for f in result.features)


def test_api_decode_from_model_scorecard(tmp_path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig

    api = HypoSpaceAPI(
        config=DecoderConfig(
            runtime=RuntimeConfig(cache_dir=str(tmp_path)),
            top_k=4,
        )
    )
    decode_result = api.decode_from_model(
        model_name=_MODEL,
        layer="layer_0",
        layer_path=_LAYER,
        inputs="Test sentence",
    )
    scorecard = api.scorecard(decode_result)

    assert 0.0 <= scorecard.faithfulness_score <= 1.0
    assert 0.0 <= scorecard.stability_score <= 1.0
    assert scorecard.risk_flag in {"ok", "low_faithfulness"}


def test_extract_cache_skips_forward_pass(tmp_path) -> None:
    extractor = NNSightExtractor(_MODEL, device="cpu", cache_dir=str(tmp_path))
    # Warm the cache with a real forward pass.
    extractor.extract("Hello world", layer_path=_LAYER)

    # Replace _trace with a spy — it must not be called on a cache hit.
    trace_calls: list[list[str]] = []
    original_trace = extractor._trace

    def spy_trace(inputs: object, layer_paths: list[str]) -> object:
        trace_calls.append(layer_paths)
        return original_trace(inputs, layer_paths)

    extractor._trace = spy_trace  # type: ignore[method-assign]
    extractor.extract("Hello world", layer_path=_LAYER)

    assert trace_calls == [], "forward pass was run despite cached entry"


def test_extract_layers_single_pass(tmp_path) -> None:
    extractor = NNSightExtractor(_MODEL, device="cpu", cache_dir=str(tmp_path))

    trace_calls: list[list[str]] = []
    original_trace = extractor._trace

    def spy_trace(inputs: object, layer_paths: list[str]) -> object:
        trace_calls.append(list(layer_paths))
        return original_trace(inputs, layer_paths)

    extractor._trace = spy_trace  # type: ignore[method-assign]
    result = extractor.extract_layers(
        "Hello world",
        layer_paths=["transformer.h.0", "transformer.h.1"],
    )

    assert len(trace_calls) == 1, "expected exactly one forward pass for two layers"
    assert set(trace_calls[0]) == {"transformer.h.0", "transformer.h.1"}
    assert len(result) == 2
    assert result["transformer.h.0"] != result["transformer.h.1"]


def test_decode_and_score_from_model(tmp_path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig

    api = HypoSpaceAPI(
        config=DecoderConfig(
            runtime=RuntimeConfig(cache_dir=str(tmp_path)),
            top_k=4,
        )
    )
    result = api.decode_and_score_from_model(
        model_name=_MODEL,
        layer="layer_0",
        layer_path=_LAYER,
        inputs="The quick brown fox",
    )

    assert isinstance(result, HypoSpaceResult)
    assert len(result.decode.features) == 4
    assert result.decode.kernel_path is not None
    assert result.scorecard.risk_flag in {"ok", "low_faithfulness"}

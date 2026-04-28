from __future__ import annotations

import math
import types
from pathlib import Path

import pytest

from core.hierarchy import Feature, HierarchyEngine
from core.kernel_library import KernelLibrary
from data.extractor import ActivationExtractor
from data.preprocessor import ActivationPreprocessor
from data.utils import resolve_layer
from interpretability.faithfulness import FaithfulnessChecker
from interpretability.mechanistic import InterventionResult
from interpretability.semantic import SemanticInterpreter


# ---------------------------------------------------------------------------
# ActivationPreprocessor — boundary values
# ---------------------------------------------------------------------------

def test_normalize_nan_propagates() -> None:
    result = ActivationPreprocessor().normalize([float("nan"), 1.0])
    assert any(math.isnan(v) for v in result)


def test_normalize_single_value() -> None:
    assert ActivationPreprocessor().normalize([0.5]) == [1.0]


def test_normalize_negative_single_value() -> None:
    assert ActivationPreprocessor().normalize([-0.5]) == [-1.0]


def test_normalize_all_zeros_returns_zeros() -> None:
    assert ActivationPreprocessor().normalize([0.0, 0.0]) == [0.0, 0.0]


# ---------------------------------------------------------------------------
# HierarchyEngine — boundary values
# ---------------------------------------------------------------------------

def test_top_k_larger_than_input_returns_all() -> None:
    feats = HierarchyEngine().extract_features([0.1, 0.2], layer="l", top_k=10)
    assert len(feats) == 2


def test_negative_values_scored_by_magnitude() -> None:
    feats = HierarchyEngine().extract_features([-0.9, 0.1], layer="l", top_k=1)
    assert feats[0].source_index == 0  # -0.9 has highest abs magnitude


def test_top_k_zero_raises() -> None:
    with pytest.raises(ValueError):
        HierarchyEngine().extract_features([0.5], layer="l", top_k=0)


def test_top_k_negative_raises() -> None:
    with pytest.raises(ValueError):
        HierarchyEngine().extract_features([0.5], layer="l", top_k=-1)


def test_empty_input_returns_empty() -> None:
    assert HierarchyEngine().extract_features([], layer="l", top_k=4) == []


# ---------------------------------------------------------------------------
# SemanticInterpreter — boundary scores
# ---------------------------------------------------------------------------

def test_score_at_high_boundary() -> None:
    feat = Feature(id="f", layer="l", score=0.8, source_index=0)
    labeled = SemanticInterpreter().annotate([feat])
    assert "high-intensity" in labeled[0].label  # type: ignore[index]


def test_score_just_below_high_boundary() -> None:
    feat = Feature(id="f", layer="l", score=0.79, source_index=0)
    labeled = SemanticInterpreter().annotate([feat])
    assert "medium-intensity" in labeled[0].label  # type: ignore[index]


def test_score_at_medium_boundary() -> None:
    feat = Feature(id="f", layer="l", score=0.4, source_index=0)
    labeled = SemanticInterpreter().annotate([feat])
    assert "medium-intensity" in labeled[0].label  # type: ignore[index]


def test_score_just_below_medium_boundary() -> None:
    feat = Feature(id="f", layer="l", score=0.39, source_index=0)
    labeled = SemanticInterpreter().annotate([feat])
    assert "low-intensity" in labeled[0].label  # type: ignore[index]


def test_custom_high_threshold_reclassifies_score() -> None:
    interp = SemanticInterpreter(high=0.5, medium=0.2)
    feat = Feature(id="f", layer="l", score=0.6, source_index=0)
    labeled = interp.annotate([feat])
    assert "high-intensity" in labeled[0].label  # type: ignore[index]


def test_custom_medium_threshold_reclassifies_score() -> None:
    interp = SemanticInterpreter(high=0.9, medium=0.3)
    feat = Feature(id="f", layer="l", score=0.5, source_index=0)
    labeled = interp.annotate([feat])
    assert "medium-intensity" in labeled[0].label  # type: ignore[index]


def test_thresholds_from_config_propagate() -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig
    api = HypoSpaceAPI(config=DecoderConfig(high_intensity_threshold=0.5, medium_intensity_threshold=0.2))
    feat = Feature(id="f", layer="l", score=0.6, source_index=0)
    labeled = api.semantic.annotate([feat])
    assert "high-intensity" in labeled[0].label  # type: ignore[index]


def test_existing_label_not_overwritten() -> None:
    feat = Feature(id="f", layer="l", score=0.9, source_index=0, label="keep-me")
    labeled = SemanticInterpreter().annotate([feat])
    assert labeled[0].label == "keep-me"


# ---------------------------------------------------------------------------
# FaithfulnessChecker — empty and near-zero
# ---------------------------------------------------------------------------

def test_empty_interventions_returns_no_data() -> None:
    sc = FaithfulnessChecker().evaluate([])
    assert sc.risk_flag == "no_data"
    assert sc.faithfulness_score == 0.0


def test_near_zero_mean_returns_perfect_stability() -> None:
    results = [InterventionResult("f", baseline=0.00001, ablated=0.0, effect_size=0.00001)]
    sc = FaithfulnessChecker().evaluate(results)
    assert sc.stability_score == 1.0


def test_intervention_method_propagated_to_scorecard() -> None:
    results = [InterventionResult("f", baseline=0.5, ablated=0.25, effect_size=0.25)]
    sc = FaithfulnessChecker().evaluate(results, intervention_method="pyvene-zero-ablation")
    assert sc.intervention_method == "pyvene-zero-ablation"


# ---------------------------------------------------------------------------
# KernelLibrary — load nonexistent
# ---------------------------------------------------------------------------

def test_load_latest_missing_raises(tmp_path: Path) -> None:
    lib = KernelLibrary(root=tmp_path)
    with pytest.raises(KeyError):
        lib.load_latest("nonexistent-kernel")


# ---------------------------------------------------------------------------
# ActivationExtractor — cache round-trip
# ---------------------------------------------------------------------------

def test_cache_hit_returns_same_values(tmp_path: Path) -> None:
    ex = ActivationExtractor(cache_dir=str(tmp_path))
    ex.extract([0.1, 0.5], "model", "layer")
    hit = ex.extract([0.1, 0.5], "model", "layer")
    assert hit == [0.1, 0.5]


def test_different_inputs_different_cache_entries(tmp_path: Path) -> None:
    ex = ActivationExtractor(cache_dir=str(tmp_path))
    a = ex.extract([0.1, 0.5], "model", "layer")
    b = ex.extract([0.9, 0.2], "model", "layer")
    assert a != b


def test_cache_key_differs_by_model(tmp_path: Path) -> None:
    key_a = ActivationExtractor._cache_key([0.1, 0.5], "model-a", "layer")
    key_b = ActivationExtractor._cache_key([0.1, 0.5], "model-b", "layer")
    assert key_a != key_b


def test_cache_key_differs_by_layer(tmp_path: Path) -> None:
    key_a = ActivationExtractor._cache_key([0.1, 0.5], "model", "layer-0")
    key_b = ActivationExtractor._cache_key([0.1, 0.5], "model", "layer-1")
    assert key_a != key_b


def test_cache_key_same_for_identical_inputs() -> None:
    key_a = ActivationExtractor._cache_key([0.1, 0.5], "model", "layer")
    key_b = ActivationExtractor._cache_key([0.1, 0.5], "model", "layer")
    assert key_a == key_b


# ---------------------------------------------------------------------------
# KernelLibrary — corrupt file handling
# ---------------------------------------------------------------------------

def test_load_corrupt_kernel_raises_value_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "bad-model-layer-1.0.0.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    lib = KernelLibrary(root=tmp_path)
    with pytest.raises(ValueError, match="Corrupt kernel artifact"):
        lib.load("bad-model-layer", "1.0.0")


# ---------------------------------------------------------------------------
# HypoSpaceAPI — input validation
# ---------------------------------------------------------------------------

def test_empty_model_name_raises(tmp_path: Path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path))))
    with pytest.raises(ValueError, match="model_name"):
        api.decode("", "layer_0", [0.1, 0.2])


def test_empty_layer_raises(tmp_path: Path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path))))
    with pytest.raises(ValueError, match="layer"):
        api.decode("model", "", [0.1, 0.2])


def test_invalid_semver_raises(tmp_path: Path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path))))
    with pytest.raises(ValueError, match="semver"):
        api.decode("model", "layer", [0.1, 0.2], version="not-a-version")


def test_valid_semver_passes(tmp_path: Path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path))))
    result = api.decode("model", "layer", [0.1, 0.2], version="1.2.3")
    assert result is not None


def test_oversized_activations_raises(tmp_path: Path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path), max_features=4)))
    with pytest.raises(ValueError, match="max_features"):
        api.decode("model", "layer", [0.1, 0.2, 0.3, 0.4, 0.5])


# ---------------------------------------------------------------------------
# resolve_layer — valid paths, list indexing, invalid path raises ValueError
# ---------------------------------------------------------------------------

def test_resolve_layer_attribute_path() -> None:
    model = types.SimpleNamespace(transformer=types.SimpleNamespace(h=["block0", "block1"]))
    assert resolve_layer(model, "transformer.h.0") == "block0"


def test_resolve_layer_integer_index() -> None:
    model = types.SimpleNamespace(layers=["a", "b", "c"])
    assert resolve_layer(model, "layers.2") == "c"


def test_resolve_layer_bad_attribute_raises_value_error() -> None:
    model = types.SimpleNamespace()
    with pytest.raises(ValueError, match="resolve_layer"):
        resolve_layer(model, "nonexistent.path")


def test_resolve_layer_bad_index_raises_value_error() -> None:
    model = types.SimpleNamespace(h=["only_one"])
    with pytest.raises(ValueError, match="resolve_layer"):
        resolve_layer(model, "h.99")


# ---------------------------------------------------------------------------
# RealityDecoder — cross-run match KeyError path (first run, no prior kernel)
# ---------------------------------------------------------------------------

def test_decode_first_run_match_rate_is_zero(tmp_path: Path) -> None:
    from core.config import DecoderConfig, RuntimeConfig
    from core.decoder import RealityDecoder

    decoder = RealityDecoder(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path))))
    result = decoder.decode("fresh-model", "layer_0", [0.1, 0.4, -0.2, 0.8])
    assert result.metadata["cross_run_match_rate"] == "0.000"


# ---------------------------------------------------------------------------
# FeatureBackend Protocol + HierarchyEngine injection (stdlib-only, no torch)
# ---------------------------------------------------------------------------

def test_hierarchy_engine_no_backend_falls_back_to_magnitude() -> None:
    engine = HierarchyEngine(backend="matryoshka", feature_backend=None)
    feats = engine.extract_features([0.3, -0.9, 0.1], layer="l", top_k=2)
    assert len(feats) == 2
    assert feats[0].source_index == 1  # abs(-0.9) is largest


def test_hierarchy_engine_injected_backend_is_called() -> None:
    class _FixedBackend:
        def extract(self, activations, layer, top_k):
            return [Feature(id="fixed:sae:0:99", layer=layer, score=1.0, source_index=99)]

    engine = HierarchyEngine(feature_backend=_FixedBackend())
    feats = engine.extract_features([0.1, 0.2, 0.3], layer="test-layer", top_k=3)
    assert len(feats) == 1
    assert feats[0].source_index == 99


def test_feature_backend_protocol_runtime_checkable() -> None:
    from core.hierarchy import FeatureBackend
    from data.sae_backend import MagnitudeBackend
    assert isinstance(MagnitudeBackend(), FeatureBackend)


def test_magnitude_backend_extract_top_k_clipped() -> None:
    from data.sae_backend import MagnitudeBackend
    feats = MagnitudeBackend().extract([0.1, 0.9, -0.7, 0.5], "l", top_k=2)
    assert len(feats) == 2
    assert feats[0].source_index == 1  # 0.9 is highest magnitude


def test_magnitude_backend_extract_empty_returns_empty() -> None:
    from data.sae_backend import MagnitudeBackend
    assert MagnitudeBackend().extract([], "l", top_k=4) == []


# ---------------------------------------------------------------------------
# PyVeneInterventionRunner — ablation_mode (stdlib-safe, no torch needed)
# ---------------------------------------------------------------------------

def test_pyvene_runner_intervention_method_property() -> None:
    from data.pyvene_runner import PyVeneInterventionRunner
    assert PyVeneInterventionRunner(None, "x", ablation_mode="hooks").intervention_method == "hook-zero-ablation"
    assert PyVeneInterventionRunner(None, "x", ablation_mode="pyvene_token").intervention_method == "pyvene-token-ablation"


def test_pyvene_runner_unknown_ablation_mode_raises() -> None:
    from data.pyvene_runner import PyVeneInterventionRunner
    with pytest.raises(ValueError, match="ablation_mode"):
        PyVeneInterventionRunner(None, "x", ablation_mode="bad_mode")


# ---------------------------------------------------------------------------
# ISSUE-P01 — decode_and_score() stub fallback (stdlib-safe, no torch needed)
# ---------------------------------------------------------------------------

def test_decode_and_score_no_layer_path_uses_stub(tmp_path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path))))
    run = api.decode_and_score("demo-model", "layer_0", [0.1, 0.4, -0.2, 0.8])
    assert run.scorecard.intervention_method == "stub-50pct"


def test_decode_and_score_explicit_none_layer_path_uses_stub(tmp_path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path))))
    run = api.decode_and_score("demo-model", "layer_0", [0.1, 0.4], layer_path=None, inputs=None)
    assert run.scorecard.intervention_method == "stub-50pct"


def test_decode_and_score_layer_path_without_inputs_uses_stub(tmp_path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path))))
    run = api.decode_and_score("demo-model", "layer_0", [0.1, 0.4], layer_path="transformer.h.0", inputs=None)
    assert run.scorecard.intervention_method == "stub-50pct"


# ---------------------------------------------------------------------------
# Batch decode — decode_batch / decode_and_score_batch
# ---------------------------------------------------------------------------

def test_decode_batch_returns_correct_length(tmp_path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig
    from core.decoder import DecodeResult
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path))))
    matrix = [[0.1, 0.5], [0.9, 0.2], [0.3, 0.7]]
    results = api.decode_batch("demo-model", "layer_0", matrix)
    assert len(results) == 3
    assert all(isinstance(r, DecodeResult) for r in results)


def test_decode_and_score_batch_returns_correct_length(tmp_path) -> None:
    from api import HypoSpaceAPI, HypoSpaceResult
    from core.config import DecoderConfig, RuntimeConfig
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path))))
    matrix = [[0.1, 0.5], [0.9, 0.2]]
    results = api.decode_and_score_batch("demo-model", "layer_0", matrix)
    assert len(results) == 2
    assert all(isinstance(r, HypoSpaceResult) for r in results)


def test_decode_batch_empty_matrix_returns_empty(tmp_path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path))))
    assert api.decode_batch("demo-model", "layer_0", []) == []


# ---------------------------------------------------------------------------
# Export — GovernanceScorecard.to_dict() / HypoSpaceResult.to_dict()
# ---------------------------------------------------------------------------

def test_governance_scorecard_to_dict_expected_keys() -> None:
    from interpretability.faithfulness import GovernanceScorecard
    sc = GovernanceScorecard(
        faithfulness_score=0.75,
        stability_score=0.80,
        risk_flag="ok",
        passes_thresholds=True,
        intervention_method="stub-50pct",
    )
    d = sc.to_dict()
    assert set(d.keys()) == {"faithfulness_score", "stability_score", "risk_flag", "passes_thresholds", "intervention_method"}
    assert isinstance(d["faithfulness_score"], float)
    assert isinstance(d["passes_thresholds"], bool)


def test_hypospace_result_to_dict_json_round_trip(tmp_path) -> None:
    import json
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path))))
    run = api.decode_and_score("demo-model", "layer_0", [0.1, 0.4, -0.2, 0.8])
    d = run.to_dict()
    serialized = json.dumps(d)
    loaded = json.loads(serialized)
    assert isinstance(loaded["scorecard"]["risk_flag"], str)
    assert "features" in loaded


# ---------------------------------------------------------------------------
# Streamlit helpers — _parse_activations, _feature_rows
# ---------------------------------------------------------------------------

def test_parse_activations_valid_csv() -> None:
    from viz.streamlit_app import _parse_activations
    assert _parse_activations("0.1, 0.4, -0.2") == [0.1, 0.4, -0.2]


def test_parse_activations_empty_string_returns_empty() -> None:
    from viz.streamlit_app import _parse_activations
    assert _parse_activations("") == []


def test_parse_activations_non_numeric_raises() -> None:
    from viz.streamlit_app import _parse_activations
    with pytest.raises(ValueError):
        _parse_activations("0.1,abc")


def test_feature_rows_expected_keys() -> None:
    from core.hierarchy import Feature
    from viz.streamlit_app import _feature_rows
    feat = Feature(id="l:feature:0", layer="l", score=0.5, source_index=0, label="hi")
    rows = _feature_rows([feat])
    assert len(rows) == 1
    assert set(rows[0].keys()) == {"id", "layer", "source_index", "score", "label"}

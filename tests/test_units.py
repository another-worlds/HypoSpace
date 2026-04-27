from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.hierarchy import Feature, HierarchyEngine
from core.kernel_library import KernelLibrary
from data.extractor import ActivationExtractor
from data.preprocessor import ActivationPreprocessor
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

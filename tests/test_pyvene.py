from __future__ import annotations

import importlib

import pytest

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_PYVENE_AVAILABLE = importlib.util.find_spec("pyvene") is not None
_NNSIGHT_AVAILABLE = importlib.util.find_spec("nnsight") is not None

pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE or not _NNSIGHT_AVAILABLE,
    reason="torch and nnsight are required for PyVeneInterventionRunner tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(n: int = 3):
    from core.hierarchy import Feature

    return [
        Feature(
            id=f"feature_{i}",
            layer="layer_0",
            score=0.1 * (i + 1),
            source_index=i,
            label=None,
        )
        for i in range(n)
    ]


def _make_runner(tmp_path, model_name: str = "gpt2", layer_path: str = "transformer.h.0"):
    from data.nnsight_extractor import NNSightExtractor
    from data.pyvene_runner import PyVeneInterventionRunner

    ex = NNSightExtractor(model_name, device="cpu", cache_dir=str(tmp_path))
    ex._load_model()
    return PyVeneInterventionRunner(lm=ex._lm, layer_path=layer_path, device="cpu")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_require_raises_without_torch(monkeypatch) -> None:
    """require() raises ImportError when torch probe reports False."""
    import data.pyvene_runner as mod
    from data.pyvene_runner import PyVeneInterventionRunner

    monkeypatch.setattr(mod, "_torch_available", lambda: False)
    with pytest.raises(ImportError, match="pip install torch"):
        PyVeneInterventionRunner.require()


def test_runner_produces_one_result_per_feature(tmp_path) -> None:
    """run_interventions returns exactly one InterventionResult per feature."""
    runner = _make_runner(tmp_path)
    features = _make_features(3)
    results = runner.run_interventions(features, "The quick brown fox")
    assert len(results) == 3
    from interpretability.mechanistic import InterventionResult
    for r in results:
        assert isinstance(r, InterventionResult)


def test_effect_size_nonnegative(tmp_path) -> None:
    """All returned effect_size values are >= 0."""
    runner = _make_runner(tmp_path)
    features = _make_features(4)
    results = runner.run_interventions(features, "Hello world")
    for r in results:
        assert r.effect_size >= 0.0


def test_feature_ids_preserved(tmp_path) -> None:
    """feature_id in each result matches the originating Feature.id."""
    runner = _make_runner(tmp_path)
    features = _make_features(3)
    results = runner.run_interventions(features, "Test input")
    result_ids = {r.feature_id for r in results}
    feature_ids = {f.id for f in features}
    assert result_ids == feature_ids


def test_ablated_field_is_zero(tmp_path) -> None:
    """ablated field is 0.0 — the dimension was zero-ablated."""
    runner = _make_runner(tmp_path)
    features = _make_features(2)
    results = runner.run_interventions(features, "Test input")
    for r in results:
        assert r.ablated == 0.0


def test_hook_fallback_when_pyvene_absent(tmp_path, monkeypatch) -> None:
    """Hook path is used when pyvene is unavailable; results still have real effect sizes."""
    import data.pyvene_runner as mod
    monkeypatch.setattr(mod, "pyvene_available", lambda: False)

    runner = _make_runner(tmp_path)
    features = _make_features(3)
    results = runner.run_interventions(features, "The quick brown fox")
    assert len(results) == 3
    for r in results:
        assert r.effect_size >= 0.0


def test_results_differ_from_50pct_stub(tmp_path) -> None:
    """Real effect sizes are not uniformly score * 0.5 (distinguishes from stub)."""
    runner = _make_runner(tmp_path)
    features = _make_features(4)
    results = runner.run_interventions(features, "The quick brown fox")

    stub_effect_sizes = [f.score * 0.5 for f in features]
    real_effect_sizes = [r.effect_size for r in results]

    # At least one real effect size differs from the stub value
    assert any(
        abs(real - stub) > 1e-6
        for real, stub in zip(real_effect_sizes, stub_effect_sizes)
    ), "Expected real interventions to produce effect sizes different from the 50% stub"


@pytest.mark.skipif(not _PYVENE_AVAILABLE, reason="pyvene required")
def test_pyvene_path_is_exercised(tmp_path) -> None:
    """pyvene code path is entered when pyvene is installed."""
    import data.pyvene_runner as mod

    calls: list[str] = []
    original = mod.PyVeneInterventionRunner._ablate_with_pyvene

    def patched(self, inputs, source_index):
        calls.append("pyvene")
        return original(self, inputs, source_index)

    mod.PyVeneInterventionRunner._ablate_with_pyvene = patched
    try:
        runner = _make_runner(tmp_path)
        runner.run_interventions(_make_features(2), "Hello")
    finally:
        mod.PyVeneInterventionRunner._ablate_with_pyvene = original

    assert "pyvene" in calls

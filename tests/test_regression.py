from __future__ import annotations

import json
from pathlib import Path

import pytest

from api import HypoSpaceAPI
from core.config import DecoderConfig, RuntimeConfig


@pytest.mark.parametrize("case", json.loads(Path("tests/fixtures/mini_regression_set.json").read_text(encoding="utf-8")))
def test_fixed_mini_set_regression(case: dict, tmp_path: Path) -> None:
    """Regression guard for MVP fixed mini-set outputs."""
    api = HypoSpaceAPI(
        config=DecoderConfig(
            runtime=RuntimeConfig(cache_dir=str(tmp_path)),
            top_k=int(case["top_k"]),
        )
    )

    run = api.decode_and_score(
        model_name=case["model"],
        layer=case["layer"],
        raw_activations=case["activations"],
        version="0.1.0",
    )

    features = run.decode.features
    assert [feature.source_index for feature in features] == case["expected_indices"]
    assert [feature.score for feature in features] == pytest.approx(case["expected_scores"], abs=1e-6)
    assert [feature.label for feature in features] == case["expected_labels"]

    assert run.scorecard.faithfulness_score == pytest.approx(case["expected_faithfulness"], abs=1e-6)
    assert run.scorecard.stability_score == pytest.approx(case["expected_stability"], abs=1e-6)
    assert run.scorecard.risk_flag == case["expected_risk_flag"]


def test_mechanistic_coverage_for_top_features(tmp_path: Path) -> None:
    """KPI guard: top features should all have mechanistic checks (>=80%)."""
    api = HypoSpaceAPI(config=DecoderConfig(runtime=RuntimeConfig(cache_dir=str(tmp_path)), top_k=5))
    decoded = api.decode(model_name="demo", layer="layer_kpi", raw_activations=[0.9, 0.4, -0.7, 0.1, -0.3])

    interventions = api.mechanistic.run_interventions(decoded.features)
    feature_ids = {feature.id for feature in decoded.features}
    intervention_ids = {row.feature_id for row in interventions}

    coverage = len(feature_ids & intervention_ids) / max(len(feature_ids), 1)
    assert coverage >= 0.8

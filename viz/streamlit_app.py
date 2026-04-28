from __future__ import annotations

"""Streamlit UI: three-tab interactive explorer for kernels, canvas, and governance."""

import csv
import io
import json
from typing import List

from api import HypoSpaceAPI
from core.config import DecoderConfig, GovernanceConfig, RuntimeConfig
from core.hierarchy import Feature
from interpretability.mechanistic import InterventionResult
from viz.canvas import SemanticCanvas


def _parse_activations(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _feature_rows(features: List[Feature]) -> List[dict[str, float | int | str | None]]:
    return [
        {
            "id": feature.id,
            "layer": feature.layer,
            "source_index": feature.source_index,
            "score": round(feature.score, 4),
            "label": feature.label,
        }
        for feature in features
    ]


def _intervention_rows(rows: List[InterventionResult]) -> List[dict[str, float | str]]:
    return [
        {
            "feature_id": row.feature_id,
            "baseline": round(row.baseline, 4),
            "ablated": round(row.ablated, 4),
            "effect_size": round(row.effect_size, 4),
        }
        for row in rows
    ]


def _to_csv(rows: List[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def run() -> None:
    import streamlit as st

    st.set_page_config(page_title="HypoSpace MVP", layout="wide")
    st.title("HypoSpace MVP Explorer")

    with st.sidebar:
        st.header("Run configuration")
        model_name = st.text_input("Model", value="demo-model")
        layer = st.text_input("Layer", value="layer_0")
        raw_activations = st.text_area("Activations", value="0.1,0.4,-0.2,0.8")
        top_k = st.slider("Top-K features", min_value=1, max_value=16, value=8)
        min_faithfulness = st.slider("Min faithfulness", min_value=0.0, max_value=1.0, value=0.65)
        min_stability = st.slider("Min stability", min_value=0.0, max_value=1.0, value=0.60)
        run_button = st.button("Decode")

    if not run_button:
        st.info("Configure inputs in the sidebar and click **Decode**.")
        return

    config = DecoderConfig(
        top_k=top_k,
        runtime=RuntimeConfig(device="cpu"),
        governance=GovernanceConfig(
            min_faithfulness_score=min_faithfulness,
            min_stability_score=min_stability,
        ),
    )
    api = HypoSpaceAPI(config=config)

    try:
        values = _parse_activations(raw_activations)
        decode_result = api.decode(model_name=model_name, layer=layer, raw_activations=values)
        scorecard = api.scorecard(decode_result)
        interventions = api.mechanistic.run_interventions(decode_result.features)
    except Exception as e:
        st.error(str(e))
        return

    kernel_tab, semantic_tab, governance_tab = st.tabs(
        ["Kernel Explorer", "Semantic Canvas", "Mechanistic Probes + Governance"]
    )

    with kernel_tab:
        st.subheader("Decoded kernel features")
        st.caption(f"Kernel artifact: {decode_result.kernel_path}")
        feature_rows = _feature_rows(decode_result.features)
        st.dataframe(feature_rows, use_container_width=True)
        st.download_button(
            label="Download features CSV",
            data=_to_csv(feature_rows),
            file_name="features.csv",
            mime="text/csv",
        )

    with semantic_tab:
        st.subheader("Concept graph preview")
        canvas = SemanticCanvas()
        layout = canvas.to_layout(decode_result.features)
        edges = canvas.to_edges(decode_result.features)

        if layout:
            st.scatter_chart(layout, x="x", y="y", color="score", size="score")
            st.download_button(
                label="Download canvas CSV",
                data=_to_csv(layout),
                file_name="canvas.csv",
                mime="text/csv",
            )
            st.markdown("**Nearest-neighbor links**")
            st.dataframe(
                [
                    {"source": source, "target": target, "strength": round(strength, 4)}
                    for source, target, strength in edges
                ],
                use_container_width=True,
            )
        else:
            st.warning("No features extracted for current input.")

    with governance_tab:
        st.subheader("Mechanistic checks")
        intervention_rows = _intervention_rows(interventions)
        st.dataframe(intervention_rows, use_container_width=True)
        st.markdown("**Governance scorecard**")
        if scorecard.intervention_method == "stub-50pct":
            st.caption(
                "⚠ Interventions are synthetic (effect_size = baseline × 0.5). "
                "Use decode_and_score_from_model() with torch for real causal measurements."
            )
        scorecard_dict = {
            "faithfulness_score": round(scorecard.faithfulness_score, 4),
            "stability_score": round(scorecard.stability_score, 4),
            "risk_flag": scorecard.risk_flag,
            "passes_thresholds": scorecard.passes_thresholds,
            "intervention_method": scorecard.intervention_method,
        }
        st.json(scorecard_dict)
        st.download_button(
            label="Download scorecard JSON",
            data=json.dumps(scorecard_dict, indent=2),
            file_name="scorecard.json",
            mime="application/json",
        )


if __name__ == "__main__":
    run()

# HypoSpace

HypoSpace is a lightweight model-interpretability project that turns raw layer activations into structured, inspectable concepts with governance signals. The current MVP focuses on a practical end-to-end flow:

`model -> activations -> hierarchical kernels -> semantic labels -> mechanistic checks -> governance scorecard -> UI`

## Intro

Modern model debugging often breaks into disconnected workflows: feature extraction in one script, interpretation in another notebook, and trust checks in separate experiments. HypoSpace unifies that workflow behind a small API, CLI, and Streamlit app so teams can iterate on interpretability faster.

At a high level, HypoSpace:
- extracts and normalizes activations,
- decodes them into reusable kernel-like features,
- annotates features with semantic labels,
- runs mechanistic intervention checks,
- produces a governance scorecard for faithfulness and stability.

## Getting Started

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip pytest streamlit
```

### 2) Run the CLI demo

```bash
python main.py --model demo-model --layer layer_0 --activations 0.1,0.4,-0.2,0.8 --top-k 4
```

The command writes kernel artifacts into `.hypo_cache/` by default.

### 3) Launch the Streamlit UI

```bash
streamlit run viz/streamlit_app.py
```

MVP tabs:
- Kernel Explorer
- Semantic Canvas
- Mechanistic Probes + Governance

### 4) Run tests

```bash
pytest -q
```

## Vision

HypoSpace aims to become a persistent, cross-run "concept operating system" for neural model internals:
- **Reusable concepts** across runs and model iterations via versioned kernel templates.
- **Higher trust** through intervention-driven faithfulness checks, not labels alone.
- **Operational readiness** with governance scorecards and risk flags for low-confidence interpretations.
- **Accessible workflows** for both researchers (API/CLI) and practitioners (interactive UI).

In short: faster time-to-first-insight, stronger interpretability evidence, and safer deployment decisions.

## Architecture

The current project is organized as follows:

- `api.py` — top-level `HypoSpaceAPI` façade that orchestrates decode + score workflows.
- `core/`
  - `config.py` — runtime/decoder/governance configuration models.
  - `decoder.py` — decode pipeline entrypoint.
  - `hierarchy.py` — hierarchical kernel logic.
  - `kernel_library.py` — kernel templates, save/load, and matching utilities.
- `data/`
  - `extractor.py` — activation extraction and cache integration.
  - `preprocessor.py` — normalization and preprocessing transforms.
- `interpretability/`
  - `semantic.py` — semantic annotations for decoded features.
  - `mechanistic.py` — intervention and mechanistic probe routines.
  - `faithfulness.py` — governance scorecard and threshold checks.
- `viz/`
  - `streamlit_app.py` — interactive MVP UI.
  - `canvas.py` — visualization helpers for concept relationships.
- `main.py` — CLI entrypoint for decode-and-score execution.
- `tests/` — smoke, contract, and regression coverage for MVP behavior.

### Request/Processing Flow

1. `HypoSpaceAPI.decode(...)` extracts + normalizes activations.
2. `RealityDecoder.decode(...)` generates top-k feature concepts and kernel artifacts.
3. `SemanticInterpreter` attaches human-readable labels.
4. `MechanisticAnalyzer` runs interventions on selected features.
5. `FaithfulnessChecker` computes governance metrics and threshold outcomes.

This keeps the pipeline modular while preserving a single-call happy path via `decode_and_score(...)`.

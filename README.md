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

**Minimal** — stdlib pipeline, Streamlit UI, no GPU required:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip pytest streamlit
```

**Full** — adds live HuggingFace extraction, real zero-ablation interventions, and faster cache:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install nnsight pyvene diskcache
# or via extras: pip install -e ".[nnsight,pyvene,cache,dev]"
```

| Feature | Minimal | Full |
|---|---|---|
| CLI decode + governance scorecard | ✅ | ✅ |
| Streamlit UI | ✅ | ✅ |
| Kernel library, versioning, caching | ✅ JSON cache | ✅ diskcache |
| Mechanistic interventions | stub (synthetic 50%) | real zero-ablation |
| Live HuggingFace model extraction | ❌ | ✅ |
| Full test suite | 50 pass, 19 skipped | 69 pass, 0 skipped |

### 2) Check subsystem health

```bash
python main.py --diagnostics
```

Prints a JSON report showing the status of every subsystem and which optional dependencies are available. On a minimal install expect `"overall_status": "degraded"` — the mechanistic stub and absent diskcache both warn. After a full install only the mechanistic probe remains degraded (by design — real causal measurements require a trained SAE backend). Run this after any environment change.

### 3) Run the CLI demo

```bash
python main.py --model demo-model --layer layer_0 --activations 0.1,0.4,-0.2,0.8 --top-k 4
```

The command writes kernel artifacts into `.hypo_cache/` by default.

### 4) Live model extraction (optional — requires nnsight + torch)

```bash
pip install nnsight torch
```

```python
from api import HypoSpaceAPI
from core.config import DecoderConfig, RuntimeConfig

api = HypoSpaceAPI(config=DecoderConfig(top_k=8, runtime=RuntimeConfig(device="cpu")))

result = api.decode_and_score_from_model(
    model_name="gpt2",
    layer="layer_0",
    layer_path="transformer.h.0",  # dot-notation path into the model
    inputs="The quick brown fox",
    token_index=-1,                # which token position to extract (-1 = last)
    version="0.1.0",
)
# result.decode  — top-k features
# result.scorecard  — faithfulness / stability / risk flags
```

Activations are cached under `.hypo_cache/nnsight/`; repeated calls with identical arguments skip the forward pass entirely.

### 5) Launch the Streamlit UI

```bash
streamlit run viz/streamlit_app.py
```

MVP tabs:
- Kernel Explorer
- Semantic Canvas
- Mechanistic Probes + Governance

### 6) Run tests

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
- `diagnostics.py` — `run_diagnostics()` entry point; probes each subsystem independently and returns a `DiagnosticsReport`. Also accessible as `HypoSpaceAPI.diagnostics()` or `python main.py --diagnostics`.
- `core/`
  - `config.py` — runtime/decoder/governance configuration models.
  - `decoder.py` — decode pipeline entrypoint.
  - `hierarchy.py` — hierarchical kernel logic.
  - `kernel_library.py` — kernel templates, save/load, and matching utilities.
- `data/`
  - `extractor.py` — activation extraction and SHA256-keyed disk cache.
  - `nnsight_extractor.py` — live extraction from HuggingFace models via nnsight tracing.
  - `pyvene_runner.py` — zero-ablation interventions via pyvene/hooks.
  - `preprocessor.py` — normalization and preprocessing transforms.
  - `utils.py` — UTC timestamp helper.
- `interpretability/`
  - `semantic.py` — semantic annotations for decoded features.
  - `mechanistic.py` — intervention and mechanistic probe routines.
  - `faithfulness.py` — governance scorecard and threshold checks.
- `viz/`
  - `streamlit_app.py` — interactive MVP UI.
  - `canvas.py` — visualization helpers for concept relationships.
- `main.py` — CLI entrypoint for decode-and-score execution.
- `tests/` — smoke, contract, regression, diagnostics, nnsight, and pyvene coverage (nnsight/pyvene modules skipped when optional deps absent).

### Request/Processing Flow

1. `HypoSpaceAPI.decode(...)` extracts + normalizes activations.
2. `RealityDecoder.decode(...)` generates top-k feature concepts and kernel artifacts.
3. `SemanticInterpreter` attaches human-readable labels.
4. `MechanisticAnalyzer` runs interventions on selected features.
5. `FaithfulnessChecker` computes governance metrics and threshold outcomes.

This keeps the pipeline modular while preserving a single-call happy path via `decode_and_score(...)`.

### Live Extraction Path (nnsight)

When called via `decode_from_model()` or `decode_and_score_from_model()`, the pipeline adds a live extraction step before the decode:

1. `NNSightExtractor.extract()` runs a forward pass through the HuggingFace model and captures the target layer's activations.
2. Activations are written to the disk cache and then follow the same Path A steps above.

The model is loaded once per `HypoSpaceAPI` instance and reused as long as `model_name` does not change.

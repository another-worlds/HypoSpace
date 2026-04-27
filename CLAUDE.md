# HypoSpace — AI Assistant Guide

## Project Overview

HypoSpace is a Python 3.11+ model interpretability toolkit. It takes raw neural network activations and produces a structured "reality decode": top-k concept features ranked by magnitude, semantic auto-labels, mechanistic intervention checks, and a governance scorecard with configurable faithfulness/stability thresholds.

Current stage: working E2E skeleton with CLI, Streamlit UI, full test suite, **nnsight live extraction** via `HypoSpaceAPI.decode_from_model()`, and **pyvene zero-ablation interventions** via `PyVeneInterventionRunner`.

---

## Architecture

```
HypoSpace/
├── api.py                        # HypoSpaceAPI — public one-call façade
├── main.py                       # CLI entrypoint (argparse, JSON output)
├── diagnostics.py                # run_diagnostics() — per-subsystem health probes, DiagnosticsReport
├── core/
│   ├── config.py                # DecoderConfig, RuntimeConfig, GovernanceConfig
│   ├── decoder.py               # RealityDecoder — orchestrates the decode pipeline
│   ├── hierarchy.py             # HierarchyEngine + Feature dataclass + FeatureBackend Protocol — dispatches to SAE or magnitude fallback
│   └── kernel_library.py        # KernelLibrary — semver-versioned JSON persistence
├── data/
│   ├── extractor.py             # ActivationExtractor — disk cache (SHA256 keys)
│   ├── nnsight_extractor.py     # NNSightExtractor — live extraction via nnsight model tracing
│   ├── preprocessor.py          # ActivationPreprocessor — max-abs normalization
│   ├── pyvene_runner.py         # PyVeneInterventionRunner — real zero-ablation via pyvene/hooks
│   ├── sae_backend.py           # MagnitudeBackend, MatryoshkaBackend, build_backend() — SAE inference (torch-optional)
│   └── utils.py                 # utc_timestamp() helper, resolve_layer() dot-notation traversal
├── interpretability/
│   ├── semantic.py              # SemanticInterpreter — intensity-band auto-labels
│   ├── mechanistic.py           # MechanisticAnalyzer — synthetic 50% ablation stub (CPU fallback; real zero-ablation in PyVeneInterventionRunner)
│   └── faithfulness.py          # FaithfulnessChecker, GovernanceScorecard, GovernanceThresholdError
├── viz/
│   ├── streamlit_app.py         # Streamlit 3-tab interactive UI
│   └── canvas.py                # SemanticCanvas — feature points and nearest-neighbor edges
└── tests/
    ├── test_smoke.py            # E2E integration tests (6 tests)
    ├── test_contracts.py        # JSON payload structure contracts (2 tests)
    ├── test_regression.py       # Fixed mini-set regression + KPI guard (10 tests: 9 fixture cases + 1 KPI guard)
    ├── test_nnsight.py          # nnsight live extraction tests (11 tests; skipped if nnsight/torch absent)
    ├── test_pyvene.py           # PyVeneInterventionRunner tests (8 tests; skipped if torch absent)
    ├── test_sae_backend.py      # SAE backend tests (10 tests; skipped if torch absent)
    ├── test_diagnostics.py      # diagnostics module tests (37 tests)
    ├── test_canvas.py           # SemanticCanvas edge-case tests (7 tests)
    ├── test_units.py            # negative-path and boundary-value unit tests (42 tests)
    └── fixtures/
        └── mini_regression_set.json
```

### Processing Pipeline

**Path A — raw activations (`decode()` / `decode_and_score()`):**
```
raw_activations
  → ActivationExtractor    (disk cache, SHA256 key)
  → ActivationPreprocessor (max-abs normalization)
  → RealityDecoder → HierarchyEngine  (SAE via MatryoshkaBackend if sae_path set, else magnitude top-k)
  → KernelLibrary          (save artifact, compute cross-run match rate)
  → SemanticInterpreter    (intensity-band labels)
  → DecodeResult                         ← decode() stops here

  # decode_and_score() continues:
  → MechanisticAnalyzer   (synthetic 50% stub) or PyVeneInterventionRunner (real zero-ablation, when torch available)
  → FaithfulnessChecker   (faithfulness + stability scores)
  → HypoSpaceResult
```

**Path B — live model extraction via nnsight (`decode_from_model()` / `decode_and_score_from_model()`):**
```
model_name + inputs + layer_path
  → NNSightExtractor.extract()  (nnsight forward-pass trace; own input-keyed cache)
  → ActivationPreprocessor (max-abs normalization)
  → RealityDecoder → …          (same as Path A from here)
  → DecodeResult                         ← decode_from_model() stops here

  # decode_and_score_from_model() continues as in Path A
```

---

## Development Commands

```bash
# Run the CLI (prints JSON to stdout)
python main.py --model demo-model --layer layer_0 --activations 0.1,0.4,-0.2,0.8

# All CLI flags (raw-activation path)
python main.py \
  --model demo-model \
  --layer layer_0 \
  --activations 0.1,0.4,-0.2,0.8 \
  --top-k 8 \
  --device cpu \
  --min-faithfulness 0.65 \
  --min-stability 0.60 \
  --high-intensity-threshold 0.8 \
  --medium-intensity-threshold 0.4 \
  --version 0.1.0
  # --fail-on-low-confidence   (exit code 2 if thresholds not met)

# Live-model extraction path (requires nnsight + torch)
python main.py \
  --model gpt2 \
  --layer layer_0 \
  --layer-path transformer.h.0 \
  --inputs "The quick brown fox" \
  --token-index -1 \
  --version 0.1.0

# Run deep subsystem diagnostics (no other flags needed)
python main.py --diagnostics

# Launch the Streamlit UI (install ui extras first if not already installed)
# pip install -e ".[ui]"   # or: pip install streamlit
streamlit run viz/streamlit_app.py

# Run all tests (minimal install: 94 pass, 19 skipped; full install: 113 pass, 0 skipped)
python -m pytest -q   # reliable in all environments
# or: pytest -q       # works when pytest is installed in the active venv

# Install full optional stack (CPU torch — matches CI)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install nnsight pyvene diskcache
# or: pip install -e ".[nnsight,pyvene,cache,dev]"

# Run a specific test module
pytest tests/test_smoke.py -v
pytest tests/test_contracts.py -v
pytest tests/test_regression.py -v
pytest tests/test_diagnostics.py -v

# Run nnsight live-extraction tests (requires nnsight + torch to be installed)
pytest tests/test_nnsight.py -v

# Run pyvene intervention tests (requires torch to be installed)
pytest tests/test_pyvene.py -v
```

No build step required. No environment variables needed — all configuration is Python dataclasses.

### nnsight live extraction

`HypoSpaceAPI.decode_from_model()` extracts activations directly from a HuggingFace model:

```python
from api import HypoSpaceAPI
from core.config import DecoderConfig, RuntimeConfig

api = HypoSpaceAPI(config=DecoderConfig(top_k=8, runtime=RuntimeConfig(device="cpu")))

# One-call path (decode + governance scorecard):
result = api.decode_and_score_from_model(
    model_name="gpt2",             # any HuggingFace model id
    layer="layer_0",               # HypoSpace artifact name
    layer_path="transformer.h.0",  # nnsight attribute path into the model
    inputs="The quick brown fox",  # text or token ids
    token_index=-1,                # which token position to extract (-1 = last)
    version="0.1.0",
)
# result is a HypoSpaceResult with .decode (DecodeResult) and .scorecard

# Decode only (no governance):
decode_result = api.decode_from_model(
    model_name="gpt2",
    layer="layer_0",
    layer_path="transformer.h.0",
    inputs="The quick brown fox",
)
scorecard = api.scorecard(decode_result)
```

The model is loaded once per `HypoSpaceAPI` instance and reused across calls as long as
`model_name` does not change. Activations are cached by input key
`(model_name, layer_path, inputs, token_index)` under `.hypo_cache/nnsight/`; repeated
calls with identical arguments skip the forward pass entirely.

To extract multiple layers in a single forward pass use `NNSightExtractor` directly:

```python
from data.nnsight_extractor import NNSightExtractor

ex = NNSightExtractor("gpt2", device="cpu", cache_dir=".hypo_cache")
layers = ex.extract_layers(
    "The quick brown fox",
    layer_paths=["transformer.h.0", "transformer.h.6", "transformer.h.11"],
)
# layers is Dict[str, List[float]] — one entry per layer_path
```

`layer_path` uses dot-notation with integer segments for list indexing:
- GPT-2: `"transformer.h.0"` (first transformer block)
- LLaMA-style: `"model.layers.0"`
- BERT-style: `"bert.encoder.layer.0"`

The extracted tensor must have shape `(batch, seq_len, hidden_dim)` or `(seq_len, hidden_dim)`. Tuple outputs (e.g. `(hidden_states, past_key_values, ...)`) are handled automatically — the first element is used.

---

## Configuration

Configuration flows from `core/config.py` dataclasses. All fields have sensible defaults; pass instances to override.

```python
from core.config import DecoderConfig, RuntimeConfig, GovernanceConfig

config = DecoderConfig(
    backend="matryoshka",           # "matryoshka" | "topk" | "jumprelu" | "magnitude"
    top_k=8,                        # Number of top features to extract
    sae_path=None,                  # Path to SAE checkpoint dir/file; None → magnitude fallback
    high_intensity_threshold=0.8,   # Score ≥ this → "high-intensity" label
    medium_intensity_threshold=0.4, # Score ≥ this → "medium-intensity" label
    runtime=RuntimeConfig(
        device="cpu",         # "cpu" or "cuda"
        max_features=2048,
        cache_dir=".hypo_cache",  # Disk cache root
    ),
    governance=GovernanceConfig(
        min_faithfulness_score=0.65,   # Mean effect size threshold
        min_stability_score=0.60,      # Coefficient-of-variation-based stability
        fail_on_low_confidence=False,  # Raise GovernanceThresholdError if True
    ),
)
api = HypoSpaceAPI(config=config)
```

The CLI mirrors these fields as flags (`--top-k`, `--min-faithfulness`, `--high-intensity-threshold`, etc.).

---

## Key Data Structures

All data structures are `@dataclass(slots=True)` — do **not** add `__dict__`-based attributes.

| Dataclass | Location | Purpose |
|---|---|---|
| `Feature` | `core/hierarchy.py` | Single extracted concept: `id`, `layer`, `score`, `source_index`, `label` (id uses `:sae:` infix when SAE backend active) |
| `FeatureBackend` | `core/hierarchy.py` | `@runtime_checkable` Protocol — `extract(activations, layer, top_k) -> list[Feature]`; satisfied by `MagnitudeBackend` and `MatryoshkaBackend` |
| `SAEEncodeResult` | `data/sae_backend.py` | Raw sparse SAE output: `feature_indices`, `feature_scores` — intermediate before `Feature` construction |
| `DecodeResult` | `core/decoder.py` | Output of decode step: `model_name`, `layer`, `features[]`, `kernel_path`, `metadata{}` |
| `KernelTemplate` | `core/kernel_library.py` | Persisted artifact with semver `version` and feature list |
| `InterventionResult` | `interpretability/mechanistic.py` | `feature_id`, `baseline`, `ablated`, `effect_size` |
| `GovernanceScorecard` | `interpretability/faithfulness.py` | `faithfulness_score`, `stability_score`, `risk_flag`, `passes_thresholds`, `intervention_method` |
| `HypoSpaceResult` | `api.py` | Top-level result: `.decode` + `.scorecard` |
| `CheckDetail` | `diagnostics.py` | One assertion within a probe: `name`, `passed`, `detail` |
| `ProbeResult` | `diagnostics.py` | Outcome of a single subsystem probe: `subsystem`, `status`, `latency_ms`, `checks[]`, `warnings[]`, `errors[]` |
| `DependencyStatus` | `diagnostics.py` | Optional-dependency availability: `name`, `available`, `note` |
| `DiagnosticsReport` | `diagnostics.py` | Full health report: `schema_version`, `generated_at_utc`, `overall_status`, `dependencies[]`, `probes[]` |

---

## Coding Conventions

- `from __future__ import annotations` at the top of every file
- `@dataclass(slots=True)` for all data-holding classes — **never** add instance attributes outside `__init__` on slotted dataclasses
- Full type hints throughout; use `pathlib.Path` for all file paths
- One primary class per file; related dataclasses may share a file (see `config.py`)
- `snake_case` for functions/variables/modules, `PascalCase` for classes, `UPPER_CASE` for module-level constants
- Private helpers prefixed with `_` (e.g., `_intensity_band`, `_cache_key`)
- Stdlib only in all modules except `viz/` (Streamlit), `data/nnsight_extractor.py` (nnsight + torch), `data/pyvene_runner.py` (pyvene + torch), `data/sae_backend.py` (torch — optional, with magnitude fallback), and `tests/` (pytest)
- JSON with UTF-8 encoding for all persisted artifacts
- Timestamps always via `data.utils.utc_timestamp()` (UTC ISO format)

---

## Persistence Layer

Artifacts are stored under `.hypo_cache/` (configurable via `RuntimeConfig.cache_dir`):

```
.hypo_cache/
├── manifest.json                    # Kernel registry: versions, file mappings, latest pointer
├── demo-model-layer_0-0.1.0.json    # KernelTemplate artifact ({model_name}-{layer}-{version}.json)
└── activations/
    └── <sha256_hex>.json            # Cached raw activations (keyed by content hash)
```

**Kernel versioning:** Filenames follow `{model_name}-{layer}-{version}.json`. Versions are semver-sorted; `KernelLibrary.load_latest()` returns the highest version. `KernelLibrary.match()` computes cross-run feature overlap by `source_index`. `KernelLibrary.merge()` combines two versioned kernels: for each `source_index` present in both, the higher-scoring feature is kept; indices present only in the candidate are added. The result is saved as a new versioned artifact.

---

## Testing

```bash
python -m pytest -q    # 103 stdlib-only tests always pass; 133 total when all optional deps installed
```

**Test modules:**
- `test_smoke.py` — Full E2E pipeline, semver loading, kernel match/merge, merge semantics, governance errors, SAE fallback (6 tests)
- `test_contracts.py` — Validates JSON payload keys/types for kernel artifacts and canvas output (2 tests)
- `test_regression.py` — Parametrized against `tests/fixtures/mini_regression_set.json`; KPI guard requires ≥80% mechanistic coverage of top features (10 tests)
- `test_nnsight.py` — Live extraction via `NNSightExtractor` and `decode_from_model()`; entire module is skipped when nnsight/torch are not installed (11 tests)
- `test_pyvene.py` — `PyVeneInterventionRunner` hook-fallback path and effect-size correctness; entire module is skipped when torch is not installed (8 tests)
- `test_sae_backend.py` — `MagnitudeBackend`, `MatryoshkaBackend`, `build_backend()` factory, API fallback path; entire module is skipped when torch is not installed (10 tests)
- `test_diagnostics.py` — 37 tests covering all 12 probes in `diagnostics.py`, JSON serializability, CLI flag, and `HypoSpaceAPI.diagnostics()` (37 tests)
- `test_canvas.py` — `SemanticCanvas` edge cases: empty input, single feature, sort order, edge values (7 tests)
- `test_units.py` — Negative-path and boundary-value tests for preprocessor, hierarchy, semantic, faithfulness, kernel_library, extractor, input validation, utils, decoder, FeatureBackend protocol (42 tests)

Tests use `tmp_path` fixtures for isolation. Never modify `tests/fixtures/mini_regression_set.json` without updating expected outputs — this is the regression baseline.

---

## Important Patterns and Constraints

**Do:**
- Add new SAE backends by implementing the `FeatureBackend` protocol in `data/sae_backend.py` and adding a branch in `build_backend()` — `api.py` auto-constructs the backend from `DecoderConfig.backend` + `DecoderConfig.sae_path`; `None` sae_path silently falls back to magnitude top-k
- Use `HypoSpaceAPI` as the integration point for any new workflow; it composes all components
- Keep all inter-module data passing via the defined dataclasses (no raw dicts between layers)
- Store new persisted data as JSON in `.hypo_cache/` following the existing manifest pattern

**Don't:**
- Add external dependencies to `core/` or `interpretability/` — these must remain stdlib-only
- Add external dependencies to `data/` except in `data/nnsight_extractor.py` (nnsight + torch), `data/pyvene_runner.py` (pyvene + torch), `data/sae_backend.py` (torch — optional, with magnitude fallback), and `data/extractor.py` (diskcache — optional, with JSON fallback), which are the designated optional-dependency modules
- Change `GovernanceConfig` defaults without updating `test_regression.py` fixture expectations
- Use `__dict__` or `setattr` on slotted dataclasses
- Create classes with inheritance hierarchies — the codebase uses composition
- Change the CLI's exit code contract: `0` = success, `2` = `GovernanceThresholdError`; `--diagnostics` also exits `0`
- Add probes to `diagnostics.py` that trigger network I/O or model downloads — probes must complete in < 100 ms on CPU with no external calls

---

## Roadmap Context

Completed post-MVP integrations:
- **nnsight** — `NNSightExtractor` in `data/nnsight_extractor.py`; wired into `HypoSpaceAPI.decode_from_model()`
- **pyvene** — `PyVeneInterventionRunner` in `data/pyvene_runner.py`; provides real zero-ablation interventions when torch is available; `MechanisticAnalyzer` remains as the CPU fallback stub
- **diskcache** — optional `diskcache.Cache` backend in `data/extractor.py` with JSON fallback when unavailable
- **CI/CD** — GitHub Actions configured in `.github/workflows/`; `test_nnsight.py` runs in the torch job
- **Cache key correctness** — `ActivationExtractor._cache_key()` now scopes keys to `(model_name, layer, values)` to prevent cross-model collisions
- **Live-model CLI** — `--layer-path`, `--inputs`, `--token-index` flags expose `decode_and_score_from_model()` from the command line
- **Configurable semantic thresholds** — `DecoderConfig.high_intensity_threshold` / `medium_intensity_threshold` control intensity-band labeling; settable via CLI and Python API
- **Input validation** — `decode()` validates non-empty model/layer, semver version, and `max_features` length; `KernelLibrary.load()` raises `ValueError` on corrupt JSON artifacts
- **SAE backend wiring** — `data/sae_backend.py` implements `FeatureBackend` protocol with `MagnitudeBackend` (stdlib) and `MatryoshkaBackend` (torch; matryoshka/topk/jumprelu variants); `DecoderConfig.sae_path` selects the checkpoint; `None` silently falls back to magnitude top-k; `api.py` is the sole composition root via `_build_sae_backend()`; diagnostics schema bumped to `1.1.0` with `_probe_sae_backend` as probe #12

KPIs to maintain: time-to-first-insight, concept consistency across runs (cross-run match rate), faithfulness coverage (≥80% of top features have mechanistic checks), CPU viability.

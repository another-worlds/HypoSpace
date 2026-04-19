# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -U pip pytest streamlit

# Run tests
pytest -q

# Run a single test
pytest tests/test_smoke.py::test_api_smoke_flow -q

# CLI demo (writes artifacts to .hypo_cache/)
python main.py --model demo-model --layer layer_0 --activations 0.1,0.4,-0.2,0.8 --top-k 4

# Launch Streamlit UI
streamlit run viz/streamlit_app.py
```

## Architecture

The pipeline is: `activations → extract → normalize → decode → semantic annotate → mechanistic check → governance scorecard`

**Entry points:**
- `api.py` — `HypoSpaceAPI` is the primary façade. `decode_and_score(model_name, layer, raw_activations)` is the single-call happy path; `decode()` and `scorecard()` are available separately.
- `main.py` — CLI wrapper around `HypoSpaceAPI`; exits with code `2` on `GovernanceThresholdError`.
- `viz/streamlit_app.py` — UI wrapper; calls `api.decode()` + `api.scorecard()` independently to populate three tabs.

**Configuration (`core/config.py`):**
`DecoderConfig` is the top-level config object passed everywhere. It composes `RuntimeConfig` (device, batch size, cache dir) and `GovernanceConfig` (faithfulness/stability thresholds, `fail_on_low_confidence`). Tests always pass `tmp_path` as `cache_dir` to keep runs isolated.

**Core pipeline (`core/`):**
- `HierarchyEngine` (`hierarchy.py`) — selects top-k features by absolute activation magnitude. Currently MVP-only; intended to delegate to Matryoshka SAE / USAE backends.
- `RealityDecoder` (`decoder.py`) — orchestrates `HierarchyEngine` and `KernelLibrary`. After extracting features it attempts a cross-run match against the persisted kernel and records a `cross_run_match_rate` in metadata before saving a new `KernelTemplate`.
- `KernelLibrary` (`kernel_library.py`) — JSON-based persistence under `cache_dir`. Maintains a `manifest.json` with semver-sorted version history per kernel ID; `load_latest()` reads that manifest. Kernel IDs are `"{model_name}-{layer}"`. `match()` compares features by `source_index` and score proximity; `merge()` resolves two versions by taking the highest-scoring feature per source index.

**Data (`data/`):**
- `ActivationExtractor` — SHA-256-keyed disk cache under `cache_dir/activations/`; cache key is derived from `(model_name, layer, values)`.
- `ActivationPreprocessor` — divides by max absolute value to normalize into `[-1, 1]`.

**Interpretability (`interpretability/`):**
- `SemanticInterpreter` — template-based labeling using intensity bands (`high/medium/low`). Designed to accept an optional LLM narrator later.
- `MechanisticAnalyzer` — placeholder ablation: sets `ablated = baseline * 0.5`, `effect_size = baseline - ablated`. Intended to be replaced by pyvene interventions.
- `FaithfulnessChecker` — computes `faithfulness_score` as mean `effect_size`, `stability_score` as `1 - normalized_std(baselines)`. Raises `GovernanceThresholdError` when `fail_on_low_confidence=True` and thresholds are not met.

**Visualization (`viz/`):**
- `SemanticCanvas` — data-only helper; `to_points()` returns score-sorted `(id, score)` pairs; `to_edges()` builds nearest-neighbor links between consecutive score-ranked features.

## Test layout

| File | Purpose |
|---|---|
| `tests/test_smoke.py` | End-to-end API flow, manifest/semver behavior, match/merge, governance error path |
| `tests/test_contracts.py` | JSON artifact schema and `SemanticCanvas` output shape |
| `tests/test_regression.py` | Parametrized golden-file regression on `tests/fixtures/mini_regression_set.json`; KPI coverage guard (≥80% of top features must have mechanistic checks) |

`pytest.ini` sets `pythonpath = .` so all imports resolve from the repo root without `src/` gymnastics.

## Key conventions

- All dataclasses use `slots=True` for performance.
- `from __future__ import annotations` is used in every module for forward-reference compatibility.
- Cache dirs must be unique per test — always pass `tmp_path` from pytest fixtures as `cache_dir`.
- Kernel artifact filenames follow `{kernel_id}-{version}.json`; version sorting is semver-numeric (not lexicographic) via `KernelLibrary._version_key`.
- `GovernanceThresholdError` is the only intentional exception surface; all other errors propagate naturally.

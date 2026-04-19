# HypoSpace — AI Assistant Guide

## Project Overview

HypoSpace is an **MVP-stage** Python 3.11+ model interpretability toolkit. It takes raw neural network activations and produces a structured "reality decode": top-k concept features ranked by magnitude, semantic auto-labels, mechanistic intervention checks, and a governance scorecard with configurable faithfulness/stability thresholds.

Current stage: working E2E skeleton with CLI, Streamlit UI, and a full test suite. No external ML dependencies yet — the heavy SAE backends (pyvene, nnsight) are planned for later stages.

---

## Architecture

```
HypoSpace/
├── api.py                        # HypoSpaceAPI — public one-call façade
├── main.py                       # CLI entrypoint (argparse, JSON output)
├── core/
│   ├── config.py                # DecoderConfig, RuntimeConfig, GovernanceConfig
│   ├── decoder.py               # RealityDecoder — orchestrates the decode pipeline
│   ├── hierarchy.py             # HierarchyEngine + Feature dataclass
│   └── kernel_library.py        # KernelLibrary — semver-versioned JSON persistence
├── data/
│   ├── extractor.py             # ActivationExtractor — disk cache (SHA256 keys)
│   ├── preprocessor.py          # ActivationPreprocessor — max-abs normalization
│   └── utils.py                 # utc_timestamp() helper
├── interpretability/
│   ├── semantic.py              # SemanticInterpreter — intensity-band auto-labels
│   ├── mechanistic.py           # MechanisticAnalyzer — 50% ablation interventions
│   └── faithfulness.py          # FaithfulnessChecker, GovernanceScorecard, GovernanceThresholdError
├── viz/
│   ├── streamlit_app.py         # Streamlit 3-tab interactive UI
│   └── canvas.py                # SemanticCanvas — feature points and nearest-neighbor edges
└── tests/
    ├── test_smoke.py            # E2E integration tests (4 tests)
    ├── test_contracts.py        # JSON payload structure contracts (2 tests)
    ├── test_regression.py       # Fixed mini-set regression + KPI guard (2 tests)
    └── fixtures/
        └── mini_regression_set.json
```

### Processing Pipeline

```
raw_activations
  → ActivationExtractor   (disk cache, SHA256 key)
  → ActivationPreprocessor (max-abs normalization)
  → RealityDecoder → HierarchyEngine  (top-k by magnitude)
  → KernelLibrary         (save artifact, compute cross-run match rate)
  → SemanticInterpreter   (intensity-band labels)
  → MechanisticAnalyzer   (50% ablation, effect size)
  → FaithfulnessChecker   (faithfulness + stability scores)
  → HypoSpaceResult
```

---

## Development Commands

```bash
# Run the CLI (prints JSON to stdout)
python main.py --model demo-model --layer layer_0 --activations 0.1,0.4,-0.2,0.8

# All CLI flags
python main.py \
  --model demo-model \
  --layer layer_0 \
  --activations 0.1,0.4,-0.2,0.8 \
  --top-k 8 \
  --device cpu \
  --min-faithfulness 0.65 \
  --min-stability 0.60 \
  --version 0.1.0
  # --fail-on-low-confidence   (exit code 2 if thresholds not met)

# Launch the Streamlit UI
streamlit run viz/streamlit_app.py

# Run all tests
pytest -q

# Run a specific test module
pytest tests/test_smoke.py -v
pytest tests/test_contracts.py -v
pytest tests/test_regression.py -v
```

No build step required. No environment variables needed — all configuration is Python dataclasses.

---

## Configuration

Configuration flows from `core/config.py` dataclasses. All fields have sensible defaults; pass instances to override.

```python
from core.config import DecoderConfig, RuntimeConfig, GovernanceConfig

config = DecoderConfig(
    backend="matryoshka",     # SAE backend (only "matryoshka" implemented in MVP)
    top_k=8,                  # Number of top features to extract
    runtime=RuntimeConfig(
        device="cpu",         # "cpu" or "cuda"
        batch_size=1,
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

The CLI mirrors these fields as flags (`--top-k`, `--min-faithfulness`, etc.).

---

## Key Data Structures

All data structures are `@dataclass(slots=True)` — do **not** add `__dict__`-based attributes.

| Dataclass | Location | Purpose |
|---|---|---|
| `Feature` | `core/hierarchy.py` | Single extracted concept: `id`, `layer`, `score`, `source_index`, `label` |
| `DecodeResult` | `core/decoder.py` | Output of decode step: `model_name`, `layer`, `features[]`, `kernel_path`, `metadata{}` |
| `KernelTemplate` | `core/kernel_library.py` | Persisted artifact with semver `version` and feature list |
| `InterventionResult` | `interpretability/mechanistic.py` | `feature_id`, `baseline`, `ablated`, `effect_size` |
| `GovernanceScorecard` | `interpretability/faithfulness.py` | `faithfulness_score`, `stability_score`, `risk_flag`, `passes_thresholds` |
| `HypoSpaceResult` | `api.py` | Top-level result: `.decode` + `.scorecard` |

---

## Coding Conventions

- `from __future__ import annotations` at the top of every file
- `@dataclass(slots=True)` for all data-holding classes — **never** add instance attributes outside `__init__` on slotted dataclasses
- Full type hints throughout; use `pathlib.Path` for all file paths
- One primary class per file; related dataclasses may share a file (see `config.py`)
- `snake_case` for functions/variables/modules, `PascalCase` for classes, `UPPER_CASE` for module-level constants
- Private helpers prefixed with `_` (e.g., `_intensity_band`, `_cache_key`)
- Stdlib only in all modules except `viz/` (which uses Streamlit) and `tests/` (pytest)
- JSON with UTF-8 encoding for all persisted artifacts
- Timestamps always via `data.utils.utc_timestamp()` (UTC ISO format)

---

## Persistence Layer

Artifacts are stored under `.hypo_cache/` (configurable via `RuntimeConfig.cache_dir`):

```
.hypo_cache/
├── manifest.json                    # Kernel registry: versions, file mappings, latest pointer
├── demo-layer_0-0.1.0.json          # KernelTemplate artifact (semver filename)
└── activations/
    └── <sha256_hex>.json            # Cached raw activations (keyed by content hash)
```

**Kernel versioning:** Filenames follow `{model}-{layer}-{version}.json`. Versions are semver-sorted; `KernelLibrary.load_latest()` returns the highest version. `KernelLibrary.match()` computes cross-run feature overlap by `source_index`.

---

## Testing

```bash
pytest -q          # All 8 tests should pass
```

**Test modules:**
- `test_smoke.py` — Full E2E pipeline, semver loading, kernel match/merge, governance errors
- `test_contracts.py` — Validates JSON payload keys/types for kernel artifacts and canvas output
- `test_regression.py` — Parametrized against `tests/fixtures/mini_regression_set.json`; KPI guard requires ≥80% mechanistic coverage of top features

Tests use `tmp_path` fixtures for isolation. Never modify `tests/fixtures/mini_regression_set.json` without updating expected outputs — this is the regression baseline.

---

## Important Patterns and Constraints

**Do:**
- Add new backends by subclassing/replacing `HierarchyEngine` — the `backend` string in `DecoderConfig` controls dispatch
- Use `HypoSpaceAPI` as the integration point for any new workflow; it composes all components
- Keep all inter-module data passing via the defined dataclasses (no raw dicts between layers)
- Store new persisted data as JSON in `.hypo_cache/` following the existing manifest pattern

**Don't:**
- Add external dependencies to `core/`, `data/`, or `interpretability/` — these must remain stdlib-only during the MVP phase
- Change `GovernanceConfig` defaults without updating `test_regression.py` fixture expectations
- Use `__dict__` or `setattr` on slotted dataclasses
- Create classes with inheritance hierarchies — the codebase uses composition
- Change the CLI's exit code contract: `0` = success, `2` = `GovernanceThresholdError`

---

## Roadmap Context

Planned post-MVP integrations (see `ROADMAP.md`):
- **pyvene** — replace the 50% ablation stub in `MechanisticAnalyzer` with real interventions
- **nnsight** — replace `ActivationExtractor` stub with live model hooks
- **diskcache / joblib** — replace the hand-rolled activation cache in `data/extractor.py`
- **CI/CD** — no GitHub Actions configured yet; add `.github/workflows/` when needed

KPIs to maintain: time-to-first-insight, concept consistency across runs (cross-run match rate), faithfulness coverage (≥80% of top features have mechanistic checks), CPU viability.

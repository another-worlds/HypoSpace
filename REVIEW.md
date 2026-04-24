# HypoSpace — Distributed Agent Project Structure Review

**Date:** 2026-04-24  
**Branch reviewed:** `claude/distributed-agents-review-irGHP`  
**Method:** Three parallel Explore agents fanned out across (1) core/api/cli/diagnostics, (2) data/interpretability, and (3) tests/viz/docs/CI. Findings are consolidated below by severity.

**Scope:** Architecture, data pipeline, interpretability, CLI, tests, visualization, documentation, CI/CD, and packaging.

---

## Summary Table

| Area | Severity | Issues Found |
|---|---|---|
| Packaging & Distribution | Critical | 4 |
| CI/CD | High | 5 |
| Code Correctness / Bugs | High | 3 |
| Architecture & Design | Medium | 5 |
| Reliability | Medium | 3 |
| Type Safety | Medium | 3 |
| Test Coverage | Medium | 5 |
| Documentation | Low | 6 |
| Project Metadata | Low | 4 |

---

## 1. Packaging & Distribution — Critical

### 1.1 No `pyproject.toml` or `setup.py`
The project cannot be installed as a Python package (`pip install .`). All entry points are executed as scripts from the repo root, which breaks when users try to integrate HypoSpace into other projects or virtual environments.

**Recommendation:** Add a minimal `pyproject.toml` with `[project]` metadata, `requires-python = ">=3.11"`, and a `[project.scripts]` entry for the CLI.

### 1.2 No `requirements.txt`
Runtime dependencies (`streamlit`) and optional dependencies (`nnsight`, `torch`, `pyvene`, `diskcache`) are documented only in README/QUICKSTART prose. There is no machine-readable dependency file, making reproducible installs fragile.

**Recommendation:** Add `requirements.txt` (core runtime deps) and `requirements-dev.txt` (pytest, optional deps) alongside the future `pyproject.toml`.

### 1.3 No `LICENSE` file
The repository has no license declaration. Without one, the code is legally "all rights reserved" by default, which prevents redistribution, forking, or use in downstream projects.

**Recommendation:** Add a `LICENSE` file. MIT or Apache-2.0 are common choices for research toolkits.

### 1.4 Incomplete `.gitignore`
The current `.gitignore` covers `__pycache__/`, `.pytest_cache/`, `.hypo_cache/`, and `*.pyc` but is missing:

- `.venv/`, `venv/`, `env/` — virtual environments
- `*.egg-info/`, `dist/`, `build/` — packaging artifacts
- `.coverage`, `htmlcov/` — test coverage reports
- `.vscode/`, `.idea/`, `*.swp` — IDE files

---

## 2. CI/CD Gaps — High

**File:** `.github/workflows/ci.yml`

### 2.1 Optional-dep tests never run in CI
The workflow runs `test_smoke.py`, `test_contracts.py`, `test_regression.py`, and `test_diagnostics.py` (32 tests). The 11 tests in `test_nnsight.py` and 8 tests in `test_pyvene.py` are never executed — a total of **19 tests uncovered by CI** (`.github/workflows/ci.yml:25`, `.github/workflows/ci.yml:42`).

**Recommendation:** Add a third CI job that installs `torch` (CPU-only) and runs all six test modules, or at minimum runs `test_pyvene.py` (torch only, no model download required).

### 2.2 No Python version matrix
Both CI jobs use `python-version: "3.11"` only. The project claims Python 3.11+ support but never validates 3.12 or 3.13 compatibility.

### 2.3 No linting or type-checking step
Neither `ruff`/`flake8` nor `mypy` runs in CI. Type hint inconsistencies (see §6) and style issues go undetected.

**Recommendation:** Add a `lint` job: `pip install ruff && ruff check .`

### 2.4 No test coverage reporting
There is no `coverage.py` integration or upload to a coverage service (e.g., Codecov). Coverage gaps identified in §7 are invisible to contributors.

### 2.5 Exit code contract not validated
The documented CLI contract (`exit 0` = success, `exit 2` = `GovernanceThresholdError`) is not tested in CI. A regression that changes exit codes would not be caught.

---

## 3. Code Correctness / Bugs — High

### 3.1 Cache key includes activation values (`data/extractor.py:52–55`)
`ActivationExtractor._cache_key()` hashes `(values, model_name, layer)` where `values` is the full list of activation floats. This means:
- Every unique set of activations gets its own cache entry — the cache never provides a "hit" for different activations of the same model/layer.
- For `NNSightExtractor`, which caches by `(model_name, layer_path, inputs, token_index)`, the two caching strategies are incompatible if activations flow through both paths.

**Recommendation:** Remove `values` from the key. Cache should be keyed by `(model_name, layer)` for the raw-activation path (where the caller is responsible for identity), or switch to an explicit content-addressed store with a documented contract.

### 3.2 Stability metric numerically unstable (`interpretability/faithfulness.py:63–72`)
`_stability()` computes a coefficient of variation as `std / (abs(mean) + 1e-6)`. When the mean effect size is near zero (small-magnitude features or near-perfect ablation), the denominator approaches `1e-6` and the ratio explodes, assigning artificially low stability scores to features that are actually very stable.

```python
# faithfulness.py:68–71
mean = sum(effects) / n
std = (sum((e - mean) ** 2 for e in effects) / n) ** 0.5
normalized_std = std / (abs(mean) + 1e-6)   # ← problematic near mean≈0
return max(0.0, 1.0 - normalized_std)
```

**Recommendation:** Use a range-normalized or min-max stability metric, or explicitly return a neutral score (e.g., `1.0`) when all effect sizes are below a meaningful threshold.

### 3.3 Tuple unpacking assumes tensor-first (`data/nnsight_extractor.py:173–180`)
`_tensor_to_floats()` handles tuple outputs by taking `raw[0]`, assuming the first element is always the hidden-state tensor. Some HuggingFace layers return `(hidden_states, past_key_values, attentions, ...)` where the first element may itself be a tuple on certain model versions.

```python
# nnsight_extractor.py:177–178
if isinstance(raw, tuple):
    raw = raw[0]   # ← may still be a tuple for some architectures
```

**Recommendation:** Add a recursive unwrap loop (or a `while isinstance(raw, tuple): raw = raw[0]`) and emit a warning if more than one unwrap step was needed.

---

## 4. Architecture & Design Issues — Medium

### 4.1 `_resolve_layer()` duplicated in two modules
`data/nnsight_extractor.py:187` and `data/pyvene_runner.py:186` contain verbatim copies of the same function that navigates dot-notation attribute paths through a model object. Any fix or extension must be applied in two places.

**Recommendation:** Move `_resolve_layer()` to `data/utils.py` and import it in both modules.

### 4.2 Stub interventions indistinguishable from real ones at runtime
`MechanisticAnalyzer.run_interventions()` (`interpretability/mechanistic.py:20`) always returns synthetic 50%-ablation results. `PyVeneInterventionRunner.run_interventions()` (`data/pyvene_runner.py:48`) returns real zero-ablation results. Both return `List[InterventionResult]` with no flag indicating provenance. A caller that receives a scorecard cannot determine whether it was computed from stub or real interventions.

**Recommendation:** Add an `intervention_method: str` field to `GovernanceScorecard` (e.g., `"stub-50pct"` vs `"pyvene-zero-ablation"`), or add a `source` field to `InterventionResult`.

### 4.3 Two decode paths with no call-site documentation
`HypoSpaceAPI.decode()` (Path A, stub interventions) and `HypoSpaceAPI.decode_from_model()` (Path B, nnsight + pyvene) have materially different semantics but no inline documentation explaining which path is active, when fallback occurs, or how to choose between them (`api.py:42–160`).

### 4.4 `RuntimeConfig.batch_size` is a dead field
`RuntimeConfig.batch_size` (`core/config.py:21`) is defined and documented but is never read by any other module. It does not affect `HierarchyEngine`, `NNSightExtractor`, or any other component.

**Recommendation:** Either remove the field or wire it into `NNSightExtractor` for future batched extraction.

### 4.5 `top_k=0` silently promoted to 1 (`core/hierarchy.py:35`)
```python
top_k = max(1, top_k)   # hierarchy.py:35 — no warning emitted
```
Callers that accidentally pass `top_k=0` receive one feature with no indication that their input was invalid.

---

## 5. Reliability — Medium

### 5.1 `KernelLibrary` manifest has no file lock (`core/kernel_library.py:111–128`)
`_read_manifest()` reads from disk and `_update_manifest()` writes back without any locking primitive. Concurrent calls to `save()` from two processes or threads will produce interleaved reads and writes, potentially corrupting `manifest.json`.

**Recommendation:** Use `fcntl.flock()` (Unix) or a `threading.Lock()` around the read-modify-write cycle.

### 5.2 NaN/inf values pass through the preprocessor silently (`data/preprocessor.py:9–14`)
`ActivationPreprocessor.normalize()` performs `max(abs(v) for v in numbers)`. If any value is `inf`, `max_abs = inf` and all outputs become `0.0` or `NaN`. These invalid values then propagate through `HierarchyEngine` and produce nonsensical features.

**Recommendation:** Add a validation guard at the `HypoSpaceAPI.decode()` boundary (`api.py:42`) that raises `ValueError` for NaN/inf inputs rather than propagating them silently.

### 5.3 `SemanticInterpreter.annotate()` mutates input features in-place (`interpretability/semantic.py:16`)
```python
feature.label = self._intensity_band(feature.score)   # semantic.py:16
```
Callers that hold references to the original `Feature` objects will observe silent mutation. This is a common source of aliasing bugs.

**Recommendation:** Return new `Feature` instances (e.g., via `dataclasses.replace(feature, label=...)`) rather than mutating in place.

---

## 6. Type Safety — Medium

### 6.1 `features: list` missing element type (`api.py:164`)
The private `_annotate` helper declares `features: list` with no generic parameter. The correct annotation is `features: List[Feature]`.

### 6.2 Pervasive `object` type hints in optional-dep modules
`data/nnsight_extractor.py` and `data/pyvene_runner.py` use `object` as the type for model instances, tensors, and layer references (e.g., `_lm: object | None`, `raw: object`). This defeats static type checkers and IDE autocompletion.

**Recommendation:** Use `TYPE_CHECKING` guards to import `torch.nn.Module` and related types without creating a hard runtime dependency:
```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
```

### 6.3 No input validation at the API boundary
`HypoSpaceAPI.decode()` accepts `raw_activations: Iterable[float]` but does not validate for NaN, inf, or empty input before passing to the preprocessor. Validation should happen at the boundary, not inside internal components.

---

## 7. Test Coverage Gaps — Medium

### 7.1 Seven modules lack dedicated unit tests
The following modules are covered only indirectly through smoke/regression tests:

| Module | Covered by | Gaps |
|---|---|---|
| `core/hierarchy.py` | `test_smoke.py`, `test_diagnostics.py` | Silent `top_k` promotion, empty input |
| `core/kernel_library.py` | `test_smoke.py` | Corrupt manifest, concurrent saves |
| `data/extractor.py` | `test_diagnostics.py` | Cache hit/miss logic, diskcache fallback |
| `data/preprocessor.py` | `test_diagnostics.py` | NaN/inf passthrough, all-zero input |
| `interpretability/semantic.py` | `test_regression.py` | In-place mutation, label thresholds |
| `interpretability/mechanistic.py` | `test_regression.py`, `test_diagnostics.py` | Effect size formula, edge cases |
| `interpretability/faithfulness.py` | `test_smoke.py`, `test_regression.py` | Near-zero mean instability, empty interventions |

### 7.2 `viz/canvas.py` has no standalone tests
`SemanticCanvas.to_points()` and `to_edges()` are validated only as part of `test_contracts.py:27–39`. There are no tests for single-feature input, zero-score features, or large feature sets.

### 7.3 `viz/streamlit_app.py` has zero test coverage
The UI parsing helpers (`_parse_activations`, `_feature_rows`, `_intervention_rows`) are pure functions testable without Streamlit. The `run()` function itself cannot be unit-tested but its helpers can.

### 7.4 No negative-path tests
No test exercises: NaN/inf activations, empty activation arrays, `top_k` larger than the activation vector, or a kernel library with a corrupted manifest.

### 7.5 No concurrent `KernelLibrary` access tests
The race condition in §5.1 has no corresponding test that would catch a regression if locking were added and later removed.

---

## 8. Documentation Issues — Low

### 8.1 No module-level docstrings
None of the 14 Python modules have a module-level docstring. A one-line description per file would significantly improve navigability for new contributors.

### 8.2 Misleading `SemanticInterpreter` class docstring (`interpretability/semantic.py:8`)
The docstring reads: *"Template-first semantic interpreter..."* but the implementation uses hardcoded score thresholds (0.8, 0.4) with no template system. The docstring should describe what the class actually does.

### 8.3 `MechanisticAnalyzer` not clearly identified as a stub (`interpretability/mechanistic.py`)
The class docstring mentions "CPU fallback" but does not state explicitly that `effect_size` is always `baseline * 0.5` and that results are synthetic. Users inspecting governance scorecards from the stub path may interpret them as real causal measurements.

### 8.4 Test count in `CLAUDE.md` is outdated
`CLAUDE.md` states *"47 pass, nnsight/pyvene tests skipped when deps absent"*. The actual counts are:

- Stdlib-only (always run): **32** tests (5 smoke + 2 contracts + 2 regression + 23 diagnostics)
- With nnsight + torch: **+11** tests
- With torch (pyvene): **+8** tests
- **Total when all deps available: 51**

### 8.5 Incompatible cache strategies not documented
`ActivationExtractor` uses content-keyed caching (activation values → SHA256). `NNSightExtractor` uses input-keyed caching (`(model, layer_path, inputs, token_index)`). These strategies are incompatible if the same activation is obtained via both paths. The boundary and intended usage for each cache are not documented anywhere.

### 8.6 Post-MVP backlog in `ROADMAP.md` has no status tracking
The four post-MVP backlog items (universal concept space, causal path tracing, persistent kernel library, governance export) have no status, owner, or priority information.

---

## 9. Project Metadata — Low

### 9.1 No `CONTRIBUTING.md`
There are no contribution guidelines covering code style, branch naming, PR expectations, or how to run the full test suite including optional deps.

### 9.2 No `CHANGELOG.md`
Version history is not tracked. The `KernelLibrary` uses semver artifact versioning internally, but there is no user-facing changelog.

### 9.3 No `.github/PULL_REQUEST_TEMPLATE.md`
PRs have no standard template for describing changes, test coverage, or checklist items.

### 9.4 No `.github/issue_templates/`
Bug reports and feature requests have no structured templates, leading to inconsistent issue quality.

---

## Well-Covered Areas

The following areas are in good shape and require no immediate action:

- **E2E smoke tests** (`test_smoke.py`) — 5 comprehensive tests covering the full pipeline, semver loading, kernel match/merge, and governance error propagation.
- **Regression guard** (`test_regression.py`) — 9 parametrized fixture cases + KPI enforcement (≥80% mechanistic coverage of top features).
- **Diagnostics module** (`test_diagnostics.py`) — 23 tests covering all 10 subsystem probes with edge cases; the diagnostics module itself is the most thoroughly tested component.
- **`CLAUDE.md`** — Excellent developer reference covering architecture, dataclasses, conventions, and testing expectations. The most complete documentation in the project.
- **`HypoSpaceAPI` façade** — Clean composition of all pipeline components behind a single entry point with good separation of concerns.
- **Semver kernel versioning** — `KernelLibrary` sort, merge, and match logic is well-tested and correctly handles non-lexicographic version ordering.

# HypoSpace Roadmap (MVP 6–8 weeks)

## MVP Goal

Build a lightweight end-to-end pipeline that turns PyTorch model activations into interpretable, verifiable concepts:

`model -> activations -> hierarchical kernels -> semantic labels -> mechanistic checks -> governance scorecard -> UI`

---

## Stage 1 (Weeks 1–2): Foundation + E2E Skeleton ✅ Complete

### Deliverables
- Base project structure: `core/`, `data/`, `interpretability/`, `viz/`, `api.py`, `main.py`, `tests/`
- `data/extractor.py`: activation extraction with SHA256-keyed disk cache
- `data/nnsight_extractor.py`: live extraction via nnsight model tracing
- `core/decoder.py` (v0): unified `decode(...)` entry point
- CLI smoke-run that saves artifacts to local cache

### Acceptance Criteria
- On 10–50 examples the system extracts activations and produces an initial feature set.
- The team can run the demo without manual code patches.

---

## Stage 2 (Weeks 3–4): Hierarchical Kernels + Persistence ✅ Complete

### Deliverables
- `core/hierarchy.py`: `HierarchyEngine` with magnitude-based top-k feature ranking; `backend` field stored for future SAE dispatch but no real SAE implemented yet
- `core/kernel_library.py`: `KernelTemplate`, semver versioning + training metadata, save/load/match/merge API
- Initial cross-run concept matching

### Acceptance Criteria
- A repeated run on the same model restores and matches key concepts.
- Pipeline runs on CPU without GPU. Feature ranking is magnitude-based (SAE backend integration deferred to post-MVP).

---

## Stage 3 (Weeks 5–6): Semantic + Mechanistic + Governance ✅ Complete

### Deliverables
- `interpretability/semantic.py`: template-based auto-interpretation with intensity-band labels
- `interpretability/mechanistic.py`: 50% ablation interventions on top-k features; pyvene zero-ablation via `data/pyvene_runner.py`
- `interpretability/faithfulness.py`: intervention-based checks, governance scorecard (faithfulness / stability / risk flags)

### Acceptance Criteria
- For selected features: description, intervention result, and scorecard are all available.
- Low confidence is explicitly flagged in the report.

---

## Stage 4 (Weeks 7–8): Visualization + Hardening ✅ Complete

### Deliverables
- `viz/streamlit_app.py` with tabs: Kernel Explorer, Semantic Canvas, Mechanistic Probes + Governance
- `viz/canvas.py`: hierarchy/relationship/activation-strength visualization
- Tests: smoke E2E, data-format contract tests, regression tests on a fixed mini-set
- Quickstart documentation

### Acceptance Criteria
- A new user can launch the demo and get a working report without deep configuration.
- Key scenarios pass smoke + regression checks.

---

## MVP KPIs

- **Time-to-first-insight**: first valid report < 15 minutes on a small dataset.
- **Concept consistency**: stable concept matching across repeated runs.
- **Faithfulness coverage**: ≥ 80% of top features have a mechanistic check.
- **CPU viability**: pipeline runs without GPU in a limited profile.

---

## Post-MVP Completions

Backlog items shipped after MVP:

| Integration | Status | Notes |
|---|---|---|
| nnsight live extraction | ✅ Done | `NNSightExtractor` in `data/nnsight_extractor.py`; wired into `HypoSpaceAPI.decode_from_model()` |
| pyvene interventions | ✅ Done | `PyVeneInterventionRunner` in `data/pyvene_runner.py`; provides real zero-ablation when torch is available; `MechanisticAnalyzer` 50% stub remains as CPU fallback |
| diskcache / CI-CD | ✅ Done | GitHub Actions added; cache layer updated |
| Full dep stack documented | ✅ Done | torch + nnsight + pyvene + diskcache install path documented in README, QUICKSTART, CLAUDE.md, and requirements-dev.txt; two-level install (minimal vs full) with capability table |

---

## Post-MVP Backlog

- Full USAE alignment for a cross-model universal concept space.
- Extended causal path tracing scenarios.
- Library of reusable Persistent Kernels across model families.
- Governance scorecard export to standardized reports.
- Interactive graph/canvas visualization (edges currently rendered as a dataframe only; no spatial layout, hover, or drill-down).
- Batch input workflows (no multi-example batch decode path in API or CLI).
- Streamlit multi-model comparison and result export (single-model only; no cross-run UI or export to JSON/CSV/PDF).

---

## Known Issues

Logged from the 2026-04-24 distributed-agent review. Issues fixed in the 2026-04-25 fix pass are marked ✅; remaining open issues retain their `# ISSUE-<ID>` annotation in source.

### Critical — Packaging

| ID | File | Issue |
|---|---|---|
| ISSUE-C01 ✅ | `pyproject.toml` | No `pyproject.toml`/`setup.py`; project cannot be pip-installed |
| ISSUE-C02 ✅ | `requirements.txt` | No `requirements.txt`; runtime and optional deps documented in prose only |
| ISSUE-C03 ✅ | `LICENSE` | No `LICENSE` file; legally all-rights-reserved |
| ISSUE-C04 ✅ | `.gitignore` | Missing `.venv/`, `*.egg-info/`, `.coverage`, and IDE file entries |

### High — CI/CD

| ID | File | Issue |
|---|---|---|
| ISSUE-CI01 ✅ | `.github/workflows/ci.yml` | 19 tests in `test_nnsight.py` and `test_pyvene.py` never run in CI — both now run in the `test-with-torch` job (HuggingFace model cached via `actions/cache`) |
| ISSUE-CI02 ✅ | `.github/workflows/ci.yml` | No Python version matrix (only 3.11 tested) |
| ISSUE-CI03 ✅ | `.github/workflows/ci.yml` | No linting (`ruff`/`flake8`) or type-checking (`mypy`) step |
| ISSUE-CI04 | `.github/workflows/ci.yml` | No test coverage reporting (deferred — requires external service token) |
| ISSUE-CI05 ✅ | `.github/workflows/ci.yml` | CLI exit code contract (`0`/`2`) not validated in CI |

### High — Correctness

| ID | File:Line | Issue |
|---|---|---|
| ISSUE-H01 ✅ | `data/extractor.py:_cache_key()` | Cache key includes activation values — every unique vector gets its own entry; cache never reuses for the same model/layer; incompatible with `NNSightExtractor`'s input-keyed strategy |
| ISSUE-H02 ✅ | `api.py:decode()` | No NaN/inf guard — invalid inputs pass through normalization silently, producing nonsensical downstream features |
| ISSUE-H03 ✅ | `interpretability/faithfulness.py:_stability()` | Stability denominator `abs(mean) + 1e-6` collapses to `1e-6` when mean effect size ≈ 0; ratio explodes, assigning artificially low stability to near-zero stable features |
| ISSUE-H04 ✅ | `data/nnsight_extractor.py:_tensor_to_floats()` | Tuple unwrap does `raw[0]` once only; nested tuples from some HuggingFace architectures still appear as a tuple after the first unwrap |

### Medium — Design & Reliability

| ID | File | Issue |
|---|---|---|
| ISSUE-M01 ✅ | `data/utils.py:resolve_layer()` | `_resolve_layer()` is duplicated verbatim in both files — consolidated in `data/utils.py` |
| ISSUE-M02 ✅ | `interpretability/semantic.py:annotate()` | `annotate()` mutates `Feature.label` in-place; callers holding a reference to the original object observe silent changes |
| ISSUE-M03 ✅ | `interpretability/faithfulness.py:GovernanceScorecard` | Stub `InterventionResult` values use real field names but are synthetic (`baseline * 0.5`); `GovernanceScorecard` has no provenance flag to distinguish stub from real pyvene results |
| ISSUE-M04 ✅ | `core/hierarchy.py:extract_features()` | `top_k ≤ 0` is silently promoted to 1 via `max(1, top_k)`; now raises `ValueError` |
| ISSUE-M05 ✅ | `core/kernel_library.py:_update_manifest()` | No file lock on manifest read-modify-write; concurrent `save()` calls can corrupt `manifest.json` |
| ISSUE-M06 ✅ | `api.py:_run_real_interventions()` | `features` parameter typed as bare `list` instead of `list[Feature]` |
| ISSUE-M07 ✅ | `core/config.py:RuntimeConfig` | `batch_size` field is defined and documented but never read by any component; removed |

### Low — Documentation

| ID | File | Issue |
|---|---|---|
| ISSUE-L01 ✅ | `interpretability/semantic.py:SemanticInterpreter` | Class docstring says "template-first" but implementation uses hardcoded thresholds (0.8, 0.4); no template system exists |
| ISSUE-L02 ✅ | `interpretability/mechanistic.py:run_interventions()` | Docstring does not state that `effect_size` is always `baseline * 0.5`; users may interpret stub governance scorecards as real causal measurements |

# Changelog

## [0.1.4] — 2026-04-27

### Added
- `data/sae_backend.py` — new optional-dependency module (torch) implementing the `FeatureBackend` protocol:
  - `MagnitudeBackend` — stdlib-only magnitude top-k; satisfies the protocol without torch
  - `SAEEncodeResult` — `@dataclass(slots=True)` intermediate for raw sparse SAE output
  - `MatryoshkaBackend` — loads a Linear encoder checkpoint (directory `encoder.pt` or single `.pt` file), runs Linear → ReLU → sparse top-k; supports `matryoshka`, `topk`, and `jumprelu` backend strings
  - `build_backend()` — factory called by `api.py`; returns `None` (magnitude fallback) when torch is absent, `sae_path=None`, or checkpoint load fails (emits `UserWarning`)
- `core/hierarchy.py`: `FeatureBackend` `@runtime_checkable` Protocol; `_magnitude_top_k()` extracted as module-level function; `HierarchyEngine` accepts `feature_backend: FeatureBackend | None = None` and dispatches to it
- `core/config.py`: `DecoderConfig.sae_path: str | None = None` — path to SAE checkpoint; `None` triggers silent magnitude fallback
- `core/decoder.py`: `RealityDecoder.__init__` threads `feature_backend` through to `HierarchyEngine`
- `api.py`: `HypoSpaceAPI._build_sae_backend()` constructs the backend at init time from config; wired into `RealityDecoder`
- `diagnostics.py`: `_probe_sae_backend` (probe #12, skips without torch; 6 checks: import, `sae_available()`, `build_backend` null path, `MagnitudeBackend.extract()`, protocol `isinstance`, injection); `sae_backend` `DependencyStatus` entry; schema bumped `1.0.0 → 1.1.0`
- 20 new tests: 5 in `test_units.py` (protocol/injection, stdlib-only), 10 in new `test_sae_backend.py` (skipped without torch), 1 in `test_smoke.py` (SAE fallback E2E), 4 in `test_diagnostics.py`

### Changed
- `Feature.id` format: magnitude backend continues `"{layer}:feature:{rank}"`; SAE backend produces `"{layer}:sae:{rank}:{sae_dict_index}"` — machine-readable provenance, no artifact schema change
- Test suite: 94 → 103 stdlib-only tests; 113 → 133 total with full optional stack
- `DIAGNOSTICS_VERSION`: `"1.0.0"` → `"1.1.0"`

---

## [0.1.3] — 2026-04-27

### Fixed
- `data/utils.py:resolve_layer()` now raises `ValueError` with the failed segment and full path as context, instead of propagating a bare `AttributeError`/`IndexError`
- `api.py:_run_real_interventions()` no longer accesses the private `NNSightExtractor._lm` attribute directly; uses the new `loaded_model` property instead
- CLI threshold flags (`--min-faithfulness`, `--min-stability`, `--high-intensity-threshold`, `--medium-intensity-threshold`) now reject values outside `[0, 1]` with a clear argparse error
- `core/decoder.py`: first-run `KeyError` on cross-run match rate is now surfaced as a `warnings.warn()` instead of being silently swallowed
- `data/pyvene_runner.py:_layer_index()` now warns when no integer segment is found in the layer path and the index falls back to 0
- `requirements.txt`: removed `streamlit` (it is optional per `pyproject.toml`); replaced with an install hint comment

### Added
- `NNSightExtractor.loaded_model` property — raises `RuntimeError` if called before `extract()`, replacing direct `_lm` access at call sites
- 5 new unit tests in `test_units.py`: `resolve_layer()` valid attribute path, valid integer index, bad attribute (asserts `ValueError`), out-of-bounds index (asserts `ValueError`), and decoder first-run cross-run match rate equals `"0.000"`

### Changed
- `core/config.py`: removed unused `from typing import Dict` import
- Test suite: 89 → 94 stdlib-only tests; 108 → 113 total with full optional stack

---

## [0.1.2] — 2026-04-27

### Fixed
- `ActivationExtractor._cache_key()` now scopes cache entries to `(model_name, layer, values)` — different models with identical activation vectors no longer collide in the cache
- `KernelLibrary.load()` raises `ValueError` on corrupt JSON artifacts instead of propagating a bare `json.JSONDecodeError`

### Added
- CLI live-model extraction flags: `--layer-path`, `--inputs`, `--token-index` expose `decode_and_score_from_model()` directly from `main.py`; missing nnsight/torch produces a clear error and exit code 1
- `DecoderConfig.high_intensity_threshold` and `DecoderConfig.medium_intensity_threshold` make semantic intensity-band labeling configurable (defaults unchanged: 0.8 / 0.4); wired into `SemanticInterpreter`, `HypoSpaceAPI`, and CLI flags `--high-intensity-threshold` / `--medium-intensity-threshold`
- `HypoSpaceAPI.decode()` validates non-empty `model_name` / `layer`, semver `version` format (`X.Y.Z`), and activation array length against `max_features`
- `test_nnsight.py` (11 tests) now runs in the `test-with-torch` CI job alongside `test_pyvene.py`
- 12 new unit tests in `test_units.py` covering cache key scoping, configurable thresholds, corrupt-kernel guard, and input validation

### Changed
- Test suite: 77 → 89 stdlib-only tests; 96 → 108 total with full optional stack

---

## [0.1.1] — 2026-04-26

### Changed
- Replaced `object` annotations with `Any` in all source modules
- Added one-line module-level docstrings to all 17 source files
- `python -m pytest` documented as the canonical test invocation in all docs

### Added
- `CONTRIBUTING.md`, `CHANGELOG.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md`
- `tests/test_canvas.py` (7 tests) and `tests/test_units.py` (20 tests)
- Test suite grows from 69 → 96 (77 pass on minimal install, 96 on full)

---

## [0.1.0] — 2026-04-25

### Added
- Full E2E pipeline: activations → features → semantic labels → mechanistic checks → governance scorecard
- CLI (`main.py`) and Streamlit UI (`viz/streamlit_app.py`)
- KernelLibrary with semver versioning, save/load/match/merge
- NNSightExtractor for live HuggingFace model extraction
- PyVeneInterventionRunner for real zero-ablation interventions (torch + pyvene)
- diskcache optional backend for ActivationExtractor
- Governance probe system (11 subsystem probes, DiagnosticsReport)
- GitHub Actions CI (stdlib, diskcache, torch/pyvene, lint jobs)

### Fixed
- Cache key incompatibility between ActivationExtractor and NNSightExtractor (ISSUE-H01)
- NaN/inf guard at api.py boundary (ISSUE-H02)
- Stability metric near-zero denominator (ISSUE-H03)
- Nested tuple unwrap in NNSightExtractor (ISSUE-H04)
- pyvene_runner.py compatibility with nnsight 0.6.x (`lm.model` → `lm._module`)

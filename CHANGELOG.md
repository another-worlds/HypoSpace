# Changelog

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

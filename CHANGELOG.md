# Changelog

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

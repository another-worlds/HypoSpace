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
- `core/hierarchy.py`: Matryoshka SAE wrapper (priority backend) with CPU fallback for compact dictionaries
- `core/kernel_library.py`: `KernelTemplate`, semver versioning + training metadata, save/load/match/merge API
- Initial cross-run concept matching

### Acceptance Criteria
- A repeated run on the same model restores and matches key concepts.
- CPU fallback works in limited mode (small dictionaries, batch=1).

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
| pyvene interventions | ✅ Done | `PyVeneInterventionRunner` in `data/pyvene_runner.py`; replaces stub in `MechanisticAnalyzer` |
| diskcache / CI-CD | ✅ Done | GitHub Actions added; cache layer updated |

---

## Post-MVP Backlog

- Full USAE alignment for a cross-model universal concept space.
- Extended causal path tracing scenarios.
- Library of reusable Persistent Kernels across model families.
- Governance scorecard export to standardized reports.

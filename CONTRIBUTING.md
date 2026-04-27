# Contributing to HypoSpace

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-dev.txt
```

This installs the full optional stack: torch (CPU), nnsight, pyvene, diskcache, and pytest.

## Running tests

```bash
# Full suite — all 108 tests (requires full optional stack)
python -m pytest -q

# Targeted test module
python -m pytest tests/test_smoke.py -v
python -m pytest tests/test_diagnostics.py -v

# Stdlib-only (no optional deps required)
python -m pytest tests/test_smoke.py tests/test_contracts.py tests/test_regression.py tests/test_diagnostics.py -v
```

## Code style

- `from __future__ import annotations` as the first line of every file
- Module-level docstring (one line) immediately after the `from __future__` import
- `snake_case` for functions/variables/modules, `PascalCase` for classes, `UPPER_CASE` for module-level constants
- Private helpers prefixed with `_`
- Full type hints throughout; use `pathlib.Path` for file paths
- No `object` annotations — use `Any` from `typing` when the type is truly unknown
- `@dataclass(slots=True)` for all data-holding classes

## Branch naming

```
<type>/<short-description>
```

Examples: `fix/nan-guard`, `feat/usae-backend`, `docs/contributing`

## PR expectations

- `python -m pytest -q` must pass (108/108 or more)
- `python main.py --diagnostics` must show no new errors
- No new `object` type annotations in any module
- One-line module docstring in every new source file
- New behaviour is covered by tests

## Adding optional dependencies

Only these modules may import optional packages:

| Module | Allowed optional deps |
|---|---|
| `data/nnsight_extractor.py` | `nnsight`, `torch` |
| `data/pyvene_runner.py` | `pyvene`, `torch` |
| `data/extractor.py` | `diskcache` |
| `viz/streamlit_app.py`, `viz/canvas.py` | `streamlit` |

All other modules (`core/`, `interpretability/`, `api.py`, `main.py`, `diagnostics.py`) must remain stdlib-only.

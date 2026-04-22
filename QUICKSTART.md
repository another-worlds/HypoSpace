# HypoSpace Quickstart (MVP)

Fast reference — see README for full detail on each step.

## 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip pytest streamlit
```

## 2) Check subsystem health

```bash
python main.py --diagnostics
```

Prints a JSON report showing the status of every subsystem and which optional dependencies (torch, nnsight, pyvene, diskcache) are available. Run this first after any environment change.

## 3) Run CLI demo

```bash
python main.py --model demo-model --layer layer_0 --activations 0.1,0.4,-0.2,0.8 --top-k 4
```

Writes kernel artifacts into `.hypo_cache/` (or your configured cache dir).

## 4) Live model extraction (optional — requires nnsight + torch)

```bash
pip install nnsight torch
python - <<'EOF'
from api import HypoSpaceAPI
api = HypoSpaceAPI()
result = api.decode_and_score_from_model(
    model_name="gpt2",
    layer="layer_0",
    layer_path="transformer.h.0",
    inputs="The quick brown fox",
    token_index=-1,
    version="0.1.0",
)
print(result.scorecard)
EOF
```

## 5) Run Streamlit explorer

```bash
streamlit run viz/streamlit_app.py
```

Tabs: Kernel Explorer · Semantic Canvas · Mechanistic Probes + Governance

## 6) Run tests

```bash
pytest -q
```

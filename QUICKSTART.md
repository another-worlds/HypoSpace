# HypoSpace Quickstart (MVP)

## 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip pytest streamlit
```

## 2) Run CLI demo

```bash
python main.py --model demo-model --layer layer_0 --activations 0.1,0.4,-0.2,0.8 --top-k 4
```

The command writes kernel artifacts into `.hypo_cache/` (or your configured cache dir).

## 3) Run Streamlit explorer

```bash
streamlit run viz/streamlit_app.py
```

Tabs included in the MVP UI:

- Kernel Explorer
- Semantic Canvas
- Mechanistic Probes + Governance

## 4) Run tests

```bash
pytest -q
```

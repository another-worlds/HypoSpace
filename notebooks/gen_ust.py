#!/usr/bin/env python3
"""Generator: produces universal_semantic_transposer.ipynb"""

import json
from pathlib import Path

cells = []
_cid = [0]

def _id():
    _cid[0] += 1
    return f"c{_cid[0]:04d}"

def md(text: str):
    cells.append({"cell_type": "markdown", "id": _id(), "metadata": {}, "source": text})

def code(text: str):
    cells.append({
        "cell_type": "code", "execution_count": None,
        "id": _id(), "metadata": {}, "outputs": [], "source": text.lstrip("\n"),
    })

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Title
# ─────────────────────────────────────────────────────────────────────────────
md("""\
# Universal Semantic Transposition System (UST)

A minimal but functional system that **attaches to any neural network** at every layer \
in parallel, learning to translate raw activations into interpretable semantic concepts.

## Four classes

| Class | Role |
|---|---|
| `LayerHookCollector` | Intercepts every layer via `register_forward_hook` (torch) or `inject()` (numpy) |
| `PerLayerSAE` | Sparse Autoencoder: Linear+ReLU+sparsity loss (torch) or magnitude top-k (numpy) |
| `UniversalSemanticTransposer` | Orchestrates SAE training, cosine cross-layer tracking, KernelLibrary persistence |
| `CrossModalAligner` | Orthogonal Procrustes via `np.linalg.svd` — aligns feature spaces across modalities |

## Four modalities
1. **Text / LLM-style** — 2-layer MLP on synthetic token embeddings (noun / verb / function-word clusters)
2. **Time Series** — 2-layer 1D CNN on sine waves at 1 Hz / 5 Hz / 20 Hz
3. **Audio Spectrum** — MLP on FFT magnitudes of the *same* sine waves → tests cross-modal "5 Hz" concept
4. **Visual / Video Frames** — 2D CNN on 32×32 synthetic patterns (stripes, circles, noise)

HypoSpace imports but never modifies: `Feature`, `KernelLibrary`, `KernelTemplate`, \
`SemanticInterpreter`, `FaithfulnessChecker`, `MechanisticAnalyzer`, \
`ActivationPreprocessor`, `SemanticCanvas`, `utc_timestamp`
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Setup
# ─────────────────────────────────────────────────────────────────────────────
code("""
from __future__ import annotations

import os, sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── HypoSpace path ────────────────────────────────────────────────────────────
try:
    _ROOT = str(Path(__file__).parent.parent)
except NameError:
    _ROOT = os.path.abspath("..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.hierarchy import Feature, _magnitude_top_k
from core.kernel_library import KernelLibrary, KernelTemplate
from data.preprocessor import ActivationPreprocessor
from data.utils import utc_timestamp
from interpretability.semantic import SemanticInterpreter
from interpretability.mechanistic import MechanisticAnalyzer, InterventionResult
from interpretability.faithfulness import FaithfulnessChecker, GovernanceScorecard
from viz.canvas import SemanticCanvas

# ── torch (optional) ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    print(f"torch {torch.__version__} — trainable SAE + real models")
except ImportError:
    TORCH_AVAILABLE = False
    print("torch not available — numpy stubs + magnitude top-k fallback")

# ── nnsight (doubly optional, GPT-2 section only) ────────────────────────────
try:
    from data.nnsight_extractor import NNSightExtractor
    NNSIGHT_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    NNSIGHT_AVAILABLE = False

# ── cache (isolated from main .hypo_cache) ────────────────────────────────────
try:
    CACHE_DIR = Path(__file__).parent / ".ust_cache"
except NameError:
    CACHE_DIR = Path(".ust_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── shared HypoSpace instances ────────────────────────────────────────────────
preprocessor = ActivationPreprocessor()
interpreter  = SemanticInterpreter(high=0.8, medium=0.4)
analyzer     = MechanisticAnalyzer()
checker      = FaithfulnessChecker()
canvas       = SemanticCanvas()
library      = KernelLibrary(root=CACHE_DIR)

print(f"Cache: {CACHE_DIR.resolve()}")
print(f"TORCH={TORCH_AVAILABLE}  NNSIGHT={NNSIGHT_AVAILABLE}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — LayerHookCollector
# ─────────────────────────────────────────────────────────────────────────────
md("## Core Classes")

code("""
class LayerHookCollector:
    \"\"\"
    Attaches register_forward_hook() to every named layer of any torch.nn.Module.
    Activation shapes are normalised:
        4D (B, C, H, W) -> spatial mean-pool -> (C,)
        3D (B, C, L)    -> sequence mean-pool -> (C,)
        2D (B, H)       -> batch-0 slice      -> (H,)
    Tuple outputs: first element used.

    Numpy path: call inject(layer_name, values) after each manual forward pass.
    \"\"\"

    def __init__(self, model=None):
        self.collected: Dict[str, List[float]] = {}
        self._handles: List = []
        if TORCH_AVAILABLE and model is not None:
            self._attach(model)

    def _attach(self, model):
        for name, module in model.named_modules():
            if name == "":
                continue
            self._handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def _fn(module, inp, out):
            t = (out[0] if isinstance(out, tuple) else out).detach()
            if   t.dim() == 4: t = t[0].mean(dim=(-2, -1))
            elif t.dim() == 3: t = t[0].mean(dim=-1)
            else:               t = t[0]
            self.collected[name] = t.tolist()
        return _fn

    def inject(self, name: str, values: List[float]):
        self.collected[name] = list(values)

    def clear(self):
        self.collected.clear()

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

print("LayerHookCollector ready.")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — PerLayerSAE
# ─────────────────────────────────────────────────────────────────────────────
code("""
class PerLayerSAE:
    \"\"\"
    Sparse Autoencoder for one layer.
    Torch path : Linear encoder + ReLU, trained with recon + L1 sparsity loss.
    Numpy path : delegates to _magnitude_top_k (HypoSpace stdlib fallback).
    Both paths produce List[Feature] compatible with all HypoSpace consumers.
    \"\"\"

    EXPANSION    = 4
    TOP_K        = 8
    EPOCHS       = 50
    LR           = 1e-3
    SPARSITY     = 0.01

    def __init__(self, layer_name, input_dim, top_k=TOP_K,
                 dict_size=None, n_epochs=EPOCHS, lr=LR):
        self.layer_name = layer_name
        self.input_dim  = input_dim
        self.top_k      = top_k
        self.dict_size  = dict_size or max(input_dim * self.EXPANSION, top_k + 1)
        self.n_epochs   = n_epochs
        self.lr         = lr
        self._trained   = False
        self._losses: List[float] = []
        if TORCH_AVAILABLE:
            self._enc = nn.Linear(input_dim, self.dict_size, bias=True)
            self._dec = nn.Linear(self.dict_size, input_dim, bias=True)
            nn.init.xavier_uniform_(self._enc.weight)
            nn.init.xavier_uniform_(self._dec.weight)

    def train_on_activations(self, matrix: np.ndarray) -> List[float]:
        \"\"\"matrix: (n_samples, input_dim) float32. Returns per-epoch losses.\"\"\"
        if not TORCH_AVAILABLE:
            self._trained = True
            return []
        X   = torch.tensor(matrix, dtype=torch.float32)
        opt = optim.Adam(list(self._enc.parameters()) + list(self._dec.parameters()),
                         lr=self.lr)
        losses = []
        for _ in range(self.n_epochs):
            opt.zero_grad()
            latent = torch.relu(self._enc(X))
            loss   = ((self._dec(latent) - X)**2).mean() + self.SPARSITY * latent.abs().mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self._trained = True
        self._losses = losses
        return losses

    def extract_features(self, activation: List[float]) -> List[Feature]:
        if TORCH_AVAILABLE and self._trained:
            return self._sae_features(activation)
        return self._magnitude_features(activation)

    def _sae_features(self, activation):
        with torch.no_grad():
            latent = torch.relu(self._enc(torch.tensor(activation, dtype=torch.float32))).tolist()
        ranked = sorted(enumerate(latent), key=lambda t: t[1], reverse=True)
        return [
            Feature(id=f"{self.layer_name}:sae:{r}:{i}", layer=self.layer_name,
                    score=float(s), source_index=i)
            for r, (i, s) in enumerate(ranked[:self.top_k])
        ]

    def _magnitude_features(self, activation):
        return _magnitude_top_k(preprocessor.normalize(activation),
                                self.layer_name, self.top_k)

    def feature_vector(self, activation: List[float]) -> np.ndarray:
        \"\"\"Full latent — used for cosine similarity across layers / modalities.\"\"\"
        if TORCH_AVAILABLE and self._trained:
            with torch.no_grad():
                return torch.relu(
                    self._enc(torch.tensor(activation, dtype=torch.float32))
                ).numpy()
        return np.array(preprocessor.normalize(activation), dtype=np.float32)

print("PerLayerSAE ready.")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4b — SemanticLabeler
# ─────────────────────────────────────────────────────────────────────────────
code("""
class SemanticLabeler:
    \"\"\"
    Assigns human-readable concept names to SAE features via prototype activation.

    Workflow:
      1. Call fit(prototypes) where prototypes = {concept_name: layer_activation_vector}.
         For each prototype, run the activation through the trained SAE encoder and
         find which latent indices fire most strongly.  Those indices become labelled
         "concept detectors" in the concept_map.
      2. Call label(features) to replace Feature.label with the concept name whenever
         feature.source_index is a known concept detector.

    This is the bridge from pure statistics ("high-intensity concept around index 24")
    to genuine semantics ("5Hz frequency detector", "noun-word direction", ...).
    The key ingredient is *ground-truth knowledge of what each prototype input means* —
    something the SAE alone cannot know, but the researcher supplies.
    \"\"\"

    def __init__(self, sae: PerLayerSAE, top_per_concept: int = 2):
        self.sae             = sae
        self.top_per_concept = top_per_concept
        self.concept_map: Dict[int, str]   = {}
        self._scores:     Dict[int, float] = {}

    def fit(self, prototypes: Dict[str, List[float]]) -> "SemanticLabeler":
        \"\"\"prototypes: {concept_name -> layer activation vector for that concept}.\"\"\"
        strength: Dict[int, Tuple[str, float]] = {}
        for name, activation in prototypes.items():
            latent = self.sae.feature_vector(activation)
            ranked = sorted(range(len(latent)), key=lambda i: float(latent[i]), reverse=True)
            for idx in ranked[:self.top_per_concept]:
                score = float(latent[idx])
                if score < 1e-6:
                    continue
                if idx not in strength or score > strength[idx][1]:
                    strength[idx] = (name, score)
        self.concept_map = {i: n for i, (n, _) in strength.items()}
        self._scores     = {i: s for i, (_, s) in strength.items()}
        return self

    def label(self, features: List[Feature]) -> List[Feature]:
        \"\"\"Return new Feature list with concept names substituted for known detectors.\"\"\"
        out = []
        for f in features:
            concept = self.concept_map.get(f.source_index)
            out.append(Feature(id=f.id, layer=f.layer, score=f.score,
                               source_index=f.source_index,
                               label=concept if concept else f.label))
        return out

    def coverage(self, features: List[Feature]) -> float:
        if not features:
            return 0.0
        return sum(1 for f in features if f.source_index in self.concept_map) / len(features)

print("SemanticLabeler ready.")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — UniversalSemanticTransposer
# ─────────────────────────────────────────────────────────────────────────────
code("""
class UniversalSemanticTransposer:
    \"\"\"
    Full pipeline: hook collection -> per-layer SAE -> cross-layer cosine tracking
    -> KernelLibrary persistence -> governance scorecard.
    \"\"\"

    COSINE_THRESHOLD = 0.7

    def __init__(self, model_name, version="1.0.0", top_k=8,
                 cosine_threshold=COSINE_THRESHOLD):
        self.model_name        = model_name
        self.version           = version
        self.top_k             = top_k
        self.cosine_threshold  = cosine_threshold
        self.layer_features:   Dict[str, List[Feature]]        = {}
        self.layer_saes:       Dict[str, PerLayerSAE]          = {}
        self.cross_layer_links: List[Tuple[str, str, float]]   = []
        self.kernel_paths:     Dict[str, str]                  = {}

    def run(self, collector: LayerHookCollector,
            training_data: Optional[Dict[str, List[List[float]]]] = None,
            sae_epochs: int = 50,
            concept_prototypes: Optional[Dict[str, Dict[str, List[float]]]] = None,
            ) -> Dict[str, List[Feature]]:
        \"\"\"
        concept_prototypes: {layer_name -> {concept_name -> activation_vector}}
        When provided, SemanticLabeler replaces intensity-band labels with concept names
        for every feature whose source_index matches a known concept detector.
        \"\"\"
        if not collector.collected:
            raise ValueError("No activations — run a forward pass first.")
        for layer, act in collector.collected.items():
            if not act:
                continue
            sae = PerLayerSAE(layer_name=layer, input_dim=len(act),
                              top_k=self.top_k, n_epochs=sae_epochs)
            matrix = np.array(
                training_data[layer] if (training_data and layer in training_data) else [act],
                dtype=np.float32)
            sae.train_on_activations(matrix)
            self.layer_saes[layer] = sae
            raw_feats = sae.extract_features(act)
            if concept_prototypes and layer in concept_prototypes:
                # Concept labeling: SAE latent → prototype alignment (genuine discovery)
                labeler = SemanticLabeler(sae, top_per_concept=2)
                labeler.fit(concept_prototypes[layer])
                feats = labeler.label(raw_feats)
            else:
                # No concept prototypes supplied — fall back to intensity-band annotation
                # so downstream code (HypoSpace API) still gets labels.
                feats = interpreter.annotate(raw_feats)
            self.layer_features[layer] = feats
        self._cross_layer(collector)
        self._save(collector)
        return self.layer_features

    def _cross_layer(self, collector):
        names = list(collector.collected.keys())
        for la, lb in zip(names, names[1:]):
            if la not in self.layer_saes or lb not in self.layer_saes:
                continue
            va = self.layer_saes[la].feature_vector(collector.collected[la])
            vb = self.layer_saes[lb].feature_vector(collector.collected[lb])
            # Pad shorter vector so dimensions match
            d = max(len(va), len(vb))
            if len(va) < d:
                va = np.concatenate([va, np.zeros(d - len(va), dtype=va.dtype)])
            if len(vb) < d:
                vb = np.concatenate([vb, np.zeros(d - len(vb), dtype=vb.dtype)])
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na < 1e-8 or nb < 1e-8:
                continue
            sim = float(np.dot(va, vb) / (na * nb))
            if sim >= self.cosine_threshold:
                self.cross_layer_links.append((la, lb, sim))

    def _save(self, collector):
        for layer, features in self.layer_features.items():
            kid = f"{self.model_name}-{layer.replace('.', '_')}"
            path = library.save(KernelTemplate(
                kernel_id=kid, version=self.version,
                model_name=self.model_name, layer=layer,
                features=features,
                metadata={"created_at_utc": utc_timestamp(),
                          "backend": "sae" if TORCH_AVAILABLE else "magnitude",
                          "top_k": str(self.top_k)},
            ))
            self.kernel_paths[layer] = str(path)

    def governance_scorecard(self) -> GovernanceScorecard:
        feats = [f for fs in self.layer_features.values() for f in fs]
        if not feats:
            return GovernanceScorecard(faithfulness_score=0.0, stability_score=0.0,
                                       risk_flag="no_data", passes_thresholds=False,
                                       intervention_method="stub-50pct")
        return checker.evaluate(analyzer.run_interventions(feats),
                                intervention_method="stub-50pct")

print("UniversalSemanticTransposer ready.")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — CrossModalAligner
# ─────────────────────────────────────────────────────────────────────────────
code("""
class CrossModalAligner:
    \"\"\"
    Orthogonal Procrustes alignment (numpy-only, no scipy).
        M = A^T B,   U S Vt = svd(M),   R = U Vt
    Aligns A onto B; R minimises ||A @ R - B||_F.
    \"\"\"

    def __init__(self):
        self.rotation_: Optional[np.ndarray] = None
        self._dim: int = 0

    def _pad(self, A: np.ndarray, d: int) -> np.ndarray:
        c = A.shape[1]
        if c == d:   return A
        if c <  d:   return np.hstack([A, np.zeros((A.shape[0], d - c), dtype=A.dtype)])
        return A[:, :d]

    def fit(self, A: np.ndarray, B: np.ndarray) -> "CrossModalAligner":
        d  = max(A.shape[1], B.shape[1])
        Ap = self._pad(A, d);  Bp = self._pad(B, d)
        n  = min(Ap.shape[0], Bp.shape[0])
        U, _, Vt = np.linalg.svd(Ap[:n].T @ Bp[:n], full_matrices=False)
        self.rotation_ = U @ Vt
        self._dim = d
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        if self.rotation_ is None:
            raise RuntimeError("Call fit() first.")
        return self._pad(A, self._dim) @ self.rotation_

    def fit_transform(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return self.fit(A, B).transform(A)

    def find_correspondences(self, A_aligned, B, ids_a, ids_b,
                             threshold=0.6) -> List[Tuple[str, str, float]]:
        def _norm(M):
            n = np.linalg.norm(M, axis=1, keepdims=True)
            return M / np.where(n < 1e-8, 1.0, n)
        Bp  = self._pad(B, A_aligned.shape[1])
        sim = _norm(A_aligned) @ _norm(Bp).T
        out = []
        for i, fa in enumerate(ids_a):
            j = int(np.argmax(sim[i]))
            if float(sim[i, j]) >= threshold and j < len(ids_b):
                out.append((fa, ids_b[j], float(sim[i, j])))
        return out

print("CrossModalAligner ready.")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Synthetic data generators
# ─────────────────────────────────────────────────────────────────────────────
md("## Synthetic Data\nKnown concept structure makes cross-modal alignment results interpretable.")

code("""
def make_text_embeddings(seq_len=8, embed_dim=32, seed=0):
    \"\"\"Token embeddings: 3 clusters — noun / verb / function-word.\"\"\"
    rng = np.random.default_rng(seed)
    centres = {
        "noun": np.array([1., 0., 0.] + [0.]*(embed_dim-3), dtype=np.float32),
        "verb": np.array([0., 1., 0.] + [0.]*(embed_dim-3), dtype=np.float32),
        "func": np.array([0., 0., 1.] + [0.]*(embed_dim-3), dtype=np.float32),
    }
    pat = ["noun","verb","func","noun","verb","func","noun","func"]
    rows = [centres[pat[i % len(pat)]] +
            rng.standard_normal(embed_dim).astype(np.float32)*0.1
            for i in range(seq_len)]
    return np.stack(rows)   # (seq_len, embed_dim)


def make_time_series(n_samples=16, signal_len=64, seed=42):
    \"\"\"Sine waves at 1/5/20 Hz — each frequency is one concept.\"\"\"
    rng  = np.random.default_rng(seed)
    t    = np.linspace(0, 1.0, signal_len, dtype=np.float32)
    fmap = {"1Hz": 1.0, "5Hz": 5.0, "20Hz": 20.0}
    keys = list(fmap.keys())
    sigs, lbls = [], []
    for i in range(n_samples):
        lbl = keys[i % len(keys)]
        sig = np.sin(2*np.pi*fmap[lbl]*t) + rng.standard_normal(signal_len).astype(np.float32)*0.05
        sigs.append(sig); lbls.append(lbl)
    return np.array(sigs, dtype=np.float32)[:, np.newaxis, :], lbls   # (N,1,L)


def make_audio_spectra(n_samples=16, signal_len=64, seed=42):
    \"\"\"FFT magnitudes of the SAME sine waves — cross-modal 5 Hz concept test.\"\"\"
    sigs, lbls = make_time_series(n_samples, signal_len, seed)
    spec = np.abs(np.fft.rfft(sigs[:, 0, :], axis=1)).astype(np.float32)
    maxv = spec.max(axis=1, keepdims=True)
    return spec / np.where(maxv < 1e-8, 1.0, maxv), lbls   # (N, L//2+1)


def make_visual_frames(n_frames=12, H=32, W=32, seed=7):
    \"\"\"32x32 frames: horizontal stripes / concentric circles / noise.\"\"\"
    rng = np.random.default_rng(seed)
    cyc = ["stripes","circles","noise"]
    y, x = np.meshgrid(np.linspace(-1,1,H), np.linspace(-1,1,W), indexing="ij")
    frames, lbls = [], []
    for i in range(n_frames):
        lbl = cyc[i % len(cyc)]
        if   lbl == "stripes": frame = np.sin(2*np.pi*3*y).astype(np.float32)
        elif lbl == "circles": frame = np.sin(2*np.pi*3*np.sqrt(x**2+y**2)).astype(np.float32)
        else:                  frame = rng.standard_normal((H,W)).astype(np.float32)
        frames.append(frame); lbls.append(lbl)
    return np.array(frames, dtype=np.float32)[:, np.newaxis, :, :], lbls   # (N,1,H,W)


# smoke-test
for name, arr in [("text",     make_text_embeddings()),
                  ("ts",       make_time_series()[0]),
                  ("audio",    make_audio_spectra()[0]),
                  ("visual",   make_visual_frames()[0])]:
    print(f"  {name:8s}: {arr.shape}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Model definitions
# ─────────────────────────────────────────────────────────────────────────────
md("## Model Definitions\nTiny models for each modality. Torch path uses real `nn.Module`; numpy path uses seeded random-weight stubs.")

code("""
if TORCH_AVAILABLE:

    class TextMLP(nn.Module):
        def __init__(self, embed_dim=32, hidden_dim=64, n_classes=3):
            super().__init__()
            self.layer0 = nn.Linear(embed_dim, hidden_dim)
            self.act0   = nn.ReLU()
            self.layer1 = nn.Linear(hidden_dim, n_classes)
        def forward(self, x):   # (B, seq, embed) -> (B, classes)
            return self.layer1(self.act0(self.layer0(x.mean(dim=1))))

    class TimeSeriesCNN(nn.Module):
        def __init__(self, n_classes=3):
            super().__init__()
            self.conv1 = nn.Conv1d(1,  8, kernel_size=7, padding=3)
            self.act1  = nn.ReLU()
            self.conv2 = nn.Conv1d(8, 16, kernel_size=5, padding=2)
            self.act2  = nn.ReLU()
            self.pool  = nn.AdaptiveAvgPool1d(1)
            self.fc    = nn.Linear(16, n_classes)
        def forward(self, x):   # (B,1,L)
            return self.fc(self.pool(self.act2(self.conv2(self.act1(self.conv1(x))))).squeeze(-1))

    class AudioMLP(nn.Module):
        def __init__(self, fft_dim=33, hidden_dim=48, n_classes=3):
            super().__init__()
            self.layer0 = nn.Linear(fft_dim, hidden_dim)
            self.act0   = nn.ReLU()
            self.layer1 = nn.Linear(hidden_dim, n_classes)
        def forward(self, x):
            return self.layer1(self.act0(self.layer0(x)))

    class VisualCNN(nn.Module):
        def __init__(self, n_classes=3):
            super().__init__()
            self.conv1 = nn.Conv2d(1,  8, kernel_size=5, padding=2)
            self.act1  = nn.ReLU()
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.act2  = nn.ReLU()
            self.pool  = nn.AdaptiveAvgPool2d(1)
            self.fc    = nn.Linear(16, n_classes)
        def forward(self, x):   # (B,1,H,W)
            return self.fc(self.pool(self.act2(self.conv2(self.act1(self.conv1(x))))).flatten(1))

    def _collect_batch(model, tensor_list, collector) -> Dict[str, List[List[float]]]:
        \"\"\"Run multiple forward passes; aggregate activations per layer.\"\"\"
        training: Dict[str, List[List[float]]] = {}
        for x in tensor_list:
            collector.clear()
            with torch.no_grad():
                model(x)
            for name, acts in collector.collected.items():
                training.setdefault(name, []).append(list(acts))
        return training

    print("Torch models: TextMLP, TimeSeriesCNN, AudioMLP, VisualCNN")

else:

    class _Stub:
        \"\"\"Seeded 2-layer MLP stub (numpy fallback).\"\"\"
        def __init__(self, in_dim, hidden, out_dim, seed=0):
            rng = np.random.default_rng(seed)
            self.W0 = rng.standard_normal((in_dim, hidden)).astype(np.float32) * 0.1
            self.b0 = np.zeros(hidden, dtype=np.float32)
            self.W1 = rng.standard_normal((hidden, out_dim)).astype(np.float32) * 0.1
            self.b1 = np.zeros(out_dim, dtype=np.float32)
            self._h0 = np.zeros(hidden, dtype=np.float32)
            self._h1 = np.zeros(out_dim, dtype=np.float32)
        def forward(self, x: np.ndarray):
            flat     = x.ravel()[:self.W0.shape[0]]
            self._h0 = np.maximum(0.0, flat @ self.W0 + self.b0)
            self._h1 = self._h0 @ self.W1 + self.b1
            return self._h1
        def named_modules(self):
            return [("layer0", {}), ("layer1", {})]
        def get_activation(self, name: str) -> List[float]:
            return (self._h0 if name == "layer0" else self._h1).tolist()

    TextMLP       = lambda **kw: _Stub(32,  64, 3, seed=1)
    TimeSeriesCNN = lambda **kw: _Stub(64,  32, 3, seed=2)
    AudioMLP      = lambda **kw: _Stub(33,  48, 3, seed=3)
    VisualCNN     = lambda **kw: _Stub(1024, 64, 3, seed=4)

    def _collect_batch(model, data_list, collector) -> Dict[str, List[List[float]]]:
        training: Dict[str, List[List[float]]] = {}
        for x in data_list:
            model.forward(x)
            for name, _ in model.named_modules():
                training.setdefault(name, []).append(model.get_activation(name))
        return training

    print("Numpy stub models defined.")
""")

# ── Modality 1: Text ─────────────────────────────────────────────────────────
md("## Modality 1: Text / LLM-style\n2-layer MLP on synthetic token embeddings. Concepts: noun-type, verb-type, function-word directions.")
code("""
print("=" * 60); print("MODALITY 1: Text / LLM-style"); print("=" * 60)

text_model = TextMLP()
text_coll  = LayerHookCollector(text_model if TORCH_AVAILABLE else None)

if TORCH_AVAILABLE:
    _x0 = torch.tensor(make_text_embeddings()[np.newaxis], dtype=torch.float32)
    with torch.no_grad(): text_model(_x0)
    print("Layers:", list(text_coll.collected.keys()))
    _batch = [torch.tensor(make_text_embeddings(seed=s)[np.newaxis], dtype=torch.float32)
              for s in range(6)]
    text_train = _collect_batch(text_model, _batch, text_coll)
    text_coll.clear()
    with torch.no_grad(): text_model(_batch[0])
else:
    _emb = make_text_embeddings().ravel()
    text_model.forward(_emb)
    for _n, _ in text_model.named_modules():
        text_coll.inject(_n, text_model.get_activation(_n))
    text_train = _collect_batch(text_model, [make_text_embeddings(seed=s).ravel() for s in range(4)], text_coll)

# ── Concept prototypes: known semantic ground truth ───────────────────────────
# Run each named embedding prototype through the model and record which layer
# activations it produces.  The SemanticLabeler will then discover which SAE
# latents (or activation dimensions) fire most for each concept.
_text_proto_embeddings = {
    "noun-word direction":   np.array([1.,0.,0.]+[0.]*29, dtype=np.float32),
    "verb-action direction": np.array([0.,1.,0.]+[0.]*29, dtype=np.float32),
    "function-word circuit": np.array([0.,0.,1.]+[0.]*29, dtype=np.float32),
}
text_prototypes: Dict[str, Dict[str, List[float]]] = {}
_main_text_act = {k: list(v) for k, v in text_coll.collected.items()}
for _concept, _proto_emb in _text_proto_embeddings.items():
    text_coll.clear()
    if TORCH_AVAILABLE:
        with torch.no_grad():
            text_model(torch.tensor(_proto_emb[np.newaxis, np.newaxis, :], dtype=torch.float32))
    else:
        text_model.forward(_proto_emb)
        for _n, _ in text_model.named_modules():
            text_coll.inject(_n, text_model.get_activation(_n))
    for _lname, _lact in text_coll.collected.items():
        text_prototypes.setdefault(_lname, {})[_concept] = list(_lact)
text_coll.collected.clear()
text_coll.collected.update(_main_text_act)

text_tsp   = UniversalSemanticTransposer("text-mlp", version="1.0.0", top_k=8)
text_feats = text_tsp.run(text_coll, text_train, sae_epochs=50,
                          concept_prototypes=text_prototypes)

for layer, feats in text_feats.items():
    f0 = feats[0]
    print(f"  {layer}: {len(feats)} feats  top={f0.score:.4f} ({f0.label or '(unlabeled)'})")
    for f in feats[:4]:
        lbl = f.label or "(unlabeled)"
        print(f"    [{f.source_index:3d}] score={f.score:.4f}  → {lbl}")
print(f"Cross-layer links: {text_tsp.cross_layer_links}")

text_sc = text_tsp.governance_scorecard()
print(f"Governance (stub-50pct ablation — not real interventions):")
print(f"  faith={text_sc.faithfulness_score:.4f}  stab={text_sc.stability_score:.4f}  pass={text_sc.passes_thresholds}")
""")

# ── Modality 2: Time Series ───────────────────────────────────────────────────
md("## Modality 2: Time Series (1D CNN)\n2-layer Conv1d on sine waves. conv1 extracts waveform features; conv2 builds frequency-selective detectors.")
code("""
print("=" * 60); print("MODALITY 2: Time Series (1D CNN)"); print("=" * 60)

ts_model = TimeSeriesCNN()
ts_coll  = LayerHookCollector(ts_model if TORCH_AVAILABLE else None)

_ts_sigs, _ts_lbls = make_time_series()
print(f"Data: {_ts_sigs.shape}  labels: {list(set(_ts_lbls))}")

if TORCH_AVAILABLE:
    _x0 = torch.tensor(_ts_sigs[0:1], dtype=torch.float32)
    with torch.no_grad(): ts_model(_x0)
    print("Layers:", list(ts_coll.collected.keys()))
    _batch = [torch.tensor(_ts_sigs[i:i+1], dtype=torch.float32) for i in range(len(_ts_sigs))]
    ts_train = _collect_batch(ts_model, _batch, ts_coll)
    ts_coll.clear()
    with torch.no_grad(): ts_model(_batch[0])
else:
    ts_model.forward(_ts_sigs[0])
    for _n, _ in ts_model.named_modules():
        ts_coll.inject(_n, ts_model.get_activation(_n))
    ts_train = _collect_batch(ts_model, [_ts_sigs[i] for i in range(len(_ts_sigs))], ts_coll)

# ── Concept prototypes: pure sine waves at each target frequency ──────────────
_t64 = np.linspace(0, 1.0, 64, dtype=np.float32)
_ts_proto_signals = {
    "1Hz slow oscillation":  np.sin(2*np.pi*1.0*_t64),
    "5Hz mid-range rhythm":  np.sin(2*np.pi*5.0*_t64),
    "20Hz fast oscillation": np.sin(2*np.pi*20.0*_t64),
}
ts_prototypes: Dict[str, Dict[str, List[float]]] = {}
_main_ts_act = {k: list(v) for k, v in ts_coll.collected.items()}
for _concept, _sig in _ts_proto_signals.items():
    ts_coll.clear()
    if TORCH_AVAILABLE:
        with torch.no_grad():
            ts_model(torch.tensor(_sig[np.newaxis, np.newaxis, :], dtype=torch.float32))
    else:
        ts_model.forward(_sig)
        for _n, _ in ts_model.named_modules():
            ts_coll.inject(_n, ts_model.get_activation(_n))
    for _lname, _lact in ts_coll.collected.items():
        ts_prototypes.setdefault(_lname, {})[_concept] = list(_lact)
ts_coll.collected.clear()
ts_coll.collected.update(_main_ts_act)

ts_tsp   = UniversalSemanticTransposer("timeseries-cnn", version="1.0.0", top_k=8)
ts_feats = ts_tsp.run(ts_coll, ts_train, sae_epochs=50,
                      concept_prototypes=ts_prototypes)

for layer, feats in ts_feats.items():
    f0 = feats[0]
    print(f"  {layer}: {len(feats)} feats  top={f0.score:.4f} ({f0.label or '(unlabeled)'})")
    for f in feats[:4]:
        lbl = f.label or "(unlabeled)"
        print(f"    [{f.source_index:3d}] score={f.score:.4f}  → {lbl}")

ts_sc = ts_tsp.governance_scorecard()
print(f"Governance (stub-50pct ablation — not real interventions):")
print(f"  faith={ts_sc.faithfulness_score:.4f}  stab={ts_sc.stability_score:.4f}  pass={ts_sc.passes_thresholds}")
""")

# ── Modality 3: Audio ─────────────────────────────────────────────────────────
md("## Modality 3: Audio Spectrum (FFT MLP)\nSame sine waves as time series but represented as FFT magnitudes. The '5 Hz' concept should align cross-modally.")
code("""
print("=" * 60); print("MODALITY 3: Audio Spectrum (FFT MLP)"); print("=" * 60)

audio_model = AudioMLP()
audio_coll  = LayerHookCollector(audio_model if TORCH_AVAILABLE else None)

_audio_spec, _audio_lbls = make_audio_spectra()
print(f"Data: {_audio_spec.shape}  labels: {list(set(_audio_lbls))}")

if TORCH_AVAILABLE:
    _x0 = torch.tensor(_audio_spec[0:1], dtype=torch.float32)
    with torch.no_grad(): audio_model(_x0)
    print("Layers:", list(audio_coll.collected.keys()))
    _batch = [torch.tensor(_audio_spec[i:i+1], dtype=torch.float32) for i in range(len(_audio_spec))]
    audio_train = _collect_batch(audio_model, _batch, audio_coll)
    audio_coll.clear()
    with torch.no_grad(): audio_model(_batch[0])
else:
    audio_model.forward(_audio_spec[0])
    for _n, _ in audio_model.named_modules():
        audio_coll.inject(_n, audio_model.get_activation(_n))
    audio_train = _collect_batch(audio_model, list(_audio_spec), audio_coll)

# ── Concept prototypes: FFT of pure tones (same signal, spectral representation)
_t64_a = np.linspace(0, 1.0, 64, dtype=np.float32)
_audio_proto_tones = {
    "1Hz spectral peak":      np.sin(2*np.pi*1.0*_t64_a),
    "5Hz spectral resonance": np.sin(2*np.pi*5.0*_t64_a),
    "20Hz high-freq detector":np.sin(2*np.pi*20.0*_t64_a),
}
audio_prototypes: Dict[str, Dict[str, List[float]]] = {}
_main_audio_act = {k: list(v) for k, v in audio_coll.collected.items()}
for _concept, _sig in _audio_proto_tones.items():
    _fft = np.abs(np.fft.rfft(_sig)).astype(np.float32)
    _fft = _fft / (_fft.max() if _fft.max() > 1e-8 else 1.0)
    audio_coll.clear()
    if TORCH_AVAILABLE:
        with torch.no_grad():
            audio_model(torch.tensor(_fft[np.newaxis], dtype=torch.float32))
    else:
        audio_model.forward(_fft)
        for _n, _ in audio_model.named_modules():
            audio_coll.inject(_n, audio_model.get_activation(_n))
    for _lname, _lact in audio_coll.collected.items():
        audio_prototypes.setdefault(_lname, {})[_concept] = list(_lact)
audio_coll.collected.clear()
audio_coll.collected.update(_main_audio_act)

audio_tsp   = UniversalSemanticTransposer("audio-mlp", version="1.0.0", top_k=8)
audio_feats = audio_tsp.run(audio_coll, audio_train, sae_epochs=50,
                            concept_prototypes=audio_prototypes)

for layer, feats in audio_feats.items():
    f0 = feats[0]
    print(f"  {layer}: {len(feats)} feats  top={f0.score:.4f} ({f0.label or '(unlabeled)'})")
    for f in feats[:4]:
        lbl = f.label or "(unlabeled)"
        print(f"    [{f.source_index:3d}] score={f.score:.4f}  → {lbl}")

audio_sc = audio_tsp.governance_scorecard()
print(f"Governance (stub-50pct ablation — not real interventions):")
print(f"  faith={audio_sc.faithfulness_score:.4f}  stab={audio_sc.stability_score:.4f}  pass={audio_sc.passes_thresholds}")
""")

# ── Modality 4: Visual ────────────────────────────────────────────────────────
md("## Modality 4: Visual / Video Frames (2D CNN)\n2-layer Conv2d on 32×32 synthetic patterns. conv1 extracts local filters; conv2 builds spatial-frequency detectors.")
code("""
print("=" * 60); print("MODALITY 4: Visual / Video Frames (2D CNN)"); print("=" * 60)

visual_model = VisualCNN()
visual_coll  = LayerHookCollector(visual_model if TORCH_AVAILABLE else None)

_vis_frames, _vis_lbls = make_visual_frames()
print(f"Data: {_vis_frames.shape}  labels: {list(set(_vis_lbls))}")

if TORCH_AVAILABLE:
    _x0 = torch.tensor(_vis_frames[0:1], dtype=torch.float32)
    with torch.no_grad(): visual_model(_x0)
    print("Layers:", list(visual_coll.collected.keys()))
    _batch = [torch.tensor(_vis_frames[i:i+1], dtype=torch.float32) for i in range(len(_vis_frames))]
    visual_train = _collect_batch(visual_model, _batch, visual_coll)
    visual_coll.clear()
    with torch.no_grad(): visual_model(_batch[0])
else:
    visual_model.forward(_vis_frames[0])
    for _n, _ in visual_model.named_modules():
        visual_coll.inject(_n, visual_model.get_activation(_n))
    visual_train = _collect_batch(visual_model, list(_vis_frames), visual_coll)

# ── Concept prototypes: canonical spatial patterns ────────────────────────────
_H, _W = 32, 32
_vy, _vx = np.meshgrid(np.linspace(-1,1,_H), np.linspace(-1,1,_W), indexing="ij")
_vis_proto_frames = {
    "horizontal stripe detector": np.sin(2*np.pi*3*_vy).astype(np.float32),
    "concentric circle circuit":  np.sin(2*np.pi*3*np.sqrt(_vx**2+_vy**2)).astype(np.float32),
    "random noise detector":      np.random.default_rng(42).standard_normal((_H,_W)).astype(np.float32),
}
visual_prototypes: Dict[str, Dict[str, List[float]]] = {}
_main_vis_act = {k: list(v) for k, v in visual_coll.collected.items()}
for _concept, _frame in _vis_proto_frames.items():
    visual_coll.clear()
    if TORCH_AVAILABLE:
        with torch.no_grad():
            visual_model(torch.tensor(_frame[np.newaxis, np.newaxis], dtype=torch.float32))
    else:
        visual_model.forward(_frame)
        for _n, _ in visual_model.named_modules():
            visual_coll.inject(_n, visual_model.get_activation(_n))
    for _lname, _lact in visual_coll.collected.items():
        visual_prototypes.setdefault(_lname, {})[_concept] = list(_lact)
visual_coll.collected.clear()
visual_coll.collected.update(_main_vis_act)

visual_tsp   = UniversalSemanticTransposer("visual-cnn", version="1.0.0", top_k=8)
visual_feats = visual_tsp.run(visual_coll, visual_train, sae_epochs=50,
                              concept_prototypes=visual_prototypes)

for layer, feats in visual_feats.items():
    f0 = feats[0]
    print(f"  {layer}: {len(feats)} feats  top={f0.score:.4f} ({f0.label or '(unlabeled)'})")
    for f in feats[:4]:
        lbl = f.label or "(unlabeled)"
        print(f"    [{f.source_index:3d}] score={f.score:.4f}  → {lbl}")

visual_sc = visual_tsp.governance_scorecard()
print(f"Governance (stub-50pct ablation — not real interventions):")
print(f"  faith={visual_sc.faithfulness_score:.4f}  stab={visual_sc.stability_score:.4f}  pass={visual_sc.passes_thresholds}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# SAE Quality Verification
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## SAE Quality Verification

Before trusting any feature labels, we verify that each SAE **actually learned**
to represent its layer's activations:

- **Reconstruction MSE** — decoder output vs. original activation; low → SAE is faithful
- **Sparsity** — mean active latents per input; should be much less than dict_size
- **Loss convergence** — final epoch loss should be lower than epoch-1 loss

If any of these fail, subsequent feature labels are statistically meaningless.
""")
code("""
print("=" * 60)
print("SAE QUALITY VERIFICATION")
print("=" * 60)

all_tsps_check = {"text": text_tsp, "timeseries": ts_tsp, "audio": audio_tsp, "visual": visual_tsp}

for mod, tsp in all_tsps_check.items():
    print(f"\\n  {mod}:")
    for layer, sae in tsp.layer_saes.items():
        if not sae._trained:
            print(f"    {layer}: NOT TRAINED — labels are meaningless"); continue

        # Re-collect training activations for this layer
        train_vecs = []
        # Use the per-modality training dict stored in the globals
        _tdict = {"text": text_train, "timeseries": ts_train,
                  "audio": audio_train, "visual": visual_train}[mod]
        if layer in _tdict:
            train_vecs = _tdict[layer]

        if not train_vecs:
            print(f"    {layer}: no training cache available"); continue

        matrix = np.array(train_vecs, dtype=np.float32)

        if TORCH_AVAILABLE:
            import torch, torch.nn as nn
            X = torch.tensor(matrix, dtype=torch.float32)
            with torch.no_grad():
                latent = torch.relu(sae._enc(X))
                recon  = sae._dec(latent)
            mse      = float(((recon - X) ** 2).mean())
            sparsity = float((latent > 1e-6).float().mean())  # fraction of active latents
            n_active = float((latent > 1e-6).float().sum(dim=1).mean())  # mean active per sample
            losses   = sae._losses
            converged = (len(losses) >= 2 and losses[-1] < losses[0]) if losses else False
            print(f"    {layer}:  recon_mse={mse:.5f}  "
                  f"active_frac={sparsity:.3f}  mean_active={n_active:.1f}/{sae.dict_size}  "
                  f"converged={'YES' if converged else 'NO (check lr/epochs)'}")
            if mse > 1.0:
                print(f"      WARNING: high reconstruction MSE — features may not be meaningful")
            if n_active > sae.dict_size * 0.5:
                print(f"      WARNING: SAE is not sparse (>50% latents active) — increase L1 coeff")
        else:
            print(f"    {layer}: torch not available — magnitude top-k, no SAE quality check")

print("\\nVerification complete.")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL — Universal Kernel Assembly
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Universal Kernel Assembly

1. Collect SAE feature vectors from every (modality, layer) pair
2. Align all modalities to a shared space via Orthogonal Procrustes
3. Find cross-modal concept correspondences (cosine sim > 0.6)
4. Save a merged `KernelTemplate("2.0.0")` spanning all modalities
""")

code("""
all_tsps  = {"text": text_tsp, "timeseries": ts_tsp, "audio": audio_tsp, "visual": visual_tsp}
all_colls = {"text": text_coll,"timeseries": ts_coll,"audio": audio_coll,"visual": visual_coll}

# Build per-modality feature matrices (one row per feature, SAE latent vector reused per layer)
mod_mats: Dict[str, np.ndarray]  = {}
mod_ids:  Dict[str, List[str]]   = {}
mod_feats: Dict[str, List[Feature]] = {}

for mod, tsp in all_tsps.items():
    coll = all_colls[mod]
    vecs, ids, feats = [], [], []
    for layer, fs in tsp.layer_features.items():
        if layer not in coll.collected:
            continue
        fvec = tsp.layer_saes[layer].feature_vector(coll.collected[layer])
        for f in fs:
            vecs.append(fvec); ids.append(f.id); feats.append(f)
    if vecs:
        # Pad all vectors to the maximum dimension in this modality
        max_d = max(len(v) for v in vecs)
        padded = [np.concatenate([v, np.zeros(max_d - len(v), dtype=np.float32)])
                  if len(v) < max_d else v for v in vecs]
        mod_mats[mod]  = np.array(padded, dtype=np.float32)
        mod_ids[mod]   = ids
        mod_feats[mod] = feats
        print(f"  {mod:12s}: feature matrix {mod_mats[mod].shape}")

# Align all modalities to the reference (largest matrix)
ref = max(mod_mats, key=lambda m: mod_mats[m].shape[0])
print(f"\\nReference modality: {ref}")

aligned:  Dict[str, np.ndarray]                = {ref: mod_mats[ref]}
all_corr: List[Tuple[str, str, str, str, float]] = []

for mod, mat in mod_mats.items():
    if mod == ref:
        continue
    try:
        al = CrossModalAligner().fit_transform(mat, mod_mats[ref])
        aligned[mod] = al
        corrs = CrossModalAligner().fit(mat, mod_mats[ref]).find_correspondences(
            al, mod_mats[ref], mod_ids[mod], mod_ids[ref], threshold=0.6)
        for fa, fb, sim in corrs:
            all_corr.append((mod, ref, fa, fb, sim))
        print(f"  {mod} -> {ref}: {len(corrs)} correspondences")
    except Exception as e:
        print(f"  {mod}: alignment failed ({e})")
        aligned[mod] = mat

# Direct timeseries <-> audio (same signal, different representation)
ts_audio_corr = []
if "timeseries" in mod_mats and "audio" in mod_mats:
    al_ts = CrossModalAligner().fit_transform(mod_mats["timeseries"], mod_mats["audio"])
    ts_audio_corr = CrossModalAligner().fit(
        mod_mats["timeseries"], mod_mats["audio"]
    ).find_correspondences(al_ts, mod_mats["audio"],
                           mod_ids["timeseries"], mod_ids["audio"], threshold=0.5)
    print(f"\\nDirect timeseries <-> audio: {len(ts_audio_corr)} correspondences")
    for fa, fb, sim in ts_audio_corr[:3]:
        print(f"  TS:{fa}  <->  AUDIO:{fb}  cos={sim:.4f}")

# Save universal kernel
universal_features = []
for mod, feats in mod_feats.items():
    for f in feats:
        universal_features.append(Feature(
            id=f"{mod}:{f.id}", layer=f"{mod}:{f.layer}",
            score=f.score, source_index=f.source_index, label=f.label))

univ_kernel = KernelTemplate(
    kernel_id="universal-semantic-transposer", version="2.0.0",
    model_name="multi-modal", layer="all-layers",
    features=universal_features,
    metadata={
        "created_at_utc": utc_timestamp(),
        "modalities": ",".join(all_tsps.keys()),
        "total_features": str(len(universal_features)),
        "cross_modal_correspondences": str(len(all_corr)),
        "alignment_method": "orthogonal_procrustes_svd",
        "torch_backend": str(TORCH_AVAILABLE),
    })
univ_path = library.save(univ_kernel)
print(f"\\nUniversal kernel saved: {univ_path}")
print(f"Total features: {len(universal_features)}   Cross-modal correspondences: {len(all_corr)}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
md("## Visualizations")

# Viz 1 — layer heatmap
code("""
# ── Viz 1: Per-layer feature score heatmap ────────────────────────────────────
_mods_ordered = [("text", text_tsp, "Blues"), ("timeseries", ts_tsp, "Greens"),
                 ("audio", audio_tsp, "Oranges"), ("visual", visual_tsp, "Purples")]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("Per-Layer Feature Score Heatmap  (x = feature rank, y = layer, colour = score)", fontsize=13)

for ax, (mod, tsp, cmap) in zip(axes, _mods_ordered):
    layers = list(tsp.layer_features.keys())
    top_k  = tsp.top_k
    mat    = np.zeros((len(layers), top_k))
    for i, lyr in enumerate(layers):
        for j, f in enumerate(sorted(tsp.layer_features[lyr], key=lambda x: x.score, reverse=True)[:top_k]):
            mat[i, j] = f.score
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=max(mat.max(), 1e-6))
    ax.set_title(mod, fontsize=11)
    ax.set_xlabel("Feature rank")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([l[:14] for l in layers], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
_p = CACHE_DIR / "viz1_layer_heatmap.png"
plt.savefig(_p, dpi=100, bbox_inches="tight"); plt.show()
print(f"Saved {_p.name}")
""")

# Viz 2 — cross-layer flow
code("""
# ── Viz 2: Cross-layer concept flow (Sankey-style) ────────────────────────────
_focus_tsp  = max([text_tsp, ts_tsp, audio_tsp, visual_tsp],
                  key=lambda t: len(t.layer_features) + len(t.cross_layer_links))
_focus_name = {id(text_tsp):"text", id(ts_tsp):"timeseries",
               id(audio_tsp):"audio", id(visual_tsp):"visual"}[id(_focus_tsp)]
_layers = list(_focus_tsp.layer_features.keys())
_lx     = {n: i*2.5 for i, n in enumerate(_layers)}

fig, ax = plt.subplots(figsize=(max(6, len(_layers)*2.5), 5))
ax.set_title(f"Cross-layer Concept Flow — {_focus_name}", fontsize=12)

BH = 0.11
_cmap = plt.cm.Set1
for li, lyr in enumerate(_layers):
    feats = sorted(_focus_tsp.layer_features[lyr], key=lambda f: f.score, reverse=True)
    x = _lx[lyr]
    for rank, f in enumerate(feats[:8]):
        y = 1.0 - rank*(BH+0.02)
        ax.add_patch(plt.Rectangle((x-0.4, y-BH/2), 0.8, BH,
                                   color=_cmap((f.source_index%9)/9.), alpha=0.8))
        ax.text(x, y, f"[{f.source_index}]", ha="center", va="center",
                fontsize=6, color="white")

for la, lb, sim in _focus_tsp.cross_layer_links:
    if la in _lx and lb in _lx:
        ax.annotate("", xy=(_lx[lb]-0.4, 0.5), xytext=(_lx[la]+0.4, 0.5),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2., alpha=0.7))
        ax.text((_lx[la]+_lx[lb])/2, 0.57, f"cos={sim:.2f}",
                ha="center", fontsize=7, color="red")

ax.set_xlim(-0.9, max(_lx.values())+1.0); ax.set_ylim(-0.15, 1.35)
ax.set_xticks([_lx[n] for n in _layers])
ax.set_xticklabels([n[:12] for n in _layers], fontsize=8)
ax.set_yticks([]); ax.set_xlabel("Layer")
plt.tight_layout()
_p = CACHE_DIR / "viz2_crosslayer_flow.png"
plt.savefig(_p, dpi=100, bbox_inches="tight"); plt.show()
print(f"Saved {_p.name}")
""")

# Viz 3 — semantic canvas
code("""
# ── Viz 3: 2D Semantic Canvas (SemanticCanvas.to_layout + to_edges) ───────────
_canvas_feats = [f for fs in ts_tsp.layer_features.values() for f in fs]
_pts   = canvas.to_layout(_canvas_feats)
_edges = canvas.to_edges(_canvas_feats)

fig, ax = plt.subplots(figsize=(9, 6))
ax.set_title("2D Semantic Canvas — Time Series (SemanticCanvas.to_layout + to_edges)", fontsize=12)

for src_id, tgt_id, strength in _edges:
    s = next((p for p in _pts if p["id"]==src_id), None)
    t = next((p for p in _pts if p["id"]==tgt_id), None)
    if s and t:
        ax.plot([s["x"], t["x"]], [s["y"], t["y"]],
                color="cornflowerblue", alpha=min(float(strength), 0.6), lw=1.2)

xs = [p["x"] for p in _pts]; ys = [p["y"] for p in _pts]
sc = ax.scatter(xs, ys, c=[p["score"] for p in _pts], cmap="plasma",
                s=[40+p["score"]*120 for p in _pts],
                alpha=0.85, edgecolors="white", linewidths=0.8, vmin=0, vmax=1)
for p in sorted(_pts, key=lambda x: x["score"], reverse=True)[:5]:
    ax.annotate((p["label"] or p["id"])[:22], (p["x"], p["y"]),
                xytext=(5,5), textcoords="offset points", fontsize=6)
plt.colorbar(sc, ax=ax, label="Feature score", fraction=0.03)
ax.set_xlabel("Rank (highest score = right)"); ax.set_ylabel("Normalised score")
ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.05, 1.1)
plt.tight_layout()
_p = CACHE_DIR / "viz3_semantic_canvas.png"
plt.savefig(_p, dpi=100, bbox_inches="tight"); plt.show()
print(f"Saved {_p.name}")
""")

# Viz 4 — cross-modal PCA
code("""
# ── Viz 4: Cross-modal alignment — 2D PCA ─────────────────────────────────────
def _pca2(matrices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    d   = min(m.shape[1] for m in matrices.values())
    trunc = {k: v[:, :d] for k, v in matrices.items()}
    all_d = np.vstack(list(trunc.values()))
    c     = all_d - all_d.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(c, full_matrices=False)
        proj = c @ Vt[:2].T
    except Exception:
        rng  = np.random.default_rng(0)
        proj = c @ rng.standard_normal((d, 2)).astype(np.float32)
    result, idx = {}, 0
    for k, v in trunc.items():
        result[k] = proj[idx: idx+v.shape[0]]; idx += v.shape[0]
    return result

_pca_in  = {k: aligned.get(k, mod_mats[k]) for k in mod_mats}
_pca_out = _pca2(_pca_in)

_cols = {"text":"tomato","timeseries":"steelblue","audio":"darkorange","visual":"seagreen"}
_mrks = {"text":"o","timeseries":"s","audio":"^","visual":"D"}

fig, ax = plt.subplots(figsize=(9, 7))
ax.set_title("Cross-modal Alignment: 2D PCA — Orthogonal Procrustes shared space", fontsize=12)
handles = []
for mod, xy in _pca_out.items():
    ax.scatter(xy[:,0], xy[:,1], c=_cols[mod], marker=_mrks[mod],
               s=80, alpha=0.75, edgecolors="white", linewidths=0.5)
    cx, cy = xy[:,0].mean(), xy[:,1].mean()
    ax.text(cx, cy, mod, fontsize=9, fontweight="bold", color=_cols[mod], ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))
    handles.append(mpatches.Patch(color=_cols[mod], label=mod))

# draw ts<->audio correspondence lines
for fa, fb, sim in ts_audio_corr[:5]:
    if "timeseries" in _pca_out and "audio" in _pca_out:
        try:
            ia = mod_ids["timeseries"].index(fa); ib = mod_ids["audio"].index(fb)
            xa,ya = _pca_out["timeseries"][ia]; xb,yb = _pca_out["audio"][ib]
            ax.plot([xa,xb],[ya,yb],"k--",alpha=0.3,lw=0.9)
        except ValueError:
            pass

ax.legend(handles=handles, loc="upper right", fontsize=10)
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
plt.tight_layout()
_p = CACHE_DIR / "viz4_crossmodal_pca.png"
plt.savefig(_p, dpi=100, bbox_inches="tight"); plt.show()
print(f"Saved {_p.name}")
""")

# Viz 5 — governance table
code("""
# ── Viz 5: Governance scorecard table ─────────────────────────────────────────
_scores = {"text": text_sc, "timeseries": ts_sc, "audio": audio_sc, "visual": visual_sc}
_univ_sc = GovernanceScorecard(
    faithfulness_score=float(np.mean([s.faithfulness_score for s in _scores.values()])),
    stability_score=float(np.mean([s.stability_score for s in _scores.values()])),
    risk_flag="ok" if all(s.passes_thresholds for s in _scores.values()) else "low_faithfulness",
    passes_thresholds=all(s.passes_thresholds for s in _scores.values()),
    intervention_method="stub-50pct-aggregate",
)
_scores["universal"] = _univ_sc

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis("off")
ax.set_title("Governance Scorecard — All Modalities + Universal Kernel\\n"
             "\\u26a0 Intervention method: stub-50pct (synthetic ablation, not real causal intervention)",
             fontsize=11, pad=10)
_cols_h = ["Modality","Faithfulness","Stability","Risk Flag","Passes","Method"]
_rows, _colors = [], []
for mod, sc in _scores.items():
    _rows.append([mod.upper(), f"{sc.faithfulness_score:.4f}", f"{sc.stability_score:.4f}",
                  sc.risk_flag, "YES" if sc.passes_thresholds else "NO", sc.intervention_method])
    _colors.append(["#d5f5e3" if sc.passes_thresholds else "#fadbd8"]*len(_cols_h))

tbl = ax.table(cellText=_rows, colLabels=_cols_h, cellColours=_colors,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 1.8)
for j in range(len(_cols_h)):
    tbl[0,j].set_facecolor("#2c3e50"); tbl[0,j].set_text_props(color="white", fontweight="bold")
plt.tight_layout()
_p = CACHE_DIR / "viz5_governance.png"
plt.savefig(_p, dpi=100, bbox_inches="tight"); plt.show()

print("\\nGovernance summary:")
for mod, sc in _scores.items():
    print(f"  {mod:12s}: faith={sc.faithfulness_score:.4f}  stab={sc.stability_score:.4f}  "
          f"[{'PASS' if sc.passes_thresholds else 'FAIL'}]")
""")

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT A — Richer concept vocabulary
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Experiment A: Richer Concept Vocabulary

The original 3-prototype labelers (noun/verb/func, 1/5/20Hz, stripes/circles/noise)
leave many features unlabeled.  Here we extend each modality's vocabulary and
show how denser prototype coverage produces semantically richer circuits.

No retraining — we reuse the already-trained SAEs and just run more prototypes
through the encoder to discover more named detectors.
""")
code("""
print("=" * 60)
print("EXPERIMENT A: Richer Concept Vocabulary")
print("=" * 60)

_t64r = np.linspace(0, 1.0, 64, dtype=np.float32)

def _proto_by_layer(model, coll, prototypes, torch_fn=None, numpy_fn=None):
    \"\"\"Run named prototypes through model and collect per-layer activations.\"\"\"
    out: Dict[str, Dict[str, List[float]]] = {}
    for concept, proto in prototypes.items():
        coll.clear()
        if TORCH_AVAILABLE and torch_fn:
            with torch.no_grad():
                model(torch_fn(proto))
        else:
            numpy_fn(model, coll, proto)
        for lname, lact in coll.collected.items():
            out.setdefault(lname, {})[concept] = list(lact)
    return out

def _label_and_report(layer_saes, layer_features, proto_by_layer, title, keywords):
    print(f"\\n  {title}")
    for layer, sae in layer_saes.items():
        if layer not in proto_by_layer:
            continue
        lb = SemanticLabeler(sae, top_per_concept=2).fit(proto_by_layer[layer])
        relabeled = lb.label(layer_features[layer])
        cov = lb.coverage(relabeled)
        semantic = [(f.source_index, f.label) for f in relabeled
                    if f.label and any(k in f.label for k in keywords)]
        print(f"    {layer}: coverage={cov:.0%}  named={semantic[:4]}")

# ── Text: 7 grammatical roles ─────────────────────────────────────────────────
_rich_text = {
    "noun-word direction":   np.array([1.,0.,0.]+[0.]*29, dtype=np.float32),
    "verb-action direction": np.array([0.,1.,0.]+[0.]*29, dtype=np.float32),
    "function-word circuit": np.array([0.,0.,1.]+[0.]*29, dtype=np.float32),
    "adjective-descriptor":  np.array([0.7,0.,0.7]+[0.]*29, dtype=np.float32),
    "numeric-quantity":      np.array([0.,0.,0.,1.]+[0.]*28, dtype=np.float32),
    "proper-name circuit":   np.array([0.,0.5,0.5]+[0.]*29, dtype=np.float32),
    "negation-inversion":    np.array([-1.,0.,0.]+[0.]*29, dtype=np.float32),
}
_rtp_text = _proto_by_layer(
    text_model, text_coll, _rich_text,
    torch_fn=lambda e: torch.tensor(e[np.newaxis,np.newaxis,:], dtype=torch.float32),
    numpy_fn=lambda m,c,e: [c.inject(n, m.get_activation(n))
                             for n,_ in m.named_modules() if m.forward(e) is not None or True],
)
text_coll.collected.update(_main_text_act)
_label_and_report(text_tsp.layer_saes, text_tsp.layer_features, _rtp_text,
                  "Text (7 concepts)",
                  ["noun","verb","func","adj","numeric","proper","negat"])

# ── Time series: 8 frequencies + harmonic ─────────────────────────────────────
_rich_ts = {
    "0.5Hz ultra-slow wave":    np.sin(2*np.pi*0.5*_t64r).astype(np.float32),
    "1Hz slow oscillation":     np.sin(2*np.pi*1.0*_t64r).astype(np.float32),
    "2Hz sub-alpha rhythm":     np.sin(2*np.pi*2.0*_t64r).astype(np.float32),
    "5Hz mid-range rhythm":     np.sin(2*np.pi*5.0*_t64r).astype(np.float32),
    "10Hz alpha oscillation":   np.sin(2*np.pi*10.0*_t64r).astype(np.float32),
    "20Hz fast oscillation":    np.sin(2*np.pi*20.0*_t64r).astype(np.float32),
    "harmonic-chord [3+4.5Hz]": ((np.sin(2*np.pi*3.0*_t64r)+np.sin(2*np.pi*4.5*_t64r))/2).astype(np.float32),
    "dc-baseline":              np.ones(64, dtype=np.float32),
}
_rtp_ts = _proto_by_layer(
    ts_model, ts_coll, _rich_ts,
    torch_fn=lambda s: torch.tensor(s[np.newaxis,np.newaxis,:], dtype=torch.float32),
    numpy_fn=lambda m,c,s: [c.inject(n, m.get_activation(n))
                             for n,_ in m.named_modules() if m.forward(s) is not None or True],
)
ts_coll.collected.update(_main_ts_act)
_label_and_report(ts_tsp.layer_saes, ts_tsp.layer_features, _rtp_ts,
                  "Time series (8 concepts)",
                  ["Hz","harmonic","baseline"])

# ── Audio: FFT of the same extended signals ───────────────────────────────────
_rich_audio = {k: (lambda s: (lambda f: f/(f.max() if f.max()>1e-8 else 1.))(
    np.abs(np.fft.rfft(s)).astype(np.float32)))(v) for k,v in _rich_ts.items()}
_rtp_audio = _proto_by_layer(
    audio_model, audio_coll, _rich_audio,
    torch_fn=lambda f: torch.tensor(f[np.newaxis], dtype=torch.float32),
    numpy_fn=lambda m,c,f: [c.inject(n, m.get_activation(n))
                             for n,_ in m.named_modules() if m.forward(f) is not None or True],
)
audio_coll.collected.update(_main_audio_act)
_label_and_report(audio_tsp.layer_saes, audio_tsp.layer_features, _rtp_audio,
                  "Audio (8 concepts)",
                  ["Hz","harmonic","baseline","spectral"])

# ── Visual: 5 spatial patterns ────────────────────────────────────────────────
_H2, _W2 = 32, 32
_vy2, _vx2 = np.meshgrid(np.linspace(-1,1,_H2), np.linspace(-1,1,_W2), indexing="ij")
_rich_vis = {
    "horizontal stripe detector": np.sin(2*np.pi*3*_vy2).astype(np.float32),
    "diagonal stripe detector":   np.sin(2*np.pi*3*(_vx2+_vy2)/np.sqrt(2)).astype(np.float32),
    "concentric circle circuit":  np.sin(2*np.pi*3*np.sqrt(_vx2**2+_vy2**2)).astype(np.float32),
    "radial gradient detector":   (np.arctan2(_vy2,_vx2)/np.pi).astype(np.float32),
    "random noise detector":      np.random.default_rng(42).standard_normal((_H2,_W2)).astype(np.float32),
}
_rtp_vis = _proto_by_layer(
    visual_model, visual_coll, _rich_vis,
    torch_fn=lambda f: torch.tensor(f[np.newaxis,np.newaxis], dtype=torch.float32),
    numpy_fn=lambda m,c,f: [c.inject(n, m.get_activation(n))
                             for n,_ in m.named_modules() if m.forward(f) is not None or True],
)
visual_coll.collected.update(_main_vis_act)
_label_and_report(visual_tsp.layer_saes, visual_tsp.layer_features, _rtp_vis,
                  "Visual (5 concepts)",
                  ["stripe","circle","radial","noise","diagonal"])
""")

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT B — LLM-named concepts
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Experiment B: LLM-Named Concepts

`LLMConceptNamer` collects activation statistics for each SAE feature across
all labeled training examples, then asks Claude to synthesize a concept name.

This is the *unsupervised* version of `SemanticLabeler`: instead of requiring
the researcher to pre-define prototype inputs, Claude infers what the feature
*represents* purely from which labeled examples activate it most.

When `ANTHROPIC_API_KEY` is set, real Claude API calls are made.
Otherwise a rule-based concept-synthesis fallback runs, which still
demonstrates the structure: frequency range → oscillation name,
pattern category → spatial name, grammatical cluster → role name.
""")
code("""
import os

try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


class LLMConceptNamer:
    \"\"\"
    Names SAE features by asking Claude what concept they represent.

    For each feature, collects the top-activating labeled training examples and
    formats a prompt: "A neural feature fires most for: {description}. Name it."
    Falls back to rule-based concept synthesis when the API is unavailable.
    \"\"\"

    _FREQ_BANDS = [
        (0,  1.5,  "ultra-slow oscillation"),
        (1.5, 3.5, "slow oscillation"),
        (3.5, 7.5, "mid-range rhythm"),
        (7.5,15.,  "alpha oscillation"),
        (15., 99., "fast oscillation"),
    ]

    def __init__(self, model="claude-haiku-4-5-20251001"):
        self._model = model
        self._client = (
            _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, base_url=os.environ.get("ANTHROPIC_BASE_URL","https://api.anthropic.com"))
            if _ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY else None
        )

    def collect_stats(self, sae, labeled_training):
        \"\"\"labeled_training: [(label_str, activation_vec), ...].
        Returns Dict[source_index, [(label, score), ...]] sorted by score desc.\"\"\"
        stats: Dict[int, List[Tuple[str, float]]] = {}
        for label, act in labeled_training:
            latent = sae.feature_vector(act)
            for idx, score in enumerate(latent):
                if float(score) > 1e-4:
                    stats.setdefault(idx, []).append((label, float(score)))
        return {i: sorted(v, key=lambda x: x[1], reverse=True)
                for i, v in stats.items()}

    def name_features(self, sae, features, labeled_training):
        \"\"\"Return Dict[source_index -> concept_name] for each feature.\"\"\"
        stats = self.collect_stats(sae, labeled_training)
        names = {}
        for f in features:
            top = stats.get(f.source_index, [])[:4]
            if not top:
                continue
            desc = ", ".join(f"{lbl} ({sc:.2f})" for lbl, sc in top)
            names[f.source_index] = self._name(desc, top)
        return names

    def _name(self, desc, top_activations):
        if self._client:
            return self._call_api(desc)
        return self._rule_based(top_activations)

    def _call_api(self, desc):
        try:
            resp = self._client.messages.create(
                model=self._model, max_tokens=16,
                messages=[{"role": "user", "content":
                    f"A neural feature activates for: {desc}. "
                    f"Name this concept detector in 3-5 words (no explanation):"}])
            return resp.content[0].text.strip().strip('"').strip("'")
        except Exception as e:
            return f"api-error({str(e)[:20]})"

    def _rule_based(self, top_activations):
        \"\"\"Synthesize a concept name from common themes in top-activating labels.\"\"\"
        labels = [lbl for lbl, _ in top_activations]
        # Frequency detection
        freqs = []
        for lbl in labels:
            for part in lbl.split():
                try:
                    f = float(part.rstrip("Hz").rstrip("hz"))
                    freqs.append(f); break
                except ValueError:
                    pass
        if freqs:
            avg_f = sum(freqs) / len(freqs)
            for lo, hi, name in self._FREQ_BANDS:
                if lo <= avg_f < hi:
                    return name
        # Spatial pattern detection
        for kw, name in [("stripe","stripe-like filter"), ("circle","ring detector"),
                         ("radial","radial gradient"), ("noise","noise/texture"),
                         ("diagonal","diagonal edge filter")]:
            if any(kw in lbl.lower() for lbl in labels):
                return name
        # Grammatical role detection
        for kw, name in [("noun","nominal concept"), ("verb","action concept"),
                         ("func","function-word circuit"), ("adj","modifier circuit"),
                         ("negat","negation inversion"), ("proper","entity name")]:
            if any(kw in lbl.lower() for lbl in labels):
                return name
        # Generic fallback
        return labels[0][:30] if labels else "unknown concept"


namer = LLMConceptNamer()
api_mode = "Claude API" if namer._client else "rule-based synthesis"
print(f"LLMConceptNamer running in: {api_mode}")
print()

# ── Run on time series (most interpretable because ground truth is mathematical) ──
_ts_labeled_train = []
for i, (sig, lbl) in enumerate(zip(_ts_sigs, _ts_lbls)):
    for _lyr in ts_tsp.layer_saes:
        ts_coll.clear()
        if TORCH_AVAILABLE:
            with torch.no_grad():
                ts_model(torch.tensor(sig[np.newaxis], dtype=torch.float32))
        else:
            ts_model.forward(sig)
            for _n, _ in ts_model.named_modules():
                ts_coll.inject(_n, ts_model.get_activation(_n))
    for _lyr, _act in ts_coll.collected.items():
        _ts_labeled_train.append((lbl, _act))
    if i >= 5:
        break   # use first 6 examples for stats

ts_coll.collected.update(_main_ts_act)

print("Time Series — LLM concept names vs prototype labels:")
for layer, sae in ts_tsp.layer_saes.items():
    layer_train = [(lbl, act) for (lbl, act) in _ts_labeled_train
                   if len(act) == sae.input_dim]
    if not layer_train:
        continue
    llm_map = namer.name_features(sae, ts_tsp.layer_features[layer], layer_train)
    for f in ts_tsp.layer_features[layer][:4]:
        proto_lbl = f.label or "(unlabeled)"
        raw_llm   = llm_map.get(f.source_index)
        llm_lbl   = f"[{api_mode}] {raw_llm}" if raw_llm else "(not activated)"
        print(f"  {layer} idx[{f.source_index:3d}]  score={f.score:.3f}")
        print(f"    prototype label : {proto_lbl}")
        print(f"    LLM concept name: {llm_lbl}")

# ── Run on audio ───────────────────────────────────────────────────────────────
_audio_labeled_train = []
for i, (spec, lbl) in enumerate(zip(_audio_spec, _audio_lbls)):
    for _lyr in audio_tsp.layer_saes:
        audio_coll.clear()
        if TORCH_AVAILABLE:
            with torch.no_grad():
                audio_model(torch.tensor(spec[np.newaxis], dtype=torch.float32))
        else:
            audio_model.forward(spec)
            for _n, _ in audio_model.named_modules():
                audio_coll.inject(_n, audio_model.get_activation(_n))
    for _lyr, _act in audio_coll.collected.items():
        _audio_labeled_train.append((lbl, _act))
    if i >= 5:
        break
audio_coll.collected.update(_main_audio_act)

print("\\nAudio — LLM concept names vs prototype labels:")
for layer, sae in audio_tsp.layer_saes.items():
    layer_train = [(lbl, act) for (lbl, act) in _audio_labeled_train
                   if len(act) == sae.input_dim]
    if not layer_train:
        continue
    llm_map = namer.name_features(sae, audio_tsp.layer_features[layer], layer_train)
    for f in audio_tsp.layer_features[layer][:4]:
        proto_lbl = f.label or "(unlabeled)"
        raw_llm   = llm_map.get(f.source_index)
        llm_lbl   = f"[{api_mode}] {raw_llm}" if raw_llm else "(not activated)"
        print(f"  {layer} idx[{f.source_index:3d}]  score={f.score:.3f}")
        print(f"    prototype label : {proto_lbl}")
        print(f"    LLM concept name: {llm_lbl}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT C — Real text via GPT-2 (upgraded)
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Experiment C: Real Text Semantics via GPT-2

Extracts features from GPT-2 at three depths — early (`h.0`), mid (`h.6`), late (`h.11`) —
using four linguistically diverse prompt categories:

| Category | Example |
|---|---|
| **Color / perception** | "The sky is a vivid shade of blue" |
| **Action / motion** | "She sprinted quickly across the finish line" |
| **Abstract concept** | "Freedom and justice are fundamental values" |
| **Spatial / relation** | "The red ball is above the wooden table" |

`SemanticLabeler` uses sentence-level prototypes — each category sentence is treated
as a concept prototype. Features that fire most for color sentences get labeled
"color/perception", etc.  This demonstrates concept drift across layers:
early layers catch surface form, late layers catch meaning.
""")
code("""
if not NNSIGHT_AVAILABLE:
    print("Skipping GPT-2 experiment — install torch + nnsight to enable.")
else:
    _GPT2_LAYER_PATHS = ["transformer.h.0", "transformer.h.6", "transformer.h.11"]
    _GPT2_LAYER_NAMES = {"transformer.h.0": "early(h0)",
                         "transformer.h.6": "mid(h6)",
                         "transformer.h.11": "late(h11)"}

    # Four prompt categories with concept labels
    _gpt2_prompts = {
        "color-perception":   [
            "The sky is a vivid shade of blue",
            "Bright red roses bloomed in the garden",
            "She painted the wall a soft green",
        ],
        "action-motion":      [
            "She sprinted quickly across the finish line",
            "The dog jumped over the tall fence",
            "He carefully carried the heavy boxes upstairs",
        ],
        "abstract-concept":   [
            "Freedom and justice are fundamental values",
            "The concept of infinity fascinates mathematicians",
            "Truth is often more complex than it appears",
        ],
        "spatial-relation":   [
            "The red ball is above the wooden table",
            "Turn left at the intersection and go straight",
            "The keys were hidden beneath the old rug",
        ],
    }

    # Flatten to [(label, text), ...]
    _all_gpt2_prompts = [(lbl, txt)
                        for lbl, txts in _gpt2_prompts.items()
                        for txt in txts]

    try:
        _gpt2_extractor = NNSightExtractor("gpt2", device="cpu",
                                          cache_dir=str(CACHE_DIR / "gpt2_cache"))

        # ── Collect training activations (all prompts, all layers) ───────────
        gpt2_train: Dict[str, List[List[float]]] = {}
        gpt2_labeled_train: List[Tuple[str, str, List[float]]] = []  # (label, lyr, act)
        for lbl, txt in _all_gpt2_prompts:
            acts = _gpt2_extractor.extract_layers(txt, _GPT2_LAYER_PATHS)
            for lp, act in acts.items():
                name = _GPT2_LAYER_NAMES[lp]
                gpt2_train.setdefault(name, []).append(preprocessor.normalize(act))
                gpt2_labeled_train.append((lbl, name, act))
        print(f"GPT-2 training set: {len(_all_gpt2_prompts)} prompts × {len(_GPT2_LAYER_PATHS)} layers")

        # ── Build per-layer category prototypes ──────────────────────────────
        gpt2_prototypes: Dict[str, Dict[str, List[float]]] = {}
        for cat, prompts in _gpt2_prompts.items():
            # Average activation per category per layer
            cat_acts: Dict[str, List[np.ndarray]] = {}
            for txt in prompts:
                acts = _gpt2_extractor.extract_layers(txt, _GPT2_LAYER_PATHS)
                for lp, act in acts.items():
                    name = _GPT2_LAYER_NAMES[lp]
                    cat_acts.setdefault(name, []).append(np.array(act, dtype=np.float32))
            for lyr, vecs in cat_acts.items():
                mean_act = np.mean(np.stack(vecs), axis=0).tolist()
                gpt2_prototypes.setdefault(lyr, {})[cat] = mean_act

        # ── Inject one test activation and run transposer ────────────────────
        _test_prompt = "The color green reminds me of nature"
        _test_acts = _gpt2_extractor.extract_layers(_test_prompt, _GPT2_LAYER_PATHS)

        gpt2_coll = LayerHookCollector()
        for lp, act in _test_acts.items():
            gpt2_coll.inject(_GPT2_LAYER_NAMES[lp], act)

        gpt2_tsp = UniversalSemanticTransposer("gpt2", version="1.0.0", top_k=8)
        gpt2_feats = gpt2_tsp.run(gpt2_coll, gpt2_train, sae_epochs=30,
                                  concept_prototypes=gpt2_prototypes)

        print(f"\\nGPT-2 prompt: \\\"{_test_prompt}\\\"")
        print(f"Layers: {list(gpt2_feats.keys())}")
        print()
        for layer in ["early(h0)", "mid(h6)", "late(h11)"]:
            if layer not in gpt2_feats:
                continue
            feats = gpt2_feats[layer]
            print(f"  {layer}:")
            for f in feats[:5]:
                print(f"    [{f.source_index:4d}] {f.score:.4f}  → {f.label}")

        # ── Cross-layer concept drift ─────────────────────────────────────────
        print("\\nCross-layer concept drift (same prompt, different depths):")
        _categories = list(_gpt2_prompts.keys())
        for cat in _categories:
            counts = {}
            for layer, feats in gpt2_feats.items():
                n = sum(1 for f in feats if f.label == cat)
                counts[layer] = n
            bar = " | ".join(f"{l}: {n}" for l, n in counts.items())
            print(f"  {cat:20s} — {bar}")

        print(f"\\nCross-layer links: {len(gpt2_tsp.cross_layer_links)}")
        print(f"GPT-2 experiment complete.")

    except Exception as exc:
        print(f"GPT-2 experiment skipped: {exc}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# Optional GPT-2
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Optional: GPT-2 via nnsight

Demonstrates that the same `LayerHookCollector + PerLayerSAE` pipeline that worked
on 2-layer toy models generalises to a real 12-layer transformer.

Skipped when `NNSIGHT_AVAILABLE = False`.
Install: `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install nnsight`
""")
code("""
if not NNSIGHT_AVAILABLE:
    print("Skipping GPT-2 — install torch + nnsight to enable.")
else:
    PROMPT       = "The quick brown fox"
    LAYER_PATHS  = ["transformer.h.0", "transformer.h.6", "transformer.h.11"]
    PROMPTS_TRAIN = ["The quick brown fox", "The capital of France is", "def fibonacci(n):"]

    try:
        extractor = NNSightExtractor("gpt2", device="cpu", cache_dir=str(CACHE_DIR))

        gpt2_coll = LayerHookCollector()   # no model hook — inject from extractor
        acts = extractor.extract_layers(PROMPT, LAYER_PATHS)
        for lp, v in acts.items():
            gpt2_coll.inject(lp.replace("transformer.h.", "h"), v)
        print("GPT-2 layers:", list(gpt2_coll.collected.keys()))

        gpt2_train: Dict[str, List[List[float]]] = {}
        for p in PROMPTS_TRAIN:
            a = extractor.extract_layers(p, LAYER_PATHS)
            for lp, v in a.items():
                gpt2_train.setdefault(lp.replace("transformer.h.","h"), []).append(
                    preprocessor.normalize(v))

        gpt2_tsp   = UniversalSemanticTransposer("gpt2", version="1.0.0", top_k=8)
        gpt2_feats = gpt2_tsp.run(gpt2_coll, gpt2_train, sae_epochs=30)

        print(f"GPT-2 layers processed: {len(gpt2_feats)}")
        for layer, feats in gpt2_feats.items():
            print(f"  {layer}: top score={feats[0].score:.4f}  ({feats[0].label})")
        print(f"Cross-layer links: {len(gpt2_tsp.cross_layer_links)}")

    except Exception as exc:
        print(f"GPT-2 section skipped gracefully: {exc}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# Kernel library summary
# ─────────────────────────────────────────────────────────────────────────────
md("## Kernel Library Contents")
code("""
print("=" * 60); print("KERNEL LIBRARY CONTENTS"); print("=" * 60)
all_kernels = library.list_kernels()
print(f"Registered kernel IDs: {len(all_kernels)}\\n")
for kid, versions in sorted(all_kernels.items()):
    try:
        tmpl = library.load_latest(kid)
        print(f"  {kid}")
        print(f"    versions : {versions}")
        print(f"    features : {len(tmpl.features)}")
        print(f"    backend  : {tmpl.metadata.get('backend','?')}")
    except Exception as e:
        print(f"  {kid}: load error ({e})")
    print()
print(f"PNG outputs: {sorted(CACHE_DIR.glob('viz*.png'))}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# Final validation
# ─────────────────────────────────────────────────────────────────────────────
md("## Final Validation")
code("""
errors = []

# All modality transposers produced non-empty features
for mod, tsp in all_tsps.items():
    if not tsp.layer_features:
        errors.append(f"{mod}: no features")
    for lyr, feats in tsp.layer_features.items():
        if not feats:
            errors.append(f"{mod}/{lyr}: empty feature list")

# Universal kernel loads and has features
try:
    uk = library.load("universal-semantic-transposer", "2.0.0")
    assert uk.features, "universal kernel has no features"
except Exception as e:
    errors.append(f"universal kernel: {e}")

# SemanticCanvas works
try:
    _pts = canvas.to_layout(list(text_feats.values())[0])
    _edg = canvas.to_edges(list(text_feats.values())[0])
except Exception as e:
    errors.append(f"SemanticCanvas: {e}")

# Scorecard fields present and in range
for mod, sc in {"text":text_sc,"timeseries":ts_sc,"audio":audio_sc,"visual":visual_sc}.items():
    assert hasattr(sc,"faithfulness_score"), f"{mod} missing faithfulness_score"

# Semantic labels: at least one feature per modality should have a concept name
_concept_keywords = ["direction","circuit","oscillation","rhythm","peak","resonance",
                     "detector","stripe","circle","noise"]
for mod, tsp in all_tsps.items():
    all_labels = [f.label or "" for fs in tsp.layer_features.values() for f in fs]
    semantic_count = sum(1 for lbl in all_labels
                        if any(kw in lbl for kw in _concept_keywords))
    if semantic_count == 0:
        errors.append(f"{mod}: no semantic concept labels found (got: {all_labels[:3]})")

if errors:
    for e in errors: print(f"ERROR: {e}")
else:
    print("All validation checks passed.")
    print()
    print(f"  Modalities        : {len(all_tsps)}")
    print(f"  Universal features: {len(universal_features)}")
    print(f"  Cross-modal corr  : {len(all_corr)}")
    print(f"  Kernel artifacts  : {len(library.list_kernels())}")
    print(f"  Torch backend     : {TORCH_AVAILABLE}")
    print(f"  Cache             : {CACHE_DIR}")
    print()
    print("Sample semantic labels:")
    for mod, tsp in all_tsps.items():
        sample = [(f.source_index, f.label) for fs in tsp.layer_features.values()
                  for f in fs if f.label and any(kw in f.label for kw in _concept_keywords)]
        print(f"  {mod:12s}: {sample[:3]}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Summary

| Component | Novel contribution | HypoSpace hook |
|---|---|---|
| `LayerHookCollector` | Parallel attachment to any model at every layer | — |
| `PerLayerSAE` | Per-layer trainable sparse feature dictionary | `Feature`, `_magnitude_top_k` fallback |
| `UniversalSemanticTransposer` | Cross-layer cosine concept tracking + versioned kernel persistence | `SemanticInterpreter`, `KernelLibrary`, `FaithfulnessChecker` |
| `CrossModalAligner` | Orthogonal Procrustes cross-modal concept correspondence | feeds `KernelTemplate` |

### How to extend

- **Real SAE checkpoints**: replace `PerLayerSAE` encoder with `MatryoshkaBackend` from `data/sae_backend.py`
- **Real interventions**: pass features to `PyVeneInterventionRunner.run_interventions()` after attaching to a live model
- **More modalities**: any numpy array source → stub model → `LayerHookCollector.inject()` — `CrossModalAligner` handles arbitrary counts via a reference-modality pivot
- **Hierarchical features**: stack a second `PerLayerSAE` on top of the first to decompose latents into meta-features (MetaSAE pattern)
""")

# ─────────────────────────────────────────────────────────────────────────────
# Write the notebook
# ─────────────────────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path(__file__).parent / "universal_semantic_transposer.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written {len(cells)} cells → {out}")
print(f"File size: {out.stat().st_size // 1024} KB")

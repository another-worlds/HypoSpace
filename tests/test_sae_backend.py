from __future__ import annotations

"""Tests for data/sae_backend.py. Entire module skipped when torch is absent."""

import importlib
import warnings
from pathlib import Path

import pytest

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="torch is required for SAE backend tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_sae_checkpoint(tmp_path: Path, d_in: int = 4, dict_size: int = 8) -> str:
    """Save a random Linear encoder as encoder.pt in tmp_path."""
    import torch

    enc = torch.nn.Linear(d_in, dict_size, bias=True)
    torch.save(enc.state_dict(), tmp_path / "encoder.pt")
    return str(tmp_path)


# ---------------------------------------------------------------------------
# sae_available
# ---------------------------------------------------------------------------

def test_sae_available_returns_true() -> None:
    from data.sae_backend import sae_available
    assert sae_available() is True


# ---------------------------------------------------------------------------
# build_backend — None paths
# ---------------------------------------------------------------------------

def test_build_backend_no_path_returns_none() -> None:
    from data.sae_backend import build_backend
    assert build_backend("matryoshka", None, "cpu") is None


def test_build_backend_magnitude_returns_none() -> None:
    from data.sae_backend import build_backend
    assert build_backend("magnitude", "/any/path", "cpu") is None


def test_build_backend_unknown_backend_returns_none() -> None:
    from data.sae_backend import build_backend
    assert build_backend("unknown-backend", "/some/path", "cpu") is None


def test_build_backend_missing_checkpoint_returns_none_with_warning(tmp_path: Path) -> None:
    from data.sae_backend import build_backend
    nonexistent = str(tmp_path / "no_such_dir")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = build_backend("matryoshka", nonexistent, "cpu")
    assert result is None
    assert any("falling back to magnitude" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# SAEEncodeResult
# ---------------------------------------------------------------------------

def test_sae_encode_result_is_slotted() -> None:
    from data.sae_backend import SAEEncodeResult
    r = SAEEncodeResult(feature_indices=[0, 1], feature_scores=[0.5, 0.3])
    assert not hasattr(r, "__dict__")
    assert r.feature_indices == [0, 1]
    assert r.feature_scores == [0.5, 0.3]


# ---------------------------------------------------------------------------
# MatryoshkaBackend — using synthetic checkpoint
# ---------------------------------------------------------------------------

def test_matryoshka_backend_encode_result_type(tmp_path: Path) -> None:
    from data.sae_backend import MatryoshkaBackend, SAEEncodeResult
    ckpt = _make_tiny_sae_checkpoint(tmp_path, d_in=4, dict_size=8)
    backend = MatryoshkaBackend(sae_path=ckpt, device="cpu")
    result = backend._encode([0.1, 0.9, -0.7, 0.5])
    assert isinstance(result, SAEEncodeResult)
    assert isinstance(result.feature_indices, list)
    assert isinstance(result.feature_scores, list)
    assert len(result.feature_indices) == len(result.feature_scores)
    assert all(s >= 0.0 for s in result.feature_scores)


def test_matryoshka_backend_extract_returns_features(tmp_path: Path) -> None:
    from core.hierarchy import Feature
    from data.sae_backend import MatryoshkaBackend
    ckpt = _make_tiny_sae_checkpoint(tmp_path, d_in=4, dict_size=8)
    backend = MatryoshkaBackend(sae_path=ckpt, device="cpu")
    feats = backend.extract([0.1, 0.9, -0.7, 0.5], "layer_0", top_k=3)
    assert all(isinstance(f, Feature) for f in feats)
    assert all(":sae:" in f.id for f in feats)
    assert all(f.score >= 0.0 for f in feats)
    assert all(f.layer == "layer_0" for f in feats)


def test_matryoshka_backend_extract_top_k_respected(tmp_path: Path) -> None:
    from data.sae_backend import MatryoshkaBackend
    ckpt = _make_tiny_sae_checkpoint(tmp_path, d_in=4, dict_size=8)
    backend = MatryoshkaBackend(sae_path=ckpt, device="cpu")
    feats = backend.extract([0.5, 0.9, -0.7, 0.3], "layer_0", top_k=2)
    assert len(feats) <= 2


# ---------------------------------------------------------------------------
# End-to-end: API with sae_path=None stays on magnitude fallback
# ---------------------------------------------------------------------------

def test_api_with_sae_path_none_uses_magnitude_fallback(tmp_path: Path) -> None:
    from api import HypoSpaceAPI
    from core.config import DecoderConfig, RuntimeConfig
    config = DecoderConfig(
        backend="matryoshka",
        sae_path=None,
        top_k=2,
        runtime=RuntimeConfig(cache_dir=str(tmp_path)),
    )
    api = HypoSpaceAPI(config=config)
    result = api.decode_and_score("demo", "layer_0", [0.1, -0.9, 0.3])
    assert len(result.decode.features) == 2
    assert all(":feature:" in f.id for f in result.decode.features)

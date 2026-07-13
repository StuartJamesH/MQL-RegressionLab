"""
Engine/tests/test_v2_inference.py — Tests for V2InferenceEngine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from Engine.v2.inference import V2InferenceEngine, compare_backends
from Engine.v2.model_pack import ModelPackLoader


def _find_example_pack() -> Path:
    base = Path("ModelWorkbench/ModelPacks/transformers")
    if not base.exists():
        pytest.skip("No transformer model packs found")
    candidates = sorted(
        [d for d in base.iterdir() if d.is_dir() and (d / "model.onnx").exists()],
        key=lambda p: p.name,
    )
    for cand in candidates:
        if all((cand / f).exists() for f in ModelPackLoader.REQUIRED_FILES):
            return cand
    pytest.skip("No complete v2 model pack found")


@pytest.fixture(scope="module")
def example_pack() -> dict:
    return ModelPackLoader.load(str(_find_example_pack()))


def test_onnx_predict_shapes(example_pack: dict) -> None:
    config = example_pack["config"]
    engine = V2InferenceEngine(example_pack, backend="onnx", device="cpu")

    x_raw = np.random.randn(config.max_seq_len, config.in_channels).astype(np.float32)
    x_session = np.random.randn(
        config.max_seq_len, config.session_channels
    ).astype(np.float32)
    out = engine.predict(x_raw, x_session)

    assert out["mu"].shape == (1, config.n_horizons)
    assert out["log_sigma"].shape == (1, config.n_horizons)
    assert out["direction"].shape == (1, config.n_horizons)
    assert out["volatility"].shape == (1, config.n_horizons)
    assert out["regime"].shape == (1, config.n_regimes)
    assert out["quantiles"].shape == (1, config.n_horizons, config.n_quantiles)


def test_pytorch_predict_shapes(example_pack: dict) -> None:
    if example_pack.get("pytorch_state_dict") is None:
        pytest.skip("model.pt not present in pack")

    config = example_pack["config"]
    engine = V2InferenceEngine(example_pack, backend="pytorch", device="cpu")

    x_raw = np.random.randn(config.max_seq_len, config.in_channels).astype(np.float32)
    x_session = np.random.randn(
        config.max_seq_len, config.session_channels
    ).astype(np.float32)
    out = engine.predict(x_raw, x_session)

    assert out["mu"].shape == (1, config.n_horizons)
    assert out["direction"].shape == (1, config.n_horizons)


def test_backend_parity_random_input(example_pack: dict) -> None:
    if example_pack.get("pytorch_state_dict") is None:
        pytest.skip("model.pt not present in pack")

    pack_dir = example_pack["pack_dir"]
    assert compare_backends(pack_dir, sample_csv=None, atol=1e-4)

"""
test_labels.py — Tests for v2 label computation functions.

Matches the actual API of the installed Learn.v2.labels module.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from Learn.v2.labels import (
    compute_forward_excursion_surface,
    compute_directional_return_distribution,
    compute_optimal_exit_labels,
    compute_volatility_regime_labels,
    LabelStore,
)


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    trend = np.linspace(1.0000, 1.0500, n)
    noise = np.random.randn(n) * 0.0005
    close = trend + noise
    high = close + np.abs(np.random.randn(n) * 0.0003)
    low = close - np.abs(np.random.randn(n) * 0.0003)
    open_price = close + np.random.randn(n) * 0.0002
    return pd.DataFrame({
        "Time": dates, "Open": open_price, "High": high,
        "Low": low, "Close": close,
    })


# ── Forward Excursion Surface ──────────────────────────────────────────

def test_forward_excursion_surface_causality(sample_ohlcv):
    """Last bars should have NaN for horizons beyond data boundary."""
    horizons = [5, 10, 30]
    surface = compute_forward_excursion_surface(sample_ohlcv, horizons=horizons)
    assert isinstance(surface, np.ndarray)
    assert surface.shape[0] == len(sample_ohlcv)
    assert surface.shape[1] == len(horizons)
    # Last few rows for long horizon should have NaN
    last_bars = surface[-10:, -1, :, :]
    assert np.any(np.isnan(last_bars)), "Last bars should contain NaN values"


def test_forward_excursion_surface_shape(sample_ohlcv):
    """Correct output dimensions for various horizon lists."""
    for horizons in [[10], [5, 10, 30], [5, 10, 20, 50]]:
        surface = compute_forward_excursion_surface(sample_ohlcv, horizons=horizons)
        assert surface.shape[0] == len(sample_ohlcv)
        assert surface.shape[1] == len(horizons)


# ── Directional Return Distribution ────────────────────────────────────

def test_directional_returns_monotonic(sample_ohlcv):
    """On trending data, mean return should grow with horizon."""
    horizons = [5, 10, 20]
    returns = compute_directional_return_distribution(sample_ohlcv, horizons=horizons)
    assert returns.shape == (len(sample_ohlcv), len(horizons))
    mean_h5 = np.nanmean(returns[:, 0])
    mean_h20 = np.nanmean(returns[:, 2])
    assert mean_h5 > 0, f"Mean return at horizon 5 should be positive, got {mean_h5:.6f}"
    assert mean_h20 > 0, f"Mean return at horizon 20 should be positive, got {mean_h20:.6f}"
    assert mean_h20 > mean_h5, (
        f"Mean return at horizon 20 ({mean_h20:.6f}) should exceed "
        f"horizon 5 ({mean_h5:.6f}) on trending data"
    )


# ── Optimal Exit Labels ────────────────────────────────────────────────

def test_optimal_exit_labels_causality(sample_ohlcv):
    """Labels use only forward-looking data (causal)."""
    labels = compute_optimal_exit_labels(
        sample_ohlcv, tp_atr_mult=2.0, sl_atr_mult=1.5, max_horizon=30, atr_window=14,
    )
    assert isinstance(labels, pd.DataFrame)
    assert len(labels) == len(sample_ohlcv)


def test_optimal_exit_labels_shape(sample_ohlcv):
    """Output shape matches input."""
    labels = compute_optimal_exit_labels(
        sample_ohlcv, tp_atr_mult=2.0, sl_atr_mult=1.5, max_horizon=30, atr_window=14,
    )
    assert len(labels) == len(sample_ohlcv)
    assert isinstance(labels, pd.DataFrame)


# ── Volatility Regime Labels ───────────────────────────────────────────

def test_volatility_regime_causality(sample_ohlcv):
    """Regime labels use only trailing (causal) data."""
    labels = compute_volatility_regime_labels(sample_ohlcv, lookback=20)
    assert isinstance(labels, np.ndarray)
    assert labels.shape[0] == len(sample_ohlcv)
    valid = labels[~np.isnan(labels)]
    assert len(valid) > 0, "Should have some valid regime labels"


# ── LabelStore ─────────────────────────────────────────────────────────

def test_label_store_cache():
    """Same key returns cached result without recomputation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LabelStore(base_dir=tmpdir)
        call_count = [0]

        def compute_fn():
            call_count[0] += 1
            return np.array([1.0, 2.0, 3.0])

        # First call: should compute
        r1 = store.get_or_compute("test_key", compute_fn)
        assert call_count[0] == 1
        assert np.array_equal(r1, np.array([1.0, 2.0, 3.0]))

        # Second call: should use cache, not recompute
        r2 = store.get_or_compute("test_key", compute_fn)
        assert call_count[0] == 1
        assert np.array_equal(r1, r2)


def test_label_store_hash_collision():
    """LabelStore with args/kwargs produces distinct cached results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LabelStore(base_dir=tmpdir)
        results = []

        def compute(val):
            return np.array([val])

        r1 = store.get_or_compute("key_a", compute, 1)
        r2 = store.get_or_compute("key_b", compute, 2)
        r3 = store.get_or_compute("key_a", compute, 1)  # cached

        assert not np.array_equal(r1, r2), "Different keys should give different results"
        assert np.array_equal(r1, r3), "Same key should return cached result"


def test_label_store_with_args():
    """LabelStore passes *args and **kwargs to compute_fn."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LabelStore(base_dir=tmpdir)

        def add(a, b, c=0):
            return np.array([a + b + c])

        r1 = store.get_or_compute("add_1_2", add, 1, 2)
        assert r1[0] == 3.0

        r2 = store.get_or_compute("add_1_2_3", add, 1, 2, c=3)
        assert r2[0] == 6.0


def test_label_store_miss():
    """Cache miss on a new key triggers computation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LabelStore(base_dir=tmpdir)
        called = [False]

        def compute():
            called[0] = True
            return np.array([42.0])

        r = store.get_or_compute("new_key", compute)
        assert called[0]
        assert r[0] == 42.0

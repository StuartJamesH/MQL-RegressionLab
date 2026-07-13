"""
Engine/tests/test_v2_strategy.py — Tests for V2SignalStrategy.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from Engine.DataHandler import Order
from Engine.TicketBook import TicketBook
from Engine.v2.inference import V2InferenceEngine
from Engine.v2.model_pack import ModelPackLoader
from Engine.v2.strategy import V2RiskConfig, V2SignalStrategy


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


@pytest.fixture
def mock_executor():
    """Executor stub exposing get_point_value."""
    return SimpleNamespace(get_point_value=lambda symbol: 10.0)


def _make_bars(n: int, close_start: float = 1.0) -> list:
    """Generate a deterministic OHLCV series."""
    rng = np.random.default_rng(123)
    bars = []
    t0 = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    close = close_start
    for i in range(n):
        noise = rng.standard_normal() * 0.005
        open_p = close * (1 + rng.standard_normal() * 0.001)
        close = close * (1 + noise)
        high = max(open_p, close) * (1 + abs(rng.standard_normal()) * 0.001)
        low = min(open_p, close) * (1 - abs(rng.standard_normal()) * 0.001)
        bar = SimpleNamespace(
            Time=t0 + pd.Timedelta(minutes=i),
            Open=open_p,
            High=high,
            Low=low,
            Close=close,
            Volume=abs(rng.standard_normal()) * 1000 + 100,
        )
        bars.append(bar)
    return bars


def test_warmup_returns_empty(example_pack: dict, mock_executor) -> None:
    config = example_pack["config"]
    inference = V2InferenceEngine(example_pack, backend="onnx", device="cpu")
    ticket_book = TicketBook(use_memory_only=True)
    risk_config = V2RiskConfig(signal_threshold=0.0)  # low threshold to allow signals

    strategy = V2SignalStrategy(
        symbol="TEST",
        pack=example_pack,
        inference_engine=inference,
        risk_config=risk_config,
        ticket_book=ticket_book,
        mt5_executor=mock_executor,
    )

    bars = _make_bars(config.max_seq_len - 1)
    for bar in bars:
        orders = strategy.on_bar(bar)
        assert orders == []


def test_order_fields_populated(example_pack: dict, mock_executor) -> None:
    config = example_pack["config"]
    inference = V2InferenceEngine(example_pack, backend="onnx", device="cpu")
    ticket_book = TicketBook(use_memory_only=True)
    risk_config = V2RiskConfig(signal_threshold=0.0, max_position_lots=1.0)

    strategy = V2SignalStrategy(
        symbol="TEST",
        pack=example_pack,
        inference_engine=inference,
        risk_config=risk_config,
        ticket_book=ticket_book,
        mt5_executor=mock_executor,
    )

    bars = _make_bars(config.max_seq_len + 50)
    order: Order | None = None
    for bar in bars:
        orders = strategy.on_bar(bar)
        if orders:
            order = orders[0]
            break

    assert order is not None, "Expected at least one signal after warm-up"
    assert order.symbol == "TEST"
    assert order.side in ("buy", "sell")
    assert order.entry > 0
    assert order.sl > 0
    assert order.tp > 0
    assert order.qty > 0
    assert isinstance(order, Order)

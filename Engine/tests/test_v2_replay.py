"""
Engine/tests/test_v2_replay.py — End-to-end replay smoke test.

Uses a CSV-backed :class:`Engine.DataHandler.DataHandler` and a mock executor
so the test passes even when MetaTrader5 is unavailable.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from Engine.DataHandler import DataHandler, Order
from Engine.TicketBook import TicketBook
from Engine.v2.engine import V2LiveEngine
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


class _MockExecutor:
    """Minimal executor mock satisfying the V2LiveEngine interface."""

    def __init__(self, ticket_book: TicketBook) -> None:
        self.ticket_book = ticket_book
        self.submitted: list[Order] = []
        self.pending_batches = 0
        self.position_batches = 0

    def get_point_value(self, symbol: str) -> float:
        return 10.0

    def execute_market_order(self, order: Order) -> Order:
        self.submitted.append(order)
        return order

    def submit_stop_order(self, order: Order) -> int:
        self.submitted.append(order)
        # Record a synthetic pending ticket so the strategy sees state.
        ticket = 1000 + len(self.submitted)
        self.ticket_book.record_order(
            ticket=ticket,
            symbol=order.symbol,
            side=order.side,
            qty=float(order.qty),
            entry_price=order.entry,
            sl=order.sl or 0.0,
            tp=order.tp or 0.0,
            submission_time=datetime.utcnow(),
            expiration_time=order.expiration,
            strategy_name="replay_test",
        )
        return ticket

    def process_pending_batch(self, current_time: Optional[datetime] = None) -> None:
        self.pending_batches += 1

    def process_position_updates_batch(
        self, current_time: Optional[datetime] = None
    ) -> None:
        self.position_batches += 1

    def shutdown(self) -> None:
        pass


def _make_csv(tmp_path: Path, n_bars: int) -> Path:
    rng = np.random.default_rng(7)
    t0 = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    rows = []
    close = 100.0
    for i in range(n_bars):
        open_p = close + rng.standard_normal() * 0.05
        close = open_p + rng.standard_normal() * 0.1
        high = max(open_p, close) + abs(rng.standard_normal()) * 0.05
        low = min(open_p, close) - abs(rng.standard_normal()) * 0.05
        rows.append(
            {
                "Time": t0 + pd.Timedelta(minutes=i),
                "Open": open_p,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": abs(rng.standard_normal()) * 1000 + 100,
            }
        )
    df = pd.DataFrame(rows)
    path = tmp_path / "replay.csv"
    df.to_csv(path, index=False)
    return path


def test_csv_replay_engine(example_pack: dict, tmp_path: Path) -> None:
    config = example_pack["config"]
    n_bars = config.max_seq_len + 50
    csv_path = _make_csv(tmp_path, n_bars)

    data_handler = DataHandler.from_csv(str(csv_path))
    ticket_book = TicketBook(use_memory_only=True)
    executor = _MockExecutor(ticket_book)
    inference = V2InferenceEngine(example_pack, backend="onnx", device="cpu")

    risk_config = V2RiskConfig(
        signal_threshold=0.0,  # allow signals
        max_position_lots=1.0,
        account_equity=10_000.0,
    )
    strategy = V2SignalStrategy(
        symbol="REPLAY",
        pack=example_pack,
        inference_engine=inference,
        risk_config=risk_config,
        ticket_book=ticket_book,
        mt5_executor=executor,
    )

    engine = V2LiveEngine(data_handler, strategy, executor)
    result = engine.run()

    assert result["bars_processed"] == n_bars
    assert executor.pending_batches == n_bars
    assert executor.position_batches == n_bars

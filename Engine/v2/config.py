"""
Engine/v2/config.py — Runtime configuration dataclass for the v2 engine.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class V2RuntimeConfig:
    """
    Immutable-style runtime configuration for one symbol instance.

    Notes
    -----
    ``magic`` and ``db_path`` must be unique per running instance so that
    multiple symbols / strategies do not collide on pending orders or SQLite
    journals.
    """

    symbol: str
    magic: int
    pack_dir: str
    db_path: str
    timeframe: str = "M1"
    risk_per_trade: float = 50.0
    max_position_lots: float = 0.5
    signal_threshold: float = 0.1
    patience_bars: int = 5
    backend: str = "onnx"
    device: str = "cpu"
    log_dir: str = "Engine/v2/Trade Logs"
    account_equity: float = 10_000.0
    temperature: float = 1.0
    primary_horizon_idx: int = 2
    extreme_regime_idx: int = 3
    order_type: str = "stop"

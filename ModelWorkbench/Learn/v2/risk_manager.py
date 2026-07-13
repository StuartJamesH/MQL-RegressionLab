"""
risk_manager.py — RiskManager + RiskConfig

Enforces risk limits and computes exit conditions for active positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class RiskConfig:
    """Risk management parameters."""

    max_concurrent_positions: int = 3
    max_total_exposure_pct: float = 0.15
    trailing_stop_atr_mult: float = 1.5
    take_profit_atr_mult: float = 3.0
    hard_stop_pct: float = 0.02  # 2% of account
    atr_window: int = 14

    def __post_init__(self):
        if self.max_concurrent_positions < 1:
            raise ValueError("max_concurrent_positions must be >= 1")
        if not (0 < self.hard_stop_pct <= 1.0):
            raise ValueError("hard_stop_pct must be in (0, 1]")


class RiskManager:
    """
    Enforces risk limits and computes exit conditions.

    Usage:
        rm = RiskManager(RiskConfig())
        if rm.check_entry_allowed(equity, positions):
            tp, sl, trail = rm.compute_exit_levels(price, direction, atr, equity)
            # ... open position ...
            new_sl = rm.update_trailing_stop(position, current_bar)
    """

    def __init__(self, config: RiskConfig):
        self.config = config

    def check_entry_allowed(
        self,
        account_equity: float,
        current_positions: list,
    ) -> bool:
        """
        Check if a new position can be opened given current risk state.

        Args:
            account_equity: Current account equity.
            current_positions: List of active position dicts.

        Returns:
            True if entry is allowed, False otherwise.
        """
        # Check concurrent position limit
        active = [p for p in current_positions if p.get("active", True)]
        if len(active) >= self.config.max_concurrent_positions:
            return False

        # Check total exposure limit
        net_exposure = self.get_net_exposure(active)
        if net_exposure >= self.config.max_total_exposure_pct * account_equity:
            return False

        return True

    def compute_exit_levels(
        self,
        entry_price: float,
        direction: int,
        current_atr: float,
        account_equity: float,
    ) -> tuple[float, float, float]:
        """
        Compute take-profit, stop-loss, and initial trailing stop levels.

        Args:
            entry_price: Entry price of the position.
            direction: 1 for long, -1 for short.
            current_atr: Current ATR value (in price units).
            account_equity: Current account equity.

        Returns:
            (tp_price, sl_price, trailing_sl_initial)
        """
        atr = max(current_atr, 1e-8)
        tp_distance = self.config.take_profit_atr_mult * atr
        sl_distance = self.config.trailing_stop_atr_mult * atr
        hard_sl = entry_price * self.config.hard_stop_pct

        if direction == 1:  # Long
            tp_price = entry_price + tp_distance
            sl_price = entry_price - max(sl_distance, hard_sl)
        else:  # Short
            tp_price = entry_price - tp_distance
            sl_price = entry_price + max(sl_distance, hard_sl)

        trailing_sl_initial = sl_price

        return tp_price, sl_price, trailing_sl_initial

    def update_trailing_stop(
        self,
        position: dict,
        current_bar: dict,
    ) -> float:
        """
        Update trailing stop based on favorable price movement.

        Args:
            position: Dict with keys: direction, trailing_sl, entry_price.
            current_bar: Dict with keys: High, Low, Close, atr.

        Returns:
            Updated trailing stop price.
        """
        direction = position.get("direction", 0)
        current_sl = position.get("trailing_sl", position.get("entry_price", 0))
        atr = current_bar.get("atr", 0)

        if atr <= 0:
            return current_sl

        trail_distance = self.config.trailing_stop_atr_mult * atr

        if direction == 1:  # Long
            new_sl = current_bar.get("High", current_bar.get("Close", 0)) - trail_distance
            return max(current_sl, new_sl)
        else:  # Short
            new_sl = current_bar.get("Low", current_bar.get("Close", 0)) + trail_distance
            return min(current_sl, new_sl)

    def get_net_exposure(self, positions: list) -> float:
        """
        Sum of absolute position values.

        Args:
            positions: List of position dicts with 'size' key.

        Returns:
            Total absolute exposure in account currency units.
        """
        return sum(abs(p.get("size", 0.0)) for p in positions)

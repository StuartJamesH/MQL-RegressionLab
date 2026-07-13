"""
backtest.py — VectorizedBacktester

Single-pass vectorized backtester with realistic trading assumptions:
spread, commission, maximum hold duration, ATR-based exits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Trade:
    """Single completed trade record."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # 1=long, -1=short
    entry_price: float
    exit_price: float
    pnl_pips: float
    pnl_pct: float
    mfe_pips: float  # maximum favorable excursion
    mae_pips: float  # maximum adverse excursion
    duration_bars: int
    exit_reason: str  # "tp", "sl", "timeout", "signal_reversed", "eod"


class VectorizedBacktester:
    """
    Single-pass vectorized backtester.

    Assumptions:
      - One position at a time (sequential entry/exit).
      - Entry at next bar's Open after signal.
      - Exit at TP/SL level intra-bar (uses High/Low).
      - Spread applied at entry.
      - Commission applied per trade (round-trip).
    """

    def __init__(
        self,
        spread_pips: float = 0.3,
        commission_per_lot: float = 7.0,
        lot_size: float = 100000,
        max_hold_bars: int = 120,
    ):
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.lot_size = lot_size
        self.max_hold_bars = max_hold_bars

    def run(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        tp_atr_mult: float = 3.0,
        sl_atr_mult: float = 1.5,
    ) -> Tuple[List[Trade], np.ndarray]:
        """
        Args:
            df: OHLCV DataFrame with 'Time','Open','High','Low','Close','atr'.
            signals: (n_bars,) array in [-1, 1].
            tp_atr_mult: Take-profit ATR multiplier.
            sl_atr_mult: Stop-loss ATR multiplier.

        Returns:
            (trades: List[Trade], equity_curve: np.ndarray)
        """
        n = len(df)
        if len(signals) != n:
            raise ValueError(f"signal length {len(signals)} != df length {n}")

        times = df["Time"].values
        opens = df["Open"].values.astype(np.float64)
        highs = df["High"].values.astype(np.float64)
        lows = df["Low"].values.astype(np.float64)
        closes = df["Close"].values.astype(np.float64)
        atrs = df.get("atr")
        if atrs is None:
            raise KeyError("DataFrame must have 'atr' column")
        atrs = atrs.values.astype(np.float64)

        equity = 0.0
        equity_curve = np.zeros(n)
        trades: List[Trade] = []

        in_position = False
        pos_direction = 0
        entry_price = 0.0
        entry_bar = 0
        tp_price = 0.0
        sl_price = 0.0
        mfe = 0.0
        mae = 0.0

        for i in range(n - 1):
            equity_curve[i] = equity

            if in_position:
                # Check exits intra-bar
                h = highs[i]
                l = lows[i]

                if pos_direction == 1:  # Long
                    mfe = max(mfe, h - entry_price)
                    mae = min(mae, l - entry_price)

                    if h >= tp_price:
                        exit_price = tp_price
                        exit_reason = "tp"
                    elif l <= sl_price:
                        exit_price = sl_price
                        exit_reason = "sl"
                    elif i - entry_bar >= self.max_hold_bars:
                        exit_price = closes[i]
                        exit_reason = "timeout"
                    else:
                        # Check signal reversal
                        if signals[i] < 0 and abs(signals[i]) > 0.3:
                            exit_price = closes[i]
                            exit_reason = "signal_reversed"
                        else:
                            continue
                else:  # Short
                    mfe = max(mfe, entry_price - l)
                    mae = min(mae, entry_price - h)

                    if l <= tp_price:
                        exit_price = tp_price
                        exit_reason = "tp"
                    elif h >= sl_price:
                        exit_price = sl_price
                        exit_reason = "sl"
                    elif i - entry_bar >= self.max_hold_bars:
                        exit_price = closes[i]
                        exit_reason = "timeout"
                    else:
                        if signals[i] > 0 and abs(signals[i]) > 0.3:
                            exit_price = closes[i]
                            exit_reason = "signal_reversed"
                        else:
                            continue

                # Compute P&L
                pnl_pips = pos_direction * (exit_price - entry_price) * 10000
                pnl_pct = pos_direction * (exit_price - entry_price) / entry_price

                # Deduct spread (half at entry, half at exit is already priced)
                # Commission
                commission = self.commission_per_lot / self.lot_size * 2  # round-trip
                pnl_pct -= commission

                equity += pnl_pips * 0.01  # Convert pips to account currency

                trades.append(Trade(
                    entry_time=times[entry_bar],
                    exit_time=times[i],
                    direction=pos_direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pips=pnl_pips,
                    pnl_pct=pnl_pct,
                    mfe_pips=mfe * 10000,
                    mae_pips=mae * 10000,
                    duration_bars=i - entry_bar,
                    exit_reason=exit_reason,
                ))

                in_position = False
                continue

            # Check for entry signal
            signal_val = signals[i]
            if abs(signal_val) > 0.1:  # Threshold
                entry_bar = i + 1
                if entry_bar >= n:
                    break

                pos_direction = 1 if signal_val > 0 else -1
                atr_val = atrs[i]
                if np.isnan(atr_val) or atr_val <= 0:
                    continue

                entry_price = opens[entry_bar]
                spread_cost = self.spread_pips / 10000
                if pos_direction == 1:
                    entry_price += spread_cost  # Buy at ask
                    tp_price = entry_price + tp_atr_mult * atr_val
                    sl_price = entry_price - sl_atr_mult * atr_val
                else:
                    entry_price -= spread_cost  # Sell at bid
                    tp_price = entry_price - tp_atr_mult * atr_val
                    sl_price = entry_price + sl_atr_mult * atr_val

                mfe = 0.0
                mae = 0.0
                in_position = True

        # Close any open position at end of data
        if in_position:
            last_idx = n - 1
            exit_price = closes[last_idx]
            pnl_pips = pos_direction * (exit_price - entry_price) * 10000
            pnl_pct = pos_direction * (exit_price - entry_price) / entry_price
            commission = self.commission_per_lot / self.lot_size * 2
            pnl_pct -= commission
            equity += pnl_pips * 0.01

            trades.append(Trade(
                entry_time=times[entry_bar],
                exit_time=times[last_idx],
                direction=pos_direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pips=pnl_pips,
                pnl_pct=pnl_pct,
                mfe_pips=mfe * 10000,
                mae_pips=mae * 10000,
                duration_bars=last_idx - entry_bar,
                exit_reason="eod",
            ))

        equity_curve[-1] = equity

        return trades, equity_curve

    def to_dataframe(self, trades: List[Trade]) -> pd.DataFrame:
        """Convert trades list to DataFrame for analysis."""
        if not trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl_pips": t.pnl_pips,
            "pnl_pct": t.pnl_pct,
            "mfe_pips": t.mfe_pips,
            "mae_pips": t.mae_pips,
            "duration_bars": t.duration_bars,
            "exit_reason": t.exit_reason,
        } for t in trades])

"""
walk_forward_backtest.py — WalkForwardBacktest

Proper walk-forward backtesting: train on past, test on future.
Aggregates predictions across folds for final evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward backtesting."""

    signals: np.ndarray       # (n_bars,) — concatenated out-of-fold signals
    trades: list              # List of all trades
    metrics_per_fold: list    # List of metric dicts per fold
    equity_curve: np.ndarray  # (n_bars,) — full equity curve
    fold_boundaries: list     # List of (train_start, test_start, test_end)


class WalkForwardBacktest:
    """
    Walk-forward backtesting with purged gap between train and test.

    For each fold:
      1. Train on [0 : train_end]
      2. Predict on [train_end + gap : train_end + gap + test_size]
      3. Aggregate all predictions for final backtest
    """

    def __init__(
        self,
        n_folds: int = 5,
        train_fraction: float = 0.8,
        gap_bars: int = 120,
    ):
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if not (0 < train_fraction < 1):
            raise ValueError("train_fraction must be in (0, 1)")
        self.n_folds = n_folds
        self.train_fraction = train_fraction
        self.gap_bars = gap_bars

    def run(
        self,
        df: pd.DataFrame,
        train_fn: Callable,
        predict_fn: Callable,
        backtest_class=None,
        **backtest_kwargs,
    ) -> WalkForwardResult:
        """
        Execute walk-forward backtest.

        Args:
            df: Full OHLCV DataFrame.
            train_fn: callable(train_df) → trained_model.
            predict_fn: callable(model, test_df) → signals array.
            backtest_class: Optional VectorizedBacktester class for per-fold backtesting.
            **backtest_kwargs: Passed to backtest class constructor.

        Returns:
            WalkForwardResult with aggregated results.
        """
        n = len(df)
        train_size = int(n * self.train_fraction)
        test_size_each = (n - train_size) // self.n_folds

        all_signals = np.full(n, np.nan)
        all_trades: list = []
        metrics_per_fold: list = []
        fold_boundaries: list = []

        current_train_end = train_size

        for fold in range(self.n_folds):
            test_start = min(current_train_end + self.gap_bars, n - 1)
            test_end = min(test_start + test_size_each, n)

            if test_end <= test_start:
                break

            # Split data
            train_df = df.iloc[:current_train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()

            print(f"Fold {fold + 1}/{self.n_folds}: "
                  f"train=[0:{current_train_end}], "
                  f"test=[{test_start}:{test_end}] "
                  f"({len(test_df)} bars)")

            # Train
            model = train_fn(train_df)

            # Predict
            test_signals = predict_fn(model, test_df)
            all_signals[test_start:test_start + len(test_signals)] = test_signals

            # Optional per-fold backtest
            if backtest_class is not None:
                bt = backtest_class(**backtest_kwargs)
                fold_trades, fold_eq = bt.run(test_df, test_signals)
                all_trades.extend(fold_trades)

                from Learn.v2.backtest_metrics import BacktestMetrics
                fold_metrics = BacktestMetrics.compute(fold_trades, fold_eq)
                metrics_per_fold.append(fold_metrics)

            fold_boundaries.append((current_train_end, test_start, test_end))
            current_train_end = test_end

        # Run full backtest on all aggregated signals
        if backtest_class is not None:
            bt_full = backtest_class(**backtest_kwargs)
            all_trades, equity_curve = bt_full.run(
                df, np.nan_to_num(all_signals, nan=0.0),
            )
        else:
            equity_curve = np.zeros(n)

        return WalkForwardResult(
            signals=all_signals,
            trades=all_trades,
            metrics_per_fold=metrics_per_fold,
            equity_curve=equity_curve,
            fold_boundaries=fold_boundaries,
        )

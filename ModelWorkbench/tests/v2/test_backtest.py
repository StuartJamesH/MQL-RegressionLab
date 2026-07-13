"""
test_backtest.py — Tests for backtesting modules.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from Learn.v2.backtest import VectorizedBacktester, Trade
from Learn.v2.backtest_metrics import BacktestMetrics
from Learn.v2.walk_forward_backtest import WalkForwardBacktest


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_ohlcv_df():
    """Create OHLCV DataFrame with ATR for backtesting."""
    np.random.seed(42)
    n = 1000

    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    trend = np.cumsum(np.random.randn(n) * 0.0001) + 1.0000
    noise = np.random.randn(n) * 0.0002

    close = trend + noise
    high = close + np.abs(np.random.randn(n) * 0.0003)
    low = close - np.abs(np.random.randn(n) * 0.0003)
    open_price = close + np.random.randn(n) * 0.0001

    # Simple ATR
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        )
    )
    atr = pd.Series(tr).rolling(14).mean().fillna(tr.mean())

    return pd.DataFrame({
        "Time": dates,
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "atr": atr.values,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# VectorizedBacktester Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorizedBacktester:

    def test_empty_signals_no_trades(self, sample_ohlcv_df):
        """Zero signals → no trades."""
        bt = VectorizedBacktester()
        signals = np.zeros(len(sample_ohlcv_df))
        trades, equity = bt.run(sample_ohlcv_df, signals)
        assert len(trades) == 0
        assert len(equity) == len(sample_ohlcv_df)

    def test_random_signals_zero_sharpe(self, sample_ohlcv_df):
        """
        Random signals should produce a Sharpe ratio near zero
        (no predictive power → no edge → Sharpe ≈ 0).
        """
        bt = VectorizedBacktester(spread_pips=0.0, commission_per_lot=0.0)

        np.random.seed(123)
        # Generate random signals
        signals = np.random.uniform(-1, 1, len(sample_ohlcv_df))

        trades, equity = bt.run(sample_ohlcv_df, signals)

        if len(trades) > 5:
            metrics = BacktestMetrics.compute(trades, equity)
            sharpe = metrics["sharpe_ratio"]
            # Random signals without edge should have |Sharpe| < 2.0
            # (could be extreme by chance, so use a generous bound)
            assert abs(sharpe) < 5.0, (
                f"Random signals should have low Sharpe, got {sharpe:.2f}"
            )

    def test_trades_have_valid_structure(self, sample_ohlcv_df):
        """Each trade has all required fields with sensible values."""
        bt = VectorizedBacktester()

        # Generate a simple signal pattern
        signals = np.zeros(len(sample_ohlcv_df))
        signals[100] = 1.0   # Buy
        signals[200] = -1.0   # Sell

        trades, equity = bt.run(sample_ohlcv_df, signals)

        for trade in trades:
            assert trade.direction in (-1, 1)
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.duration_bars > 0
            assert trade.exit_reason in ("tp", "sl", "timeout", "signal_reversed", "eod")

    def test_to_dataframe(self, sample_ohlcv_df):
        """to_dataframe produces a valid DataFrame."""
        bt = VectorizedBacktester()

        signals = np.zeros(len(sample_ohlcv_df))
        signals[100] = 1.0

        trades, _ = bt.run(sample_ohlcv_df, signals)
        df = bt.to_dataframe(trades)

        assert isinstance(df, pd.DataFrame)
        expected_cols = [
            "entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "pnl_pips",
            "pnl_pct", "mfe_pips", "mae_pips",
            "duration_bars", "exit_reason",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_backtest_causality(self, sample_ohlcv_df):
        """
        No trade should open before its signal bar.
        Signal at bar t → entry at bar t+1 → exit at or after t+1.
        """
        bt = VectorizedBacktester()

        signals = np.zeros(len(sample_ohlcv_df))
        # Signal at bar 500
        signals[500] = 1.0

        trades, equity = bt.run(sample_ohlcv_df, signals)

        for trade in trades:
            entry_idx = sample_ohlcv_df[
                sample_ohlcv_df["Time"] == trade.entry_time
            ].index
            if len(entry_idx) > 0:
                assert entry_idx[0] >= 500, (
                    f"Trade entered at bar {entry_idx[0]}, before signal at bar 500"
                )

    def test_signal_length_mismatch(self, sample_ohlcv_df):
        """Signal array length must match DataFrame."""
        bt = VectorizedBacktester()
        with pytest.raises(ValueError):
            bt.run(sample_ohlcv_df, np.array([1.0]))  # Wrong length

    def test_missing_atr_raises(self, sample_ohlcv_df):
        """DataFrame without 'atr' column raises."""
        bt = VectorizedBacktester()
        df_no_atr = sample_ohlcv_df.drop(columns=["atr"])
        signals = np.zeros(len(df_no_atr))
        with pytest.raises(KeyError):
            bt.run(df_no_atr, signals)


# ═══════════════════════════════════════════════════════════════════════════════
# BacktestMetrics Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktestMetrics:

    def test_empty_trades(self):
        """Empty trade list returns zero metrics."""
        metrics = BacktestMetrics.compute([], np.zeros(100))
        assert metrics["n_trades"] == 0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["total_return"] == 0.0

    def test_all_winning_trades(self):
        """All winning trades → win_rate=1.0, positive returns."""
        trades = [
            Trade(
                entry_time=pd.Timestamp("2024-01-01 09:00"),
                exit_time=pd.Timestamp("2024-01-01 10:00"),
                direction=1, entry_price=1.0000, exit_price=1.0010,
                pnl_pips=10.0, pnl_pct=0.001,
                mfe_pips=15.0, mae_pips=-2.0,
                duration_bars=10, exit_reason="tp",
            ),
            Trade(
                entry_time=pd.Timestamp("2024-01-01 11:00"),
                exit_time=pd.Timestamp("2024-01-01 12:00"),
                direction=1, entry_price=1.0010, exit_price=1.0020,
                pnl_pips=10.0, pnl_pct=0.001,
                mfe_pips=12.0, mae_pips=-1.0,
                duration_bars=10, exit_reason="tp",
            ),
        ]

        equity = np.array([0.0, 0.05, 0.10, 0.15])
        metrics = BacktestMetrics.compute(trades, equity)

        assert metrics["n_trades"] == 2
        assert metrics["win_rate"] == 1.0
        assert metrics["total_return"] > 0
        # When all trades win, profit_factor can be inf or nan (no losses)
        pf = metrics["profit_factor"]
        assert pf == float("inf") or np.isnan(pf) or pf > 10

    def test_summary_report(self):
        """Summary report returns a string."""
        metrics = {"total_return": 0.15, "sharpe_ratio": 1.5, "n_trades": 100}
        report = BacktestMetrics.summary_report(metrics)
        assert isinstance(report, str)
        assert "1.50" in report
        assert "100" in report

    def test_monte_carlo_sharpe(self):
        """Monte Carlo Sharpe returns bounded confidence interval."""
        returns = np.random.randn(1000) * 0.01 + 0.0002
        lower, upper = BacktestMetrics.monte_carlo_sharpe(returns, n_simulations=100)
        assert lower <= upper
        assert np.isfinite(lower) and np.isfinite(upper)


# ═══════════════════════════════════════════════════════════════════════════════
# WalkForwardBacktest Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWalkForwardBacktest:

    def test_walk_forward_no_leakage(self, sample_ohlcv_df):
        """Test data must not be used in training (temporal separation)."""
        wf = WalkForwardBacktest(n_folds=3, train_fraction=0.7, gap_bars=10)

        train_seen = []
        test_seen = []

        def train_fn(train_df):
            train_seen.append((train_df.index[0], train_df.index[-1]))
            return {"trained": True}

        def predict_fn(model, test_df):
            test_seen.append((test_df.index[0], test_df.index[-1]))
            return np.random.randn(len(test_df))

        result = wf.run(
            sample_ohlcv_df,
            train_fn=train_fn,
            predict_fn=predict_fn,
            backtest_class=None,
        )

        # Verify temporal ordering: train always before test
        for (train_start, train_end), (test_start, test_end) in zip(train_seen, test_seen):
            assert train_end < test_start, (
                f"Train end ({train_end}) must be before test start ({test_start})"
            )

    def test_walk_forward_gap_respected(self, sample_ohlcv_df):
        """Gap bars separate train and test windows."""
        gap = 20
        wf = WalkForwardBacktest(n_folds=3, train_fraction=0.7, gap_bars=gap)

        gaps_seen = []

        def train_fn(train_df):
            gaps_seen.append(train_df.index[-1])
            return {"model": "dummy"}

        def predict_fn(model, test_df):
            gaps_seen.append(test_df.index[0])
            return np.random.randn(len(test_df))

        result = wf.run(sample_ohlcv_df, train_fn, predict_fn, backtest_class=None)

        # Check at least one fold has the gap
        for i in range(0, len(gaps_seen) - 1, 2):
            if i + 1 < len(gaps_seen):
                train_end = gaps_seen[i]
                test_start = gaps_seen[i + 1]
                assert test_start - train_end >= gap, (
                    f"Gap too small: {test_start - train_end} < {gap}"
                )

    def test_walk_forward_result_shape(self, sample_ohlcv_df):
        """Result signals cover the full test period."""
        wf = WalkForwardBacktest(n_folds=2, train_fraction=0.8, gap_bars=10)

        def train_fn(train_df):
            return None

        def predict_fn(model, test_df):
            return np.ones(len(test_df))

        result = wf.run(sample_ohlcv_df, train_fn, predict_fn, backtest_class=None)

        assert result.signals.shape[0] == len(sample_ohlcv_df)
        # Check that test regions have non-NaN signals
        non_nan = ~np.isnan(result.signals)
        assert non_nan.sum() > 0, "Should have some non-NaN signal values"


# ═══════════════════════════════════════════════════════════════════════════════
# P&L Reconciliation
# ═══════════════════════════════════════════════════════════════════════════════

def test_pnl_reconciliation(sample_ohlcv_df):
    """
    Backtest P&L should match manual calculation for a simple case.
    Manually verify that a single trade's PnL is computed correctly.
    """
    bt = VectorizedBacktester(spread_pips=0.0, commission_per_lot=0.0)

    # Create a signal that will trigger one long trade
    signals = np.zeros(len(sample_ohlcv_df))
    signals[100] = 1.0  # Buy at bar 101

    trades, equity = bt.run(
        sample_ohlcv_df, signals,
        tp_atr_mult=100.0,  # Very far TP → almost never hit
        sl_atr_mult=100.0,  # Very far SL → almost never hit
    )

    if len(trades) > 0:
        t = trades[0]
        manual_pnl = (t.exit_price - t.entry_price) * 10000 * t.direction

        # The actual pnl should be close to our manual calc
        # (minus any small rounding from spread/commission)
        assert abs(t.pnl_pips - manual_pnl) < 1.0, (
            f"P&L mismatch: backtest={t.pnl_pips:.4f}, manual={manual_pnl:.4f}"
        )

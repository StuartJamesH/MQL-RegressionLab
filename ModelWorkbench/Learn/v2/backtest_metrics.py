"""
backtest_metrics.py — BacktestMetrics

Comprehensive backtest performance metrics including Sharpe, Sortino,
drawdown analysis, Monte Carlo confidence intervals, and visualization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import warnings


class BacktestMetrics:
    """Computes comprehensive backtest performance metrics."""

    @staticmethod
    def compute(
        trades: list,
        equity_curve: np.ndarray,
        account_initial: float = 10000.0,
        annual_factor: float = 252,
    ) -> dict:
        """
        Compute comprehensive metrics from trades and equity curve.

        Args:
            trades: List of Trade dataclass instances.
            equity_curve: (n_bars,) array of cumulative equity values.
            account_initial: Starting account value.
            annual_factor: Days per year for annualisation.

        Returns:
            Dict of metric_name → value.
        """
        if not trades:
            return {
                "total_return": 0.0, "cagr": 0.0,
                "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
                "max_drawdown": 0.0, "max_drawdown_duration_days": 0,
                "win_rate": 0.0, "profit_factor": np.nan, "expectancy": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0,
                "trades_per_day": 0.0, "n_trades": 0,
            }

        # Extract trade P&L
        pnls = np.array([t.pnl_pips for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        n_trades = len(pnls)
        n_wins = len(wins)
        n_losses = len(losses)

        win_rate = n_wins / n_trades if n_trades > 0 else 0.0
        avg_win = float(np.mean(wins)) if n_wins > 0 else 0.0
        avg_loss = float(np.mean(np.abs(losses))) if n_losses > 0 else 0.0

        # Profit factor
        gross_profit = float(wins.sum()) if n_wins > 0 else 0.0
        gross_loss = float(np.abs(losses.sum())) if n_losses > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

        # Expectancy
        expectancy = float(np.mean(pnls)) if n_trades > 0 else 0.0

        # Returns from equity curve
        eq = equity_curve + account_initial
        eq = np.maximum(eq, 1.0)  # Prevent negative equity

        # Daily returns (approximate from bar-level equity changes)
        returns = np.diff(eq) / eq[:-1]
        returns = returns[np.isfinite(returns)]

        # Total return
        final_eq = eq[-1] if len(eq) > 0 else account_initial
        total_return = (final_eq - account_initial) / account_initial

        # CAGR
        n_days = len(returns)
        if n_days > 1:
            cagr = (final_eq / account_initial) ** (annual_factor / n_days) - 1
        else:
            cagr = 0.0

        # Sharpe ratio
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(annual_factor))
        else:
            sharpe = 0.0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and np.std(downside_returns) > 0:
            sortino = float(np.mean(returns) / np.std(downside_returns) * np.sqrt(annual_factor))
        else:
            sortino = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / peak
        max_dd = float(np.min(drawdown))
        max_dd_idx = np.argmin(drawdown)
        # Drawdown duration
        if max_dd < 0:
            dd_start = np.where(drawdown[:max_dd_idx] == 0)[0]
            dd_start = dd_start[-1] if len(dd_start) > 0 else 0
            dd_duration = max_dd_idx - dd_start
        else:
            dd_duration = 0

        # Trades per day
        if n_trades > 0 and n_days > 0:
            trades_per_day = n_trades / n_days
        else:
            trades_per_day = 0.0

        metrics = {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "max_drawdown_duration_days": int(dd_duration),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "trades_per_day": trades_per_day,
            "n_trades": n_trades,
            "n_wins": n_wins,
            "n_losses": n_losses,
            "final_equity": final_eq,
        }

        return metrics

    @staticmethod
    def monte_carlo_sharpe(
        returns: np.ndarray,
        n_simulations: int = 1000,
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for Sharpe ratio.

        Args:
            returns: Array of period returns.
            n_simulations: Number of bootstrap samples.

        Returns:
            (lower_bound, upper_bound) at 95% confidence.
        """
        if len(returns) < 10:
            return (0.0, 0.0)

        n = len(returns)
        sharpes = np.zeros(n_simulations)
        for i in range(n_simulations):
            sample = np.random.choice(returns, size=n, replace=True)
            if np.std(sample) > 0:
                sharpes[i] = np.mean(sample) / np.std(sample) * np.sqrt(252)
            else:
                sharpes[i] = 0.0

        lower = float(np.percentile(sharpes, 2.5))
        upper = float(np.percentile(sharpes, 97.5))
        return (lower, upper)

    @staticmethod
    def summary_report(metrics: dict) -> str:
        """Pretty-print formatted metrics report."""
        lines = [
            "=" * 60,
            "              BACKTEST PERFORMANCE REPORT",
            "=" * 60,
            f"  Total Return:           {metrics.get('total_return', 0):.2%}",
            f"  CAGR:                   {metrics.get('cagr', 0):.2%}",
            f"  Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):.2f}",
            f"  Sortino Ratio:          {metrics.get('sortino_ratio', 0):.2f}",
            f"  Max Drawdown:           {metrics.get('max_drawdown', 0):.2%}",
            f"  Max DD Duration:        {metrics.get('max_drawdown_duration_days', 0)} days",
            "-" * 60,
            f"  Win Rate:               {metrics.get('win_rate', 0):.2%}",
            f"  Profit Factor:          {metrics.get('profit_factor', float('nan')):.2f}",
            f"  Avg Win:                {metrics.get('avg_win', 0):.4f}",
            f"  Avg Loss:               {metrics.get('avg_loss', 0):.4f}",
            f"  Trades:                 {metrics.get('n_trades', 0)}",
            f"  Trades/Day:             {metrics.get('trades_per_day', 0):.2f}",
            f"  Final Equity:           ${metrics.get('final_equity', 0):,.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    @staticmethod
    def plot_equity_curve(
        equity_curve: np.ndarray,
        timestamps,
        drawdown: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ):
        """Plot equity curve with drawdown overlay using matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed; skipping equity curve plot.")
            return

        equity = equity_curve + 10000  # Assuming initial balance
        if drawdown is None:
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                         gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(timestamps, equity, color='steelblue', linewidth=1, label='Equity')
        ax1.fill_between(range(len(equity)), equity, 10000,
                         where=(equity >= 10000), color='green', alpha=0.1)
        ax1.fill_between(range(len(equity)), equity, 10000,
                         where=(equity < 10000), color='red', alpha=0.1)
        ax1.set_ylabel("Account Equity ($)")
        ax1.set_title("Equity Curve")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.5)
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Bar Index")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved equity curve to {save_path}")
        else:
            plt.show()
        plt.close()

"""
signal_evaluator.py — SignalEvaluator

Offline signal quality evaluation.
Computes: decile analysis, calibration curves, threshold sweeps, profit curves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
import warnings


class SignalEvaluator:
    """
    Evaluates the quality of trade signals against realized outcomes.

    All methods accept signals (predicted direction/strength) and
    realized outcomes (actual forward returns or binary win/loss labels).
    """

    def __init__(self):
        self._decile_results: Optional[pd.DataFrame] = None
        self._calibration: Optional[pd.DataFrame] = None
        self._threshold_results: Optional[pd.DataFrame] = None
        self._profit_curve: Optional[pd.DataFrame] = None

    def decile_analysis(
        self,
        signals: np.ndarray,
        realized_outcomes: np.ndarray,
    ) -> pd.DataFrame:
        """
        Sort bars by signal strength (absolute value), compute stats per decile.

        Args:
            signals: (N,) array of signal values in [-1, 1].
            realized_outcomes: (N,) array of realized returns or binary outcomes.

        Returns:
            DataFrame with per-decile: count, mean_signal, mean_outcome, win_rate,
            sharpe, cumulative_return.
        """
        valid = np.isfinite(signals) & np.isfinite(realized_outcomes)
        s = signals[valid]
        r = realized_outcomes[valid]

        if len(s) < 10:
            return pd.DataFrame()

        # Sort by absolute signal strength (descending)
        abs_s = np.abs(s)
        order = np.argsort(-abs_s)
        s_sorted = s[order]
        r_sorted = r[order]

        n = len(s_sorted)
        decile_size = n // 10

        rows = []
        for i in range(10):
            start = i * decile_size
            end = start + decile_size if i < 9 else n
            d_s = s_sorted[start:end]
            d_r = r_sorted[start:end]

            win_rate = np.mean(d_r > 0) if len(d_r) > 0 else np.nan
            sharpe = (np.mean(d_r) / (np.std(d_r) + 1e-8) * np.sqrt(252)
                      if len(d_r) > 1 else 0.0)

            rows.append({
                "decile": i + 1,
                "n": len(d_s),
                "mean_signal": float(np.mean(d_s)),
                "mean_abs_signal": float(np.mean(np.abs(d_s))),
                "mean_outcome": float(np.mean(d_r)),
                "win_rate": float(win_rate),
                "sharpe_annualized": float(sharpe),
                "cumulative_return": float(np.sum(d_r)),
            })

        self._decile_results = pd.DataFrame(rows)
        return self._decile_results

    def calibration_curve(
        self,
        predicted_probs: np.ndarray,
        realized_outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Binned predicted P(win) vs actual win rate.

        Args:
            predicted_probs: (N,) predicted probabilities in [0, 1].
            realized_outcomes: (N,) binary outcome labels {0, 1}.
            n_bins: Number of equal-width bins.

        Returns:
            DataFrame with per-bin predicted mean vs actual win rate.
        """
        valid = (np.isfinite(predicted_probs) & np.isfinite(realized_outcomes))
        p = predicted_probs[valid]
        r = realized_outcomes[valid]

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(p, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        rows = []
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() < 10:
                continue
            rows.append({
                "bin": b + 1,
                "bin_center": (bin_edges[b] + bin_edges[b + 1]) / 2,
                "n": int(mask.sum()),
                "predicted_mean": float(p[mask].mean()),
                "actual_rate": float(r[mask].mean()),
            })

        self._calibration = pd.DataFrame(rows)
        return self._calibration

    def threshold_sweep(
        self,
        signals: np.ndarray,
        realized_outcomes: np.ndarray,
        n_thresholds: int = 20,
    ) -> pd.DataFrame:
        """
        Coverage vs win-rate trade-off across signal thresholds.

        Args:
            signals: (N,) signal array.
            realized_outcomes: (N,) realized outcomes.
            n_thresholds: Number of threshold levels to sweep.

        Returns:
            DataFrame with columns: threshold, coverage, win_rate, mean_return.
        """
        valid = np.isfinite(signals) & np.isfinite(realized_outcomes)
        s = signals[valid]
        r = realized_outcomes[valid]

        thresholds = np.linspace(0, np.abs(s).max(), n_thresholds + 1)[1:]

        rows = []
        total_n = len(s)
        for thr in thresholds:
            mask = np.abs(s) >= thr
            taken = mask.sum()
            coverage = taken / total_n if total_n > 0 else 0.0

            if taken > 0:
                win_rate = np.mean(r[mask] > 0)
                mean_ret = np.mean(r[mask])
            else:
                win_rate = np.nan
                mean_ret = np.nan

            rows.append({
                "threshold": float(thr),
                "coverage": float(coverage),
                "n_taken": int(taken),
                "win_rate": float(win_rate) if not np.isnan(win_rate) else np.nan,
                "mean_return": float(mean_ret) if not np.isnan(mean_ret) else np.nan,
            })

        self._threshold_results = pd.DataFrame(rows)
        return self._threshold_results

    def profit_curve(
        self,
        signals: np.ndarray,
        realized_outcomes: np.ndarray,
        position_sizes: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Cumulative P&L if trading top N% strongest signals.

        Args:
            signals: (N,) signal array.
            realized_outcomes: (N,) realized returns.
            position_sizes: (N,) optional position sizes.

        Returns:
            DataFrame with cumulative returns at each percentile cutoff.
        """
        valid = np.isfinite(signals) & np.isfinite(realized_outcomes)
        s = signals[valid]
        r = realized_outcomes[valid]

        if position_sizes is not None:
            w = position_sizes[valid]
        else:
            w = np.ones_like(s)

        abs_s = np.abs(s)
        order = np.argsort(-abs_s)
        s_sorted = s[order]
        r_sorted = r[order]
        w_sorted = w[order]

        # Simulate trading: if signal > 0 → long (r), else → short (-r)
        pnl = np.where(s_sorted > 0, r_sorted, -r_sorted)
        weighted_pnl = pnl * w_sorted

        cum_pnl = np.cumsum(weighted_pnl)

        # Compute at percentile cutoffs
        n = len(cum_pnl)
        rows = []
        for pct in range(10, 101, 10):
            idx = min(int(n * pct / 100), n - 1)
            if idx >= 0:
                rows.append({
                    "percentile_coverage": pct,
                    "n_trades": idx + 1,
                    "cumulative_pnl": float(cum_pnl[idx]),
                    "avg_pnl_per_trade": float(cum_pnl[idx] / (idx + 1)),
                })

        self._profit_curve = pd.DataFrame(rows)
        return self._profit_curve

    def plot_all(self, save_dir=None):
        """
        Generate all evaluation plots using matplotlib.

        Args:
            save_dir: If provided, save plots to this directory.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed; skipping plots.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Decile analysis
        if self._decile_results is not None and len(self._decile_results) > 0:
            ax = axes[0, 0]
            df = self._decile_results
            ax.bar(df["decile"], df["mean_outcome"], color='steelblue')
            ax.set_title("Mean Outcome by Signal Decile")
            ax.set_xlabel("Decile (1 = strongest)")
            ax.set_ylabel("Mean Outcome")

        # Calibration curve
        if self._calibration is not None and len(self._calibration) > 0:
            ax = axes[0, 1]
            df = self._calibration
            ax.plot(df["bin_center"], df["actual_rate"], 'o-', label='Actual')
            ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect')
            ax.set_title("Calibration Curve")
            ax.set_xlabel("Predicted P(win)")
            ax.set_ylabel("Actual Win Rate")
            ax.legend()

        # Threshold sweep
        if self._threshold_results is not None and len(self._threshold_results) > 0:
            ax = axes[1, 0]
            df = self._threshold_results
            ax.plot(df["coverage"], df["win_rate"], 'o-', color='green')
            ax.set_title("Coverage vs Win Rate")
            ax.set_xlabel("Coverage (fraction of bars)")
            ax.set_ylabel("Win Rate")

        # Profit curve
        if self._profit_curve is not None and len(self._profit_curve) > 0:
            ax = axes[1, 1]
            df = self._profit_curve
            ax.plot(df["percentile_coverage"], df["cumulative_pnl"], 'o-', color='red')
            ax.set_title("Cumulative P&L vs Coverage")
            ax.set_xlabel("Top N% Signals")
            ax.set_ylabel("Cumulative P&L")

        plt.tight_layout()
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, "signal_evaluation.png")
            plt.savefig(path, dpi=150)
            print(f"Saved signal evaluation plots to {path}")
        else:
            plt.show()
        plt.close()

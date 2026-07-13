"""
position_sizing.py — KellyPositionSizer

Kelly criterion position sizing based on predicted edge.
f* = (p_win * avg_win - p_loss * avg_loss) / (avg_win * avg_loss)
Uses half-Kelly by default for robustness.
"""

from __future__ import annotations

import numpy as np


class KellyPositionSizer:
    """
    Kelly criterion position sizing based on predicted edge.

    Computes the optimal fraction of capital to risk on each trade,
    then translates that to position size in account currency units.

    Uses half-Kelly (fractional Kelly) by default to reduce volatility.
    """

    def __init__(self, max_position_pct: float = 0.05, half_kelly: bool = True):
        """
        Args:
            max_position_pct: Maximum fraction of account equity per position.
            half_kelly: If True, use f_star / 2 for more conservative sizing.
        """
        if max_position_pct <= 0 or max_position_pct > 1:
            raise ValueError(f"max_position_pct must be in (0, 1], got {max_position_pct}")
        self.max_position_pct = max_position_pct
        self.half_kelly = half_kelly

    def compute_size(
        self,
        win_prob: float,
        avg_win: float,
        avg_loss: float,
        account_equity: float,
        current_exposure: float = 0.0,
    ) -> float:
        """
        Compute position size in account currency units.

        Args:
            win_prob: Estimated probability of winning [0, 1].
            avg_win: Average win amount (positive).
            avg_loss: Average loss amount (positive).
            account_equity: Current account equity.
            current_exposure: Current total exposure (position values).

        Returns:
            Position size in account currency units.
            Returns 0 if the edge is negative or inputs are invalid.
        """
        if account_equity <= 0:
            return 0.0
        if avg_win <= 0 or avg_loss <= 0:
            return 0.0
        if not (0 <= win_prob <= 1):
            return 0.0

        loss_prob = 1.0 - win_prob
        # Kelly fraction
        f_star = (win_prob * avg_win - loss_prob * avg_loss) / (avg_win * avg_loss)

        if self.half_kelly:
            f_star = f_star / 2.0

        # Clamp negative edge to zero
        f_star = max(f_star, 0.0)

        # Apply maximum position limit
        f_star = min(f_star, self.max_position_pct)

        # Respect exposure limit
        remaining = self.max_position_pct * account_equity - current_exposure
        size = min(f_star * account_equity, remaining)

        return max(size, 0.0)

    def batch_compute(
        self,
        win_probs: np.ndarray,
        avg_wins: np.ndarray,
        avg_losses: np.ndarray,
        account_equity: float,
    ) -> np.ndarray:
        """
        Vectorized position sizing for batch processing.

        Args:
            win_probs: (N,) array of win probabilities.
            avg_wins: (N,) array of average wins.
            avg_losses: (N,) array of average losses.
            account_equity: Current account equity.

        Returns:
            (N,) array of position sizes.
        """
        if account_equity <= 0:
            return np.zeros_like(win_probs)

        loss_probs = 1.0 - win_probs
        denom = avg_wins * avg_losses
        denom = np.where(denom > 1e-8, denom, np.inf)

        f_star = (win_probs * avg_wins - loss_probs * avg_losses) / denom

        if self.half_kelly:
            f_star = f_star / 2.0

        f_star = np.clip(f_star, 0.0, self.max_position_pct)
        sizes = f_star * account_equity

        return sizes

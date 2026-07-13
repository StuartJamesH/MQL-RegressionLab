"""
signals.py — DistributionalSignalGenerator

Transforms model distributional forecasts into actionable trade signals.

Signal range: [-1, 1]
    positive → BUY, negative → SELL, near-zero → HOLD
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class DistributionalSignalGenerator:
    """
    Transforms ModelOutput (distributional forecasts) into scalar trade signals.

    Algorithm:
        1. Expected return (mu) at primary horizon from distribution head
        2. Sharpe-like score: s = mu / sigma
        3. Directional confidence from direction head logits
        4. Composite: sign(s) * tanh(|s| * c / temperature)
        5. Regime gate: zero signal in extreme regimes
        6. Threshold gate: zero signals below minimum strength
    """

    def __init__(
        self,
        temperature: float = 1.0,
        signal_threshold: float = 0.1,
        extreme_regime_gate: bool = True,
        regime_idx: int = 3,
        primary_horizon_idx: int = 2,
    ):
        """
        Args:
            temperature: Scales signal strength (higher = softer signals).
            signal_threshold: Minimum |signal| to emit; below → 0.
            extreme_regime_gate: If True, zero out signals in extreme regime class.
            regime_idx: Which regime class is considered 'extreme' (default 3 = highest vol).
            primary_horizon_idx: Which horizon index to use for primary signal
                (0=5 bars, 1=10 bars, 2=20 bars, 3=40 bars, 4=60 bars, 5=120 bars).
        """
        self.temperature = max(temperature, 1e-3)
        self.signal_threshold = signal_threshold
        self.extreme_regime_gate = extreme_regime_gate
        self.regime_idx = regime_idx
        self.primary_horizon_idx = primary_horizon_idx

    def generate(self, model_output) -> np.ndarray:
        """
        Generate scalar trade signals from ModelOutput.

        Args:
            model_output: ModelOutput dataclass with fields:
                distribution: (mu, log_sigma) each (B, n_horizons)
                direction: (B, n_horizons) logits
                regime: (B, n_regimes) logits
                volatility: (B, n_horizons) log-volatility predictions

        Returns:
            np.ndarray shape (batch_size,) — signal in [-1, 1].
        """
        mu, log_sigma = model_output.distribution
        h = self.primary_horizon_idx

        mu = mu.detach().cpu().numpy()[:, h]
        log_sigma = log_sigma.detach().cpu().numpy()[:, h]

        # Sharpe-like score: expected return / risk
        sigma = np.exp(log_sigma) + 1e-6
        s = mu / sigma  # (batch_size,)

        # Directional confidence from sigmoid(logits)
        # direction output is per-horizon logit for P(return > 0)
        dir_logits = model_output.direction.detach().cpu().numpy()[:, h]
        p_up = 1.0 / (1.0 + np.exp(-dir_logits))  # sigmoid
        c = 2.0 * np.abs(p_up - 0.5)  # confidence in [0, 1]

        # Composite signal: sign * tanh(scaled edge * confidence)
        signal = np.sign(s) * np.tanh(np.abs(s) * c / self.temperature)

        # Regime gate: suppress signals in extreme volatility
        if self.extreme_regime_gate and hasattr(model_output, 'regime'):
            regime_logits = model_output.regime.detach().cpu().numpy()
            regime_pred = np.argmax(regime_logits, axis=1)
            signal[regime_pred == self.regime_idx] = 0.0

        # Threshold gate: zero out weak signals
        signal[np.abs(signal) < self.signal_threshold] = 0.0

        return signal.astype(np.float64)

    def generate_multi_horizon(self, model_output) -> np.ndarray:
        """
        Returns signal per horizon, shape (batch_size, n_horizons).
        """
        mu, log_sigma = model_output.distribution
        mu = mu.detach().cpu().numpy()
        log_sigma = log_sigma.detach().cpu().numpy()

        sigma = np.exp(log_sigma) + 1e-6
        s = mu / sigma  # (batch_size, n_horizons)

        dir_logits = model_output.direction.detach().cpu().numpy()
        p_up = 1.0 / (1.0 + np.exp(-dir_logits))
        c = 2.0 * np.abs(p_up - 0.5)

        signals = np.sign(s) * np.tanh(np.abs(s) * c / self.temperature)
        signals[np.abs(signals) < self.signal_threshold] = 0.0

        return signals.astype(np.float64)

    def get_confidence(self, model_output) -> np.ndarray:
        """
        Returns confidence score [0, 1] for each bar.

        Higher values mean the model is more certain about direction.
        """
        h = self.primary_horizon_idx
        dir_logits = model_output.direction.detach().cpu().numpy()[:, h]
        p_up = 1.0 / (1.0 + np.exp(-dir_logits))
        return 2.0 * np.abs(p_up - 0.5)

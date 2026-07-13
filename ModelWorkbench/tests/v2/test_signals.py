"""
test_signals.py — Tests for signal generation and position sizing.
"""

import numpy as np
import torch
import pytest
from dataclasses import dataclass

from Learn.v2.signals import DistributionalSignalGenerator
from Learn.v2.position_sizing import KellyPositionSizer
from Learn.v2.risk_manager import RiskManager, RiskConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Mock ModelOutput — matches actual ModelOutput dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MockModelOutput:
    """Mock of ModelOutput matching actual fields from model/full_model.py."""
    distribution: tuple      # (mu: Tensor(B,H), log_sigma: Tensor(B,H))
    direction: torch.Tensor  # (B, n_horizons) — binary logits for P(return > 0)
    regime: torch.Tensor     # (B, n_regimes) — multiclass logits
    volatility: torch.Tensor = None
    quantiles: torch.Tensor = None


def make_mock_output(mu, log_sigma, dir_logits, regime_logits=None):
    """Create a mock ModelOutput matching the actual ModelOutput format.

    Args:
        mu: (B,) or (B, n_horizons)
        log_sigma: same shape as mu
        dir_logits: (B,) or (B, n_horizons) — binary P(up) logits
        regime_logits: (B, n_regimes) or None (zeros)
    """
    mu = torch.as_tensor(mu, dtype=torch.float32)
    log_sigma = torch.as_tensor(log_sigma, dtype=torch.float32)
    if mu.ndim == 1:
        mu = mu.unsqueeze(1)
        log_sigma = log_sigma.unsqueeze(1)
    dir_logits = torch.as_tensor(dir_logits, dtype=torch.float32)
    if dir_logits.ndim == 1:
        dir_logits = dir_logits.unsqueeze(1)

    batch_size = mu.shape[0]
    if regime_logits is None:
        regime_logits = torch.zeros(batch_size, 4)
    else:
        regime_logits = torch.as_tensor(regime_logits, dtype=torch.float32)

    return MockModelOutput(
        distribution=(mu, log_sigma),
        direction=dir_logits,
        regime=regime_logits,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DistributionalSignalGenerator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDistributionalSignalGenerator:

    def test_signal_range(self):
        """Signals should be in [-1, 1]."""
        gen = DistributionalSignalGenerator(temperature=1.0, primary_horizon_idx=0)
        mu = torch.tensor([[1.0, 0.5, 2.0, -0.5, -1.0]], dtype=torch.float32)
        log_sigma = torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0]], dtype=torch.float32)
        dir_logits = torch.tensor([[2.0, 1.0, 3.0, -1.0, -2.0]], dtype=torch.float32)

        output = make_mock_output(mu, log_sigma, dir_logits)
        signals = gen.generate(output)
        assert signals.shape == (1,)
        assert np.all(signals >= -1.0) and np.all(signals <= 1.0)

    def test_signal_zeros_with_high_uncertainty(self):
        """When uncertainty is high (large sigma) and mu near zero, signal → 0."""
        gen = DistributionalSignalGenerator(
            temperature=1.0,
            signal_threshold=0.09,
            primary_horizon_idx=0,
        )
        mu = torch.tensor([[0.0]], dtype=torch.float32)
        log_sigma = torch.tensor([[3.0]], dtype=torch.float32)
        dir_logits = torch.tensor([[0.0]], dtype=torch.float32)

        output = make_mock_output(mu, log_sigma, dir_logits)
        signals = gen.generate(output)
        assert signals[0] == 0.0, f"Expected 0 signal, got {signals[0]}"

    def test_signal_sign_matches_expected_return(self):
        """Positive expected return → positive signal; negative → negative."""
        gen = DistributionalSignalGenerator(
            temperature=1.0, signal_threshold=0.0, primary_horizon_idx=0,
        )
        # Strong positive
        mu = torch.tensor([[5.0]], dtype=torch.float32)
        log_sigma = torch.tensor([[0.0]], dtype=torch.float32)
        dir_logits = torch.tensor([[3.0]], dtype=torch.float32)  # P(up) ≈ 0.95
        output = make_mock_output(mu, log_sigma, dir_logits)
        assert gen.generate(output)[0] > 0

        # Strong negative
        mu2 = torch.tensor([[-5.0]], dtype=torch.float32)
        log_sigma2 = torch.tensor([[0.0]], dtype=torch.float32)
        dir_logits2 = torch.tensor([[-3.0]], dtype=torch.float32)
        output2 = make_mock_output(mu2, log_sigma2, dir_logits2)
        assert gen.generate(output2)[0] < 0

    def test_extreme_regime_gate(self):
        """When regime is extreme, signal is zeroed."""
        gen = DistributionalSignalGenerator(
            temperature=1.0, signal_threshold=0.0,
            extreme_regime_gate=True, regime_idx=3, primary_horizon_idx=0,
        )
        mu = torch.tensor([[5.0]], dtype=torch.float32)
        log_sigma = torch.tensor([[0.0]], dtype=torch.float32)
        dir_logits = torch.tensor([[3.0]], dtype=torch.float32)
        regime_logits = torch.tensor([[0.0, 0.0, 0.0, 10.0]], dtype=torch.float32)

        output = make_mock_output(mu, log_sigma, dir_logits, regime_logits=regime_logits)
        assert gen.generate(output)[0] == 0.0

    def test_signal_threshold(self):
        """Weak signals below threshold are zeroed."""
        gen = DistributionalSignalGenerator(
            temperature=10.0, signal_threshold=0.5, primary_horizon_idx=0,
        )
        mu = torch.tensor([[0.1]], dtype=torch.float32)
        log_sigma = torch.tensor([[0.0]], dtype=torch.float32)
        dir_logits = torch.tensor([[0.5]], dtype=torch.float32)

        output = make_mock_output(mu, log_sigma, dir_logits)
        assert gen.generate(output)[0] == 0.0

    def test_multi_horizon_signals(self):
        """Multi-horizon signals have correct shape."""
        gen = DistributionalSignalGenerator()
        batch, horizons = 4, 5
        mu = torch.randn(batch, horizons)
        log_sigma = torch.randn(batch, horizons) * 0.5
        dir_logits = torch.randn(batch, horizons)

        output = make_mock_output(mu, log_sigma, dir_logits)
        signals = gen.generate_multi_horizon(output)
        assert signals.shape == (batch, horizons)
        assert np.all(signals >= -1.0) and np.all(signals <= 1.0)

    def test_confidence_scores(self):
        """Confidence scores in [0, 1]."""
        gen = DistributionalSignalGenerator(primary_horizon_idx=0)
        dir_logits = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        output = make_mock_output(
            torch.randn(1, 5), torch.randn(1, 5), dir_logits,
        )
        confidence = gen.get_confidence(output)
        assert confidence.shape == (1,)
        assert np.all(confidence >= 0) and np.all(confidence <= 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# KellyPositionSizer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestKellyPositionSizer:

    def test_kelly_fraction_bounded(self):
        """Kelly fraction should be in [0, max_position_pct]."""
        sizer = KellyPositionSizer(max_position_pct=0.05, half_kelly=False)
        # Good odds
        size = sizer.compute_size(0.6, 2.0, 1.0, 10000.0)
        assert 0 <= size <= 500.0

    def test_kelly_zero_edge_zero_position(self):
        """With no edge, position size should be zero."""
        sizer = KellyPositionSizer(max_position_pct=0.1, half_kelly=False)
        size = sizer.compute_size(0.5, 1.0, 1.0, 10000.0)
        assert size == 0.0

    def test_kelly_half_kelly(self):
        """Half-Kelly should be approximately half the full-Kelly size."""
        full = KellyPositionSizer(max_position_pct=1.0, half_kelly=False)
        half = KellyPositionSizer(max_position_pct=1.0, half_kelly=True)
        size_full = full.compute_size(0.55, 1.5, 1.0, 10000.0)
        size_half = half.compute_size(0.55, 1.5, 1.0, 10000.0)
        assert abs(size_half * 2.0 - size_full) < 1.0

    def test_kelly_invalid_inputs(self):
        """Kelly should return 0 for invalid inputs."""
        sizer = KellyPositionSizer()
        assert sizer.compute_size(-0.1, 1.0, 1.0, 10000.0) == 0.0
        assert sizer.compute_size(1.5, 1.0, 1.0, 10000.0) == 0.0
        assert sizer.compute_size(0.5, -1.0, 1.0, 10000.0) == 0.0

    def test_kelly_batch_compute(self):
        """Batch computation returns correct shape."""
        sizer = KellyPositionSizer()
        sizes = sizer.batch_compute(
            np.array([0.6, 0.3, 0.5]),
            np.array([2.0, 1.0, 1.0]),
            np.array([1.0, 2.0, 1.0]),
            10000.0,
        )
        assert sizes.shape == (3,)

    def test_kelly_exposure_respected(self):
        """Position size respects current exposure."""
        sizer = KellyPositionSizer(max_position_pct=0.05, half_kelly=False)
        # Good edge but already at max exposure
        size = sizer.compute_size(0.7, 3.0, 1.0, 10000.0, current_exposure=500.0)
        assert size == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# RiskManager Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskManager:

    def test_stop_loss_below_entry_for_long(self):
        """Long stop loss should be below entry."""
        rm = RiskManager(RiskConfig(trailing_stop_atr_mult=1.5))
        tp, sl, trail = rm.compute_exit_levels(100.0, 1, 2.0, 10000.0)
        assert sl < 100.0

    def test_stop_loss_above_entry_for_short(self):
        """Short stop loss should be above entry."""
        rm = RiskManager(RiskConfig(trailing_stop_atr_mult=1.5))
        tp, sl, trail = rm.compute_exit_levels(100.0, -1, 2.0, 10000.0)
        assert sl > 100.0

    def test_risk_manager_exposure_limit(self):
        """Risk manager enforces total exposure limit."""
        config = RiskConfig(max_total_exposure_pct=0.10)
        rm = RiskManager(config)
        positions = [{"size": 800.0}]  # 8% of 10000
        assert rm.check_entry_allowed(10000.0, positions)
        positions = [{"size": 1200.0}]  # 12% — over limit
        assert not rm.check_entry_allowed(10000.0, positions)

    def test_risk_manager_position_limit(self):
        """Risk manager enforces max concurrent positions."""
        config = RiskConfig(max_concurrent_positions=2)
        rm = RiskManager(config)
        positions = [{"size": 100.0}]
        assert rm.check_entry_allowed(10000.0, positions)
        positions = [{"size": 100.0}, {"size": 200.0}]
        assert not rm.check_entry_allowed(10000.0, positions)

    def test_trailing_stop_only_moves_favorably(self):
        """Trailing stop only moves in profitable direction for long."""
        rm = RiskManager(RiskConfig(trailing_stop_atr_mult=1.5))
        position = {
            "direction": 1,
            "entry_price": 100.0,
            "stop_loss": 97.0,
            "trailing_sl": 97.0,
        }
        # Price moved up — trailing stop should move up
        sl1 = rm.update_trailing_stop(position, {"High": 103.0, "Low": 101.0})
        assert sl1 >= 97.0

        # Price moved further — trailing stop continues up
        sl2 = rm.update_trailing_stop(position, {"High": 105.0, "Low": 103.0})
        assert sl2 >= sl1

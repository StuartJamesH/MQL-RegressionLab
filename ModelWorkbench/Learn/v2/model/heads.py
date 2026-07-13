"""
Prediction heads for the Causal Patch Transformer.

DistributionHead  -> Gaussian (mu, log_sigma) per horizon
QuantileHead      -> Five quantile estimates per horizon
DirectionHead     -> Binary logits for P(return > 0) per horizon
VolatilityHead    -> Log-volatility estimate per horizon
RegimeHead        -> Categorical regime classification
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp_head(d_model: int, hidden: int, out: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [nn.Linear(d_model, hidden), nn.ReLU(inplace=True)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out))
    return nn.Sequential(*layers)


class DistributionHead(nn.Module):
    """Gaussian (mu, log_sigma) per horizon."""

    def __init__(self, d_model: int, n_horizons: int = 6, hidden_size: int = 128, dropout: float = 0.0):
        super().__init__()
        self.nh = n_horizons
        self.mlp = _mlp_head(d_model, hidden_size, n_horizons * 2, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(x)
        mu = out[:, :self.nh]
        log_sigma = F.softplus(out[:, self.nh:]) + 1e-6
        return mu, log_sigma

    def extra_repr(self) -> str:
        return f"d_model -> {self.mlp[0].out_features} -> {self.nh} x 2"


class DirectionHead(nn.Module):
    """Binary logits P(return > 0) per horizon."""

    def __init__(self, d_model: int, n_horizons: int = 6, hidden_size: int = 64, dropout: float = 0.0):
        super().__init__()
        self.nh = n_horizons
        self.mlp = _mlp_head(d_model, hidden_size, n_horizons, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def extra_repr(self) -> str:
        return f"d_model -> {self.mlp[0].out_features} -> {self.nh}"


class VolatilityHead(nn.Module):
    """Log-volatility per horizon."""

    def __init__(self, d_model: int, n_horizons: int = 6, hidden_size: int = 64, dropout: float = 0.0):
        super().__init__()
        self.nh = n_horizons
        self.mlp = _mlp_head(d_model, hidden_size, n_horizons, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def extra_repr(self) -> str:
        return f"d_model -> {self.mlp[0].out_features} -> {self.nh}"


class RegimeHead(nn.Module):
    """Volatility regime classification."""

    def __init__(self, d_model: int, n_regimes: int = 4, hidden_size: int = 32, dropout: float = 0.0):
        super().__init__()
        self.nr = n_regimes
        self.mlp = _mlp_head(d_model, hidden_size, n_regimes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def extra_repr(self) -> str:
        return f"d_model -> {self.mlp[0].out_features} -> {self.nr}"


class QuantileHead(nn.Module):
    """Multi-quantile estimates per horizon for pinball loss."""

    def __init__(self, d_model: int, n_horizons: int = 6, n_quantiles: int = 5,
                 hidden_size: int = 128, dropout: float = 0.0):
        super().__init__()
        self.nh = n_horizons; self.nq = n_quantiles
        self.mlp = _mlp_head(d_model, hidden_size, n_horizons * n_quantiles, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).view(-1, self.nh, self.nq)

    def extra_repr(self) -> str:
        return f"d_model -> {self.mlp[0].out_features} -> {self.nh} x {self.nq}"

"""
TradeForecastTransformer — complete Causal Patch Transformer for multi-horizon
price forecasting with distributional, directional, volatility, and regime heads.

Composes: PatchEmbedding -> (optional MTFFusion) -> CausalTransformerEncoder -> heads.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from Learn.v2.model.config import ModelConfig
from Learn.v2.model.embedding import PatchEmbedding, TimeframeEmbedding
from Learn.v2.model.transformer import CausalTransformerEncoder
from Learn.v2.model.heads import (
    DistributionHead, DirectionHead, VolatilityHead, RegimeHead, QuantileHead,
)
from Learn.v2.model.mtf_fusion import MTFFusionModule


@dataclass
class ModelOutput:
    """Structured output from TradeForecastTransformer.forward()."""
    distribution: Optional[tuple[torch.Tensor, torch.Tensor]]
    direction: torch.Tensor
    volatility: torch.Tensor
    regime: torch.Tensor
    quantiles: Optional[torch.Tensor]


class TradeForecastTransformer(nn.Module):
    """Causal Patch Transformer for multi-horizon structured return prediction.

    Architecture
    ------------
    1. PatchEmbedding: (B, seq_len, 9) -> (B, n_patches+1, d_model) with [CLS] at pos 0.
    2. MTF Fusion (optional): cross-attention from base to HTF patches.
    3. CausalTransformerEncoder: Pre-LN causal blocks with SwiGLU FFN.
    4. Heads (from CLS token): distribution, direction, volatility, regime, quantiles.

    Parameter count ~ 8-12 M for default config.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbedding(
            seq_len=config.max_seq_len, in_channels=config.in_channels,
            session_channels=config.session_channels, patch_len=config.patch_len,
            patch_stride=config.patch_stride, d_model=config.d_model, dropout=config.dropout,
        )

        if config.use_mtf_fusion:
            self.mtf_fusion = MTFFusionModule(d_model=config.d_model, n_heads=4, dropout=0.1)
            self.timeframe_embed = TimeframeEmbedding(n_timeframes=config.n_timeframes, d_model=config.d_model)
        else:
            self.mtf_fusion = None
            self.timeframe_embed = None

        self.encoder = CausalTransformerEncoder(
            d_model=config.d_model, n_layers=config.n_layers,
            n_heads=config.n_heads, d_ff=config.d_ff, dropout=config.dropout,
        )

        if config.use_quantile_head:
            self.distribution_head = None
            self.quantile_head = QuantileHead(
                d_model=config.d_model, n_horizons=config.n_horizons,
                n_quantiles=config.n_quantiles, hidden_size=128, dropout=config.dropout,
            )
        else:
            self.distribution_head = DistributionHead(
                d_model=config.d_model, n_horizons=config.n_horizons,
                hidden_size=128, dropout=config.dropout,
            )
            self.quantile_head = None

        self.direction_head = DirectionHead(d_model=config.d_model, n_horizons=config.n_horizons, hidden_size=64, dropout=config.dropout)
        self.volatility_head = VolatilityHead(d_model=config.d_model, n_horizons=config.n_horizons, hidden_size=64, dropout=config.dropout)
        self.regime_head = RegimeHead(d_model=config.d_model, n_regimes=config.n_regimes, hidden_size=32, dropout=config.dropout)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def forward(self, x_raw: torch.Tensor, x_session: Optional[torch.Tensor] = None,
                htf_embeddings: Optional[torch.Tensor] = None,
                timeframe_idx: Optional[torch.Tensor] = None) -> ModelOutput:
        x = self.patch_embed(x_raw, x_session)
        if self.mtf_fusion is not None and htf_embeddings is not None:
            if self.timeframe_embed is not None and timeframe_idx is not None:
                x = x + self.timeframe_embed(timeframe_idx).unsqueeze(1)
            x = self.mtf_fusion(x, htf_embeddings)
        x = self.encoder(x)
        cls = x[:, 0, :]
        if self.distribution_head is not None:
            mu, log_sigma = self.distribution_head(cls)
            quantiles = None
        else:
            mu, log_sigma = None, None
            quantiles = self.quantile_head(cls) if self.quantile_head is not None else None
        return ModelOutput(
            distribution=(mu, log_sigma) if mu is not None else None,
            direction=self.direction_head(cls), volatility=self.volatility_head(cls),
            regime=self.regime_head(cls), quantiles=quantiles,
        )

    def forward_features(self, x_raw: torch.Tensor, x_session: Optional[torch.Tensor] = None,
                         htf_embeddings: Optional[torch.Tensor] = None,
                         timeframe_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.patch_embed(x_raw, x_session)
        if self.mtf_fusion is not None and htf_embeddings is not None:
            if self.timeframe_embed is not None and timeframe_idx is not None:
                x = x + self.timeframe_embed(timeframe_idx).unsqueeze(1)
            x = self.mtf_fusion(x, htf_embeddings)
        return self.encoder(x)[:, 0, :]

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        nM = self.num_parameters(True) / 1_000_000
        return (f"d_model={self.config.d_model}, n_layers={self.config.n_layers}, "
                f"n_heads={self.config.n_heads}, d_ff={self.config.d_ff}, "
                f"patch_len={self.config.patch_len}, n_horizons={self.config.n_horizons}, "
                f"n_regimes={self.config.n_regimes}, "
                f"use_mtf={self.mtf_fusion is not None}, "
                f"use_quantile={self.quantile_head is not None}, "
                f"params={nM:.1f}M")

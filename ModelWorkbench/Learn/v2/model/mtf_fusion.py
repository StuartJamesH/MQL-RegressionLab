"""Multi-Timeframe (MTF) fusion via cross-attention."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


class MTFFusionModule(nn.Module):
    """Cross-attention: base queries <-> HTF keys/values."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self._dm = d_model; self._nh = n_heads

    def forward(self, base_tokens: torch.Tensor, htf_tokens: torch.Tensor,
                htf_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = base_tokens
        attn_out, _ = self.cross_attn(self.norm(base_tokens), htf_tokens, htf_tokens,
                                       attn_mask=htf_mask, need_weights=False)
        return residual + self.dropout(attn_out)

    def extra_repr(self) -> str:
        return f"d_model={self._dm}, n_heads={self._nh}, dropout={self.dropout.p}"

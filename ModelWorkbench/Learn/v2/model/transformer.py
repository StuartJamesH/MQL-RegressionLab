"""
Causal transformer encoder with SwiGLU feed-forward blocks.

CLS at position 0 attends to all; patch tokens are left-to-right causal.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU: SiLU(gate(x)) * value(x) -> project."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=True)
        self.value = nn.Linear(d_model, d_ff, bias=True)
        self.proj = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.silu(self.gate(x)) * self.value(x)))

    def extra_repr(self) -> str:
        return f"d_model={self.gate.in_features}, d_ff={self.gate.out_features}, dropout={self.dropout.p}"


class TransformerBlock(nn.Module):
    """Pre-LN causal block with SwiGLU FFN. CLS sees all, patches are causal."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self._d = d_model; self._h = n_heads

    @staticmethod
    def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        mask[0, :] = False  # CLS sees all
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_n = self.norm1(x)
        mask = self._create_causal_mask(x_n.size(1), x.device)
        attn_out, _ = self.attn(x_n, x_n, x_n, attn_mask=mask, need_weights=False)
        x = residual + self.dropout1(attn_out)
        residual = x
        x = residual + self.dropout2(self.ffn(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"d_model={self._d}, n_heads={self._h}, d_ff={self.ffn.gate.out_features}"


class CausalTransformerEncoder(nn.Module):
    """Stack of TransformerBlock instances."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self._nl = n_layers; self._dm = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x

    def extra_repr(self) -> str:
        return f"n_layers={self._nl}, d_model={self._dm}"

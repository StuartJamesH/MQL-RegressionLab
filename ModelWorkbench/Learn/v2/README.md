# Learn/v2 — Causal Patch Transformer for Trading

> **Status:** Implemented on branch `train-v2` | **Tests:** 58/58 passing | **Model params:** 8.55M

## Overview

A complete from-scratch rearchitecture of the MQL-RegressionLab ML pipeline. Replaces the previous LightGBM-on-static-features approach with a **sequence-to-distribution causal transformer** that processes raw OHLCV data through learned patch embeddings, predicts full conditional return distributions at multiple horizons, and is trained via a three-phase curriculum:

1. **Phase 1 — Self-supervised pre-training**: MAE-style masked patch reconstruction across all instruments
2. **Phase 2 — Distributional fine-tuning**: Multi-task learning (distribution + direction + volatility + regime)
3. **Phase 3 — RL optimization**: REINFORCE with baseline, freezing bottom layers to preserve representations

## Architecture

```
Raw OHLCV (B, 512, 5) + Session features (B, 512, 4)
    │
    ▼
PatchEmbedding (Conv1d, patch_len=16, stride=8) + [CLS] token
    │
    ▼
CausalTransformerEncoder (8 layers, pre-LN, SwiGLU FFN, causal MHA)
    │
    ├──► CLS token ──► DistributionHead   → (μ, log σ) per horizon
    │              ──► DirectionHead       → P(return > 0) per horizon
    │              ──► VolatilityHead      → realized vol per horizon
    │              ──► RegimeHead          → volatility regime class
    ▼
ModelOutput
```

## Quick Start

### Prerequisites

```bash
# Create venv and install (from repo root)
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dependencies: PyTorch ≥ 2.1, numba, h5py, onnx, onnxruntime, pandas, numpy, scipy, scikit-learn, TA-Lib.

### Training

```bash
# From repo root, with ModelWorkbench as CWD
cd ModelWorkbench

# Quick test with limited data
python train_transformer.py \
    --ds-names ../data/XAUUSD_M1_520weeks.csv \
    --n-rows 100000 \
    --seq-len 256 \
    --finetune-epochs 20 \
    --batch-size 32 \
    --device cpu \
    --output-dir ModelPacks/transformers
```

Arguments:
| Flag | Default | Description |
|------|---------|-------------|
| `--ds-names` | *(required)* | One or more CSV dataset paths |
| `--pretrain-steps` | 10000 | Self-supervised pretraining steps |
| `--finetune-epochs` | 50 | Distributional fine-tuning epochs |
| `--rl-steps` | 5000 | RL policy fine-tuning steps |
| `--output-dir` | `ModelPacks/transformers` | Output directory |
| `--device` | `cuda` | `cuda`, `cpu`, or `mps` |
| `--batch-size` | 64 | Training batch size |
| `--lr` | 3e-4 | Learning rate |
| `--no-pretrain` | *(off)* | Skip self-supervised pretraining |
| `--no-rl` | *(off)* | Skip RL fine-tuning |
| `--seq-len` | 512 | Input sequence length in bars |
| `--n-rows` | 500000 | Max rows per dataset |

### Running Tests

```bash
cd ModelWorkbench
python -m pytest tests/v2/ -v
```

## Package Structure

```
Learn/v2/
├── labels.py          # Distributional label computation (Numba-accelerated)
├── data.py            # OHLCV normalization, sliding windows, session encoding
├── signals.py         # Signal generator (μ/σ → trade signal)
├── position_sizing.py # Kelly criterion position sizing
├── risk_manager.py    # Risk limits, TP/SL levels, trailing stops
├── signal_evaluator.py
├── backtest.py        # Vectorized backtester with spread/commission
├── backtest_metrics.py
├── walk_forward_backtest.py
├── deploy.py          # ONNX export + deployment packaging
├── feature_spec.py
├── parity_check.py    # Python vs MQL5 ONNX output comparison
├── model/
│   ├── config.py      # ModelConfig dataclass
│   ├── embedding.py   # PatchEmbedding, TimeframeEmbedding
│   ├── transformer.py # SwiGLU FFN, TransformerBlock, CausalTransformerEncoder
│   ├── heads.py       # Distribution, Direction, Volatility, Regime, Quantile heads
│   ├── full_model.py  # TradeForecastTransformer, ModelOutput
│   ├── mtf_fusion.py  # Multi-timeframe cross-attention
│   └── export.py      # ONNX export
└── training/
    ├── losses.py      # Gaussian NLL, pinball, quantile, focal, composite
    ├── pretrain_data.py
    ├── pretrain.py    # MAE masked patch reconstruction
    ├── finetune.py    # Distributional fine-tuning
    ├── rl_finetune.py # REINFORCE policy optimization
    ├── metrics.py     # Training tracker (W&B/TensorBoard)
    └── folds.py       # Purged walk-forward splits
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Patch embedding** (not raw bars) | Conv1d over 16-bar windows captures local patterns; 50% overlap provides redundancy |
| **Causal GPT-style decoder** | Pre-LN is more stable than post-LN; causal mask prevents lookahead |
| **SwiGLU FFN** | Better gradient flow than GELU/ReLU; modern transformer standard |
| **Distribution + quantile heads** | Gaussian NLL for parametric uncertainty; pinball loss for non-parametric fat-tailed distributions |
| **[CLS] attends to full sequence** | Patches are causally masked, but CLS pools global context for multi-horizon predictions |
| **Freeze bottom layers during RL** | Preserves pretrained representations; REINFORCE only tunes top 2 layers |
| **Numba `@njit(cache=True)` labels** | O(n×h) forward scan computes MFE/MAE at all horizons simultaneously |

## Production Deployment

```bash
# Training produces a deployment pack at ModelPacks/transformers/<name>/
# Contains:
#   model.onnx        — ONNX model for MQL5 inference
#   config.json       — Architecture + hyperparameter config
#   normalizer.json   — Input normalization stats
#   feature_spec.json — Feature computation specification
#   model.pt          — Full PyTorch checkpoint
```

The ONNX model is consumed by `MQL5/Indicators/TransformerModel.mq5` for live inference in MetaTrader 5.

## Relationship to Existing Code

- **`Learn/`** (v1) — LightGBM + static features pipeline. **Unchanged.**
- **`Learn/v2/`** — New transformer pipeline. **No dependency on v1 code.**
- All v2 code lives under `ModelWorkbench/Learn/v2/` and `ModelWorkbench/tests/v2/`.

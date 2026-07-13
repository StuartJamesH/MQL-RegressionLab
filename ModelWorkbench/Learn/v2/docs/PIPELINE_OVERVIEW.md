# V2 Pipeline Overview — Mental Maps & End-to-End Flow

> **Module:** `ModelWorkbench/Learn/v2/` | **Model:** Causal Patch Transformer (~8.5M params)  
> **Training entry point:** `ModelWorkbench/train_transformer.py`

---

## End-to-End Mental Map

This is the 80/20 view of the entire pipeline. Each numbered stage represents a discrete
transformation of data as it flows from raw CSV files to live trading signals.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        V2 PIPELINE — BIRD'S EYE VIEW                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ╔═══════════════════╗                                                          │
│  ║ ① RAW DATA        ║  Multiple OHLCV CSV files, each representing one        │
│  ║    INGESTION       ║  symbol×timeframe (e.g. BTCUSD_M5.csv, XAUUSD_H1.csv)  │
│  ╚══════╤════════════╝                                                          │
│         │                                                                        │
│         ▼                                                                        │
│  ╔═══════════════════╗                                                          │
│  ║ ② NORMALIZATION   ║  OHLCV → log-ratio pricing (causal: uses previous bar's  │
│  ║    (data.py)      ║  Close). Volume → rolling-median scaled.                 │
│  ║                   ║  Session time → cyclical sin/cos encoding (hour + dow).  │
│  ║                   ║  Output: (n_bars, 9) float32 matrix                      │
│  ╚══════╤════════════╝  [O,H,L,C,V | hour_sin, hour_cos, dow_sin, dow_cos]     │
│         │                                                                        │
│         ▼                                                                        │
│  ╔═══════════════════╗                                                          │
│  ║ ③ LABEL           ║  Computed STRICTLY CAUSALLY (no future leakage).         │
│  ║    ENGINEERING     ║  Returns look FORWARD from bar t → t+h.                 │
│  ║    (labels.py)     ║  Outputs: excursion surface, direction labels,          │
│  ║                   ║  optimal-exit outcomes, volatility regime classes.       │
│  ╚══════╤════════════╝  Cached in HDF5 via LabelStore.                          │
│         │                                                                        │
│         ▼                                                                        │
│  ╔═══════════════════╗                                                          │
│  ║ ④ SLIDING WINDOW  ║  (n_bars, 9) + label arrays → overlapping windows.       │
│  ║    ASSEMBLY       ║  Each window: seq_len bars of features + label at bar    │
│  ║    (data.py)      ║  seq_len-1. Numba-compiled for speed.                    │
│  ║                   ║  Output X: (n_windows, seq_len, n_channels)              │
│  ╚══════╤════════════╝  Output y: (n_windows, n_horizons)                       │
│         │                                                                        │
│         ▼                                                                        │
│  ╔═══════════════════╗                                                          │
│  ║ ⑤ PATCH EMBEDDING ║  Conv1d projects 16-bar patches (stride=8, 50% overlap)  │
│  ║    (embedding.py)  ║  into d_model=256 space. Adds learned position encoding │
│  ║                   ║  + [CLS] token at position 0.                            │
│  ║                   ║  Output: (B, n_patches+1, d_model)                       │
│  ╚══════╤════════════╝                                                          │
│         │                                                                        │
│         ▼                                                                        │
│  ╔═══════════════════╗                                                          │
│  ║ ⑥ CAUSAL TRANS-   ║  8-layer transformer encoder blocks (pre-LN).            │
│  ║    FORMER ENCODER  ║  [CLS] attends to ALL patches (full context).            │
│  ║    (transformer.py)║  Patch tokens are CAUSALLY masked (left-to-right).       │
│  ║                   ║  SwiGLU FFN (SiLU gating × value projection).            │
│  ╚══════╤════════════╝  Output: (B, n_patches+1, d_model)                       │
│         │                                                                        │
│         ▼                                                                        │
│  ╔═══════════════════╗                                                          │
│  ║ ⑦ PREDICTION HEADS║  From [CLS] token (global summary of sequence):         │
│  ║    (heads.py)      │  ├─ DistributionHead → (μ, log σ) per horizon [6HL]     │
│  ║                   │  ├─ DirectionHead   → P(return>0) per horizon           │
│  ║                   │  ├─ VolatilityHead  → log-vol per horizon               │
│  ║                   │  ├─ RegimeHead      → 4-class vol regime                │
│  ╚══════╤════════════╝  └─ QuantileHead    → 5 quantile levels (optional)      │
│         │                                                                        │
│         ▼                                                                        │
│  ╔═══════════════════╗                                                          │
│  ║ ⑧ SIGNAL           ║  ModelOutput → scalar trade signal in [-1, 1].          │
│  ║    GENERATION      ║  s = μ/σ → sign(s)·tanh(|s|·confidence/temperature).    │
│  ║    (signals.py)    ║  Regime gate: zero signal in extreme volatility.         │
│  ║                   ║  Threshold gate: zero weak signals below cutoff.         │
│  ╚══════╤════════════╝                                                          │
│         │                                                                        │
│         ▼                                                                        │
│  ╔═══════════════════╗                                                          │
│  ║ ⑨ TRADE EXECUTION ║  Kelly position sizing → RiskManager checks →            │
│  ║    & BACKTEST      ║  VectorizedBacktester (TP/SL with ATR, spread, comm).   │
│  ║    (position_sizing║  WalkForwardBacktest: train-on-past, test-on-future.    │
│  ║     risk_manager,  ║  Metrics: Sharpe, Sortino, max DD, Monte Carlo CI.      │
│  ║     backtest, wf)  ║                                                         │
│  ╚══════╤════════════╝                                                          │
│         │                                                                        │
│         ▼                                                                        │
│  ╔═══════════════════╗                                                          │
│  ║ ⑩ DEPLOYMENT      ║  PyTorch → ONNX export → MQL5 Indicator integration.     │
│  ║    (deploy.py,     ║  Python↔MQL5 parity checking via parity_check.py.        │
│  ║     export.py)     ║  Package: model.onnx + config.json + normalizer.json     │
│  ╚═══════════════════╝  + feature_spec.json + model.pt checkpoint.              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Three-Phase Training Curriculum

The model is trained progressively — each phase builds on the previous one:

```
Phase 1: SELF-SUPERVISED PRETRAINING       Phase 2: DISTRIBUTIONAL FINETUNING
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│ pretrain.py / pretrain_data.py  │       │ finetune.py + losses.py         │
│                                 │       │                                 │
│  MAE-style masked patch         │  ───► │  Supervised on labeled returns  │
│  reconstruction across ALL      │       │  Multi-task composite loss:     │
│  instruments simultaneously.    │       │  Gaussian NLL   (weight 1.0)    │
│                                 │       │  Direction BCE  (weight 0.3)    │
│  Mask 50% of patches randomly.  │       │  Volatility MSE (weight 0.2)    │
│  Learn price structure from     │       │  Regime CE      (weight 0.1)    │
│  unlabeled data.                │       │                                 │
│                                 │       │  Cosine LR w/ warmup.           │
│  Output: pre-trained encoder.   │       │  Spearman on held-out val.      │
└─────────────────────────────────┘       └──────────────┬──────────────────┘
                                                         │
                                          Phase 3: RL POLICY OPTIMIZATION
                                                         ▼
                                          ┌─────────────────────────────────┐
                                          │ rl_finetune.py                  │
                                          │                                 │
                                          │  REINFORCE with baseline.       │
                                          │  Freeze bottom 2 transformer     │
                                          │  layers; tune top layers+heads. │
                                          │                                 │
                                          │  Action space: HOLD/BUY/SELL.   │
                                          │  Reward: trade P&L quality.     │
                                          │  Entropy bonus for exploration. │
                                          │  Aux supervised loss anchor.    │
                                          └─────────────────────────────────┘
```

---

## Causal Guarantee (No Lookahead)

Every operation across the pipeline enforces strict causality. At bar `t`:

| Operation | What it sees | What it does NOT see |
|-----------|-------------|---------------------|
| **Normalization** | Close[t-1] for log-ratio pricing | Close[t] or anything later |
| **Session features** | Timestamp[t] (hour, day of week) | Nothing forward |
| **Feature windows** | Bars [t-seq_len+1 … t] | Bars t+1 onward |
| **Labels (targets)** | Bars [t+1 … t+max_horizon] | These ARE the prediction target — but the model at time t never sees them |
| **Purged walk-forward** | Gap of `gap_size` bars excludes test-bar labels from training | Prevents label leakage through triple-barrier windows |
| **Transformer attention** | [CLS] sees all; patches are causally masked | Future patches cannot attend to past patches |

---

## Dataset Handling: Single vs Multi-Instrument

The pipeline has two ingestion paths:

| Path | File | Use Case |
|------|------|----------|
| **Single-dataset (fine-tuning)** | `train_transformer.py` → `prepare_ohlcv_windows()` | Phase 2/3: one CSV loaded, normalized, windowed, split into train/val |
| **Multi-instrument (pre-training)** | `training/pretrain_data.py` → `MultiInstrumentDataset` | Phase 1: multiple CSVs loaded simultaneously, each with its own normalization, sampled proportionally to dataset size |

The multi-instrument dataset:
- Loads each CSV independently
- Normalizes each instrument with its OWN rolling-ATR normalization (cross-instrument comparability)
- Infers a coarse timeframe ID from the filename (M1→0, M5→1, H1→4, D1→6, etc.)
- Builds flat window indices with sample weights proportional to instrument size
- Returns tuples of `(features, features_clone, mask, instrument_id, timeframe_id)`

---

## Key Configuration Parameters

From `model/config.py` — the single source of truth:

```
d_model       = 256    # Transformer hidden dimension
n_layers      = 8      # Encoder blocks
n_heads       = 8      # Attention heads per block
d_ff          = 1024   # FFN inner dimension (SwiGLU)
dropout       = 0.1    # Regularization
patch_len     = 16     # Conv1d kernel size
patch_stride  = 8      # 50% overlap
max_seq_len   = 512    # Max input bars
n_horizons    = 6      # [5, 10, 20, 40, 60, 120] bars
n_regimes     = 4      # Vol regime classes (0=low → 3=extreme)
n_timeframes  = 5      # For MTF fusion (optional)
n_quantiles   = 5      # Pinball quantile levels (optional)
```

---

## Module Dependency Map

```
train_transformer.py  (CLI entry point)
  ├── Learn.v2.model.config          → ModelConfig
  ├── Learn.v2.model.full_model      → TradeForecastTransformer
  ├── Learn.v2.labels                → compute_directional_return_distribution
  │                                     compute_volatility_regime_labels
  ├── Learn.v2.data                  → normalize_ohlcv, SessionFeatureEncoder
  ├── Learn.v2.feature_spec          → FeatureSpec
  ├── Learn.v2.deploy                → DeploymentPackager
  ├── Learn.v2.training.metrics      → TrainingMetricsTracker
  └── Learn.train_utils              → load_ohlcv (shared with v1)

Learn/v2/
  ├── data.py          → normalize_ohlcv, create_sliding_windows, SessionFeatureEncoder
  ├── labels.py        → compute_* (4 functions), LabelStore (HDF5 cache)
  ├── signals.py       → DistributionalSignalGenerator
  ├── feature_spec.py  → FeatureSpec, FeatureDef
  ├── model/
  │   ├── config.py    → ModelConfig dataclass
  │   ├── embedding.py → PatchEmbedding, TimeframeEmbedding
  │   ├── transformer.py → SwiGLUFFN, TransformerBlock, CausalTransformerEncoder
  │   ├── heads.py     → DistributionHead, DirectionHead, VolatilityHead, RegimeHead, QuantileHead
  │   ├── mtf_fusion.py → MTFFusionModule (cross-attention for multi-timeframe)
  │   ├── full_model.py → TradeForecastTransformer, ModelOutput
  │   └── export.py    → export_to_onnx
  ├── training/
  │   ├── pretrain.py       → MaskedPatchPretraining + MAEDecoder
  │   ├── pretrain_data.py  → MultiInstrumentDataset
  │   ├── finetune.py       → DistributionalFinetuning
  │   ├── rl_finetune.py    → RLPolicyFinetuning
  │   ├── losses.py         → gaussian_nll, pinball, focal, composite_loss
  │   ├── metrics.py        → TrainingMetricsTracker
  │   └── folds.py          → PurgedWalkForwardSplit
  ├── backtest.py           → VectorizedBacktester, Trade
  ├── backtest_metrics.py   → BacktestMetrics
  ├── walk_forward_backtest.py → WalkForwardBacktest
  ├── position_sizing.py    → KellyPositionSizer
  ├── risk_manager.py       → RiskManager, RiskConfig
  ├── signal_evaluator.py   → SignalEvaluator
  ├── deploy.py             → DeploymentPackager
  └── parity_check.py       → Python↔MQL5 parity verification
```

---

## Training Invocation

```bash
# From repo root, cd into ModelWorkbench first
cd ModelWorkbench

# Single-dataset fine-tuning (Phase 2)
../.venv/bin/python train_transformer.py \
    --ds-names ../data/BTCUSD_M5_260weeks.csv \
    --n-rows 500000 \
    --seq-len 512 \
    --finetune-epochs 50 \
    --batch-size 64 \
    --target-type log_return

# Phase 1 pretraining uses MultiInstrumentDataset internally
# (accessed through the pretrain_data.py module, not directly from CLI)
```

---

## Key Design Decisions (Non-Math Summary)

| Decision | Rationale |
|----------|-----------|
| **Patch embedding** over raw bars | Conv1d over 16-bar windows captures local micro-patterns; stride=8 provides 50% overlap for redundancy |
| **Causal [CLS] token** | [CLS] at position 0 attends to the entire sequence; patches observe left-to-right causality — the model cannot cheat by peeking ahead |
| **SwiGLU FFN** | SiLU-gated linear units provide smoother gradient flow than plain ReLU/GELU; standard in modern transformers |
| **Multi-head prediction** | Distributional (μ,σ), directional, volatility, and regime heads share a single encoder — the shared representation prevents overfitting to any single objective |
| **Freeze bottom layers during RL** | Preserves the pretrained price-structure representations; REINFORCE only fine-tunes the top 2 layers and heads for policy optimization |
| **Numba `@njit(cache=True)`** | The O(n×h) forward scan for MFE/MAE at all horizons runs in compiled machine code — critical performance win over pure Python |
| **HDF5 label cache** | Excursion surfaces and triple-barrier outcomes are expensive to compute; caching by SHA-256 content hash avoids recomputation across training runs |

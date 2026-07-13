---
goal: "Major Revision: Next-Generation Profitable Trading Model"
version: "1.0"
date_created: "2026-07-11"
last_updated: "2026-07-11"
owner: "ModelWorkbench"
status: "Planned"
tags:
  - feature
  - architecture
  - ml
  - deep-learning
  - transformer
  - time-series
  - distributional-regression
  - self-supervised
  - reinforcement-learning
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan proposes a complete from-scratch rearchitecture of the MQL-RegressionLab ML pipeline. The current system — LightGBM gradient-boosted trees trained on 100+ hand-crafted technical indicators to predict discretized triple-barrier trade outcomes — achieves Spearman ~0.30, R² ~0.08, and sign hit rates that peak at 74% with only 5% coverage. These metrics are insufficient for profitable live trading.

The new design replaces static feature engineering, tabular tree models, and discretized labels with a **sequence-to-distribution causal transformer** that processes raw OHLCV data through learned embeddings, predicts full conditional return distributions at multiple horizons, and is trained via a three-phase curriculum: self-supervised pre-training, multi-task distributional regression, and reinforcement learning fine-tuning. The goal is to learn representations of market dynamics that the current approach fundamentally cannot capture.

## 1. Requirements & Constraints

- **REQ-001**: Must be implementable in Python with a single consumer GPU (e.g., RTX 3090/4090, 24 GB VRAM) — no HPC clusters
- **REQ-002**: All model inference must be strictly causal (no lookahead at any point in the pipeline)
- **REQ-003**: Must handle multiple instruments (XAUUSD, EURUSD, BTCUSD, etc.) and multiple timeframes (M1, M5, M15, M30, H1)
- **REQ-004**: Trained model must be exportable for MQL5 inference via ONNX or a lightweight C++ inference library
- **REQ-005**: Model update cadence must support weekly retraining with < 24 hours wall-clock time
- **REQ-006**: Trade signal generation must produce a continuous confidence/quality score, not just binary buy/sell
- **REQ-007**: The system must explicitly quantify prediction uncertainty for risk management
- **SEC-001**: No lookahead leakage in any feature, label, or preprocessing step — validated by causality tests
- **SEC-002**: Model pack must be serializable and versioned for audit trail
- **CON-001**: Training dataset limited to ~2 million rows per instrument due to cTrader API rate limits
- **CON-002**: Inference latency target: < 5 ms per bar on CPU (for MQL5 live trading)
- **GUD-001**: Follow the existing repo structure conventions (`ModelWorkbench/` for Python training, `MQL5/Indicators/` for live execution)
- **GUD-002**: Use PyTorch as the primary deep learning framework
- **GUD-003**: All new code goes into a `ModelWorkbench/Learn/v2/` package to avoid breaking the existing v1 pipeline
- **PAT-001**: Model pack export pattern from existing `train_lgbm.py` (serialize model + metadata + feature pipeline into a .pkl or .pt archive)
- **PAT-002**: Jupyter notebook exploration → CLI training script → backtest notebook workflow from existing pipeline

## 2. Implementation Steps

### Implementation Phase 1: Label Engineering — Distributional Targets

- **GOAL-001**: Replace ternary triple-barrier labels with continuous distributional targets that preserve all information about forward price paths and directly support risk-aware decision-making.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Create `ModelWorkbench/Learn/v2/labels.py` with `compute_forward_excursion_surface(df, horizons, atr_window)` — for each bar t, compute MFE (max favorable excursion) and MAE (max adverse excursion) at horizons [5, 10, 20, 40, 60, 120] bars, expressed in ATR units. Returns a (n_bars, n_horizons, 2) tensor. Implementation uses Numba-accelerated sliding window max/min for O(n×h) performance. | | |
| TASK-002 | Create `compute_directional_return_distribution(df, horizons)` — for each bar t and horizon h, compute the forward log return `ln(close[t+h] / close[t])`. Returns (n_bars, n_horizons) tensor. Use this as the primary regression target for the distribution head. | | |
| TASK-003 | Create `compute_optimal_exit_labels(df, tp_atr_mult, sl_atr_mult, max_horizon)` — retains the triple-barrier concept but returns the CONTINUOUS outcome: `+1` if TP hit first, `-1` if SL hit first, `0` if timeout, PLUS the normalized duration to exit (`exit_bar / max_horizon`) and the MFE/MAE ratio at exit. This bridges old and new label paradigms. | | |
| TASK-004 | Create `compute_volatility_regime_labels(df, lookback=20, n_regimes=4)` — use HMM (Hidden Markov Model) on rolling volatility+spread features to assign each bar to one of 4 volatility regimes (low, normal, high, extreme). Trained causally with expanding window. | | |
| TASK-005 | Create `LabelStore` HDF5-backed class that precomputes and caches all label tensors keyed by `(dataset_hash, params_hash)` to avoid recomputation across experiments. Store in `ModelWorkbench/data/labels/`. | | |
| TASK-006 | Unit tests in `ModelWorkbench/tests/v2/test_labels.py`: verify causality (no NaN rows at start of each horizon), verify MFE/MAE monotonicity with horizon, verify ATR normalization, verify label alignment with raw OHLCV. | | |

### Implementation Phase 2: Model Architecture — Causal Patch Transformer

- **GOAL-002**: Design and implement a sequence-to-distribution causal transformer that processes raw OHLCV patches, learns temporal representations, and outputs probabilistic forecasts of forward returns at multiple horizons.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-007 | Create `ModelWorkbench/Learn/v2/model/__init__.py` and `ModelWorkbench/Learn/v2/model/config.py` with `ModelConfig` dataclass: `d_model=256, n_layers=8, n_heads=8, d_ff=1024, patch_len=16, max_seq_len=512, n_horizons=6, dropout=0.1, use_flash_attn=True, n_regimes=4, n_instruments=10`. | | |
| TASK-008 | Create `ModelWorkbench/Learn/v2/model/embedding.py` with `PatchEmbedding` — takes raw OHLCV (5 channels: O, H, L, C, V) + session features (hour sin/cos, weekday sin/cos = 4 channels). Applies 1D convolution with `kernel_size=patch_len, stride=patch_len//2` (50% overlap) to project each patch to `d_model`. Adds learnable position embeddings and a `[CLS]` token at the end for global pooling. Implements `TimeframeEmbedding` — a learnable embedding for each timeframe (M1/M5/M15/M30/H1) added to the patch embedding. | | |
| TASK-009 | Create `ModelWorkbench/Learn/v2/model/transformer.py` with `CausalTransformerEncoder` — implements a GPT-style decoder-only transformer with causal self-attention. Uses `nn.MultiheadAttention` with `is_causal=True` or Flash Attention v2 via `torch.nn.functional.scaled_dot_product_attention`. Each layer: causal MHA → Add&Norm → SwiGLU FFN → Add&Norm. Uses pre-LayerNorm architecture. `n_layers=8` (configurable), `d_model=256`, `n_heads=8`, `d_ff=1024`. | | |
| TASK-010 | Create `ModelWorkbench/Learn/v2/model/heads.py` with four output heads: (1) `DistributionHead` — outputs `(μ, log_σ)` per horizon for a Gaussian distribution of forward log-returns, implemented as a 2-layer MLP: `d_model → 128 → n_horizons × 2`; (2) `DirectionHead` — outputs binary logits per horizon for P(return > 0), 2-layer MLP: `d_model → 64 → n_horizons`; (3) `VolatilityHead` — predicts future realized volatility per horizon, 2-layer MLP: `d_model → 64 → n_horizons`; (4) `RegimeHead` — classifies current bar into 4 volatility regimes, 2-layer MLP: `d_model → 32 → 4`. All heads read from the `[CLS]` token output. | | |
| TASK-011 | Create `ModelWorkbench/Learn/v2/model/full_model.py` with `TradeForecastTransformer(nn.Module)` — composes PatchEmbedding + CausalTransformerEncoder + 4 heads. Forward pass: `raw_ohlcv (B, seq_len, 5) → patch_embed (B, n_patches, d_model) → transformer (B, n_patches, d_model) → cls_output (B, d_model) → {distribution, direction, volatility, regime}`. Returns a `ModelOutput` dataclass. Total parameters: ~8-12M. | | |
| TASK-012 | Create `ModelWorkbench/Learn/v2/model/mtf_fusion.py` with `MTFFusionModule` — processes each timeframe independently through its own PatchEmbedding, then uses a lightweight cross-attention (1 layer, 4 heads) where base-timeframe acts as query and higher timeframes act as key/value. Outputs a fused representation for the base timeframe. This is optional — the base model without MTF fusion should also work standalone. | | |
| TASK-013 | Create `ModelWorkbench/Learn/v2/model/export.py` with `export_to_onnx(model, sample_input, path)` — exports the trained model to ONNX for MQL5 inference. Validates that the exported ONNX model produces identical outputs (within 1e-6) to the PyTorch model on a held-out batch. | | |
| TASK-014 | Unit tests in `ModelWorkbench/tests/v2/test_model.py`: verify causality (output at position t does not depend on input at position t+1), verify output shapes match config, verify gradient flow through all heads, verify ONNX export round-trip. | | |

### Implementation Phase 3: Training Methodology — Three-Phase Curriculum

- **GOAL-003**: Implement a three-phase training curriculum that first learns universal market representations via self-supervised pre-training, then fine-tunes on distributional regression targets, and finally optimizes for trading profit via reinforcement learning.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-015 | Create `ModelWorkbench/Learn/v2/training/pretrain.py` with `MaskedPatchPretraining` trainer. Uses masked autoencoding (MAE-style): randomly mask 50% of input patches, train the transformer to reconstruct the masked patches' OHLCV values using a lightweight decoder (2-layer transformer). Loss: MSE on reconstructed OHLCV normalized by ATR. Train across ALL available instruments and timeframes simultaneously (multi-instrument pre-training). Use `DataLoader` with `InfiniteRandomSampler` that cycles through datasets. Target: 10M training steps with batch_size=256, learning_rate=3e-4 with cosine schedule. | | |
| TASK-016 | Create `ModelWorkbench/Learn/v2/training/pretrain_data.py` with `MultiInstrumentDataset` — loads all CSV datasets from `data/`, normalizes each instrument's OHLCV by its rolling ATR, creates overlapping sliding windows of `max_seq_len + max_horizon` bars, and returns `(masked_ohlcv, target_ohlcv, mask_indices)` tuples. Handles different instruments having different row counts by weighting sampling probability proportional to dataset size. | | |
| TASK-017 | Create `ModelWorkbench/Learn/v2/training/finetune.py` with `DistributionalFinetuning` trainer. Phase 2 fine-tuning on the distributional regression targets. Implements the composite loss: `L_total = L_nll + λ₁·L_direction + λ₂·L_volatility + λ₃·L_regime` where `L_nll` is negative log-likelihood under the predicted Gaussian distribution (MSE on μ + penalty on σ), `L_direction` is binary cross-entropy, `L_volatility` is MSE on log volatility, and `L_regime` is cross-entropy. Default weights: λ₁=0.3, λ₂=0.2, λ₃=0.1. | | |
| TASK-018 | Extend `DistributionalFinetuning` with `QuantileLoss` option — alternatively predict quantiles [0.1, 0.25, 0.5, 0.75, 0.9] of the forward return distribution instead of Gaussian parameters. Use pinball loss: `L_τ(y, q) = max(τ(y - q), (τ - 1)(y - q))`. This is more robust to fat-tailed return distributions than Gaussian NLL. The head changes to output `n_horizons × 5` quantile values. | | |
| TASK-019 | Create `ModelWorkbench/Learn/v2/training/rl_finetune.py` with `RLPolicyFinetuning` trainer. Phase 3: cast the trading problem as contextual bandit → train a stochastic policy `π(action | state) = softmax(f_θ(state))` where actions are {BUY, SELL, HOLD} and the reward is the realized signed-quality (MFE/MAE normalized). Uses REINFORCE with baseline:
```
∇J = E[ (R - b) · ∇log π(a|s) ]
```
where `b` is the V-value predicted by the model's auxiliary value head. Only applies RL fine-tuning to the top 2 transformer layers + heads (freeze bottom 6 layers). Learning rate 1e-5, batch_size=64 trajectories of length 32. | | |
| TASK-020 | Create `ModelWorkbench/Learn/v2/training/metrics.py` with `TrainingMetricsTracker` — logs to W&B (Weights & Biases) or TensorBoard: Spearman per horizon, quantile calibration (PIT histogram), directional accuracy per horizon, regime classification F1, training/inference throughput. Also computes trading metrics: Sharpe ratio, max drawdown, profit factor on a held-out validation set at the end of each epoch. | | |
| TASK-021 | Create `ModelWorkbench/Learn/v2/training/folds.py` with `PurgedWalkForwardSplit` — implements purged walk-forward cross-validation: each fold has a training window, a purge gap (max_horizon bars to prevent leakage from overlapping outcomes), and a test window. Returns (train_indices, val_indices) tuples. Supports expanding-window and rolling-window modes. | | |
| TASK-022 | Create `ModelWorkbench/train_transformer.py` — CLI entry point mirroring `train_lgbm.py` that runs the full three-phase training pipeline. Arguments: `--ds-names` (list of CSV paths), `--pretrain-steps`, `--finetune-epochs`, `--rl-steps`, `--output-dir`. Saves model checkpoint every 1000 steps. Produces a model pack `.pt` file containing: model state_dict, config, normalization stats, training metrics history. | | |

### Implementation Phase 4: Signal Generation & Trade Execution Framework

- **GOAL-004**: Design a signal generation framework that transforms the model's distributional forecasts into actionable trade decisions with dynamic position sizing based on predicted edge and uncertainty.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-023 | Create `ModelWorkbench/Learn/v2/signals.py` with `DistributionalSignalGenerator` class. Input: model's `ModelOutput` for a bar. Process: (1) Compute expected return `E[r] = μ` at each horizon from the distribution head. (2) Compute Sharpe-like score: `s = E[r] / σ` where σ is the predicted std. (3) Compute directional confidence: `c = 2 × |P(return > 0) - 0.5|` from the direction head, scaled to [0, 1]. (4) Compute composite signal: `signal = tanh(s × c / temperature)` where temperature controls aggressiveness. `signal > 0` means BUY, `signal < 0` means SELL, near-zero means HOLD. (5) Filter by minimum threshold: only emit signal when `|signal| > signal_threshold` AND volatility regime is not extreme. | | |
| TASK-024 | Create `ModelWorkbench/Learn/v2/position_sizing.py` with `KellyPositionSizer` — computes position size as fraction of account: `f = (p_win × avg_win - p_loss × avg_loss) / (avg_win × avg_loss)`. Where `p_win` is predicted TP probability from the direction head, `avg_win` is the predicted mean return conditional on winning, and `avg_loss` is the predicted downside. Implement half-Kelly by default: `f_kelly = f / 2`. Cap position at 5% of account per trade. Account for current open positions via net exposure limit. | | |
| TASK-025 | Create `ModelWorkbench/Learn/v2/risk_manager.py` with `RiskManager` — enforces: max 3 concurrent positions per instrument, max 15% total account exposure, trailing stop-loss at 1.5× ATR, take-profit at 3× ATR, hard stop at 2% account equity, no new trades during high-impact news windows (configurable calendar), session-aware (reduced size during Asian session for XAUUSD). | | |
| TASK-026 | Create `ModelWorkbench/Learn/v2/signal_evaluator.py` with `SignalEvaluator` — computes offline signal quality metrics: (1) `signal_vs_outcome_scatter` — scatter plot of predicted signal vs realized signed-quality, (2) `decile_analysis` — sort bars by signal strength, compute win rate and avg return per decile, (3) `calibration_curve` — binned predicted P(win) vs actual win rate, (4) `threshold_sweep` — coverage vs win-rate trade-off curve, (5) `profit_curve` — cumulative P&L if trading top N% strongest signals. | | |
| TASK-027 | Unit tests in `ModelWorkbench/tests/v2/test_signals.py`: verify signal is 0 when uncertainty is high, verify signal sign matches expected return direction, verify Kelly fraction is bounded, verify position sizing respects account limits. | | |

### Implementation Phase 5: Backtesting & Validation Framework

- **GOAL-005**: Build a rigorous backtesting framework that validates model performance out-of-sample with realistic trading assumptions including spread, slippage, and commission.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-028 | Create `ModelWorkbench/Learn/v2/backtest.py` with `VectorizedBacktester` — processes the full test dataset in a single pass, generating signals at each bar, tracking open positions, and computing P&L. Key features: (1) respects causality — no signal uses future data, (2) models spread (0.2 pips for EURUSD, 0.3 for XAUUSD) as entry cost, (3) models commission at 0.5 pips per round turn, (4) handles overlapping signals (when model emits a signal while a position is already open, skip or queue), (5) records per-trade metrics: entry time, exit time, direction, P&L in pips, P&L in %, MFE, MAE, duration, exit reason (TP/SL/timeout). | | |
| TASK-029 | Create `ModelWorkbench/Learn/v2/backtest_metrics.py` with `BacktestMetrics` — computes: (1) Total return and CAGR, (2) Sharpe ratio (annualized, using 0% risk-free rate), (3) Sortino ratio, (4) Max drawdown and drawdown duration, (5) Win rate, (6) Profit factor (gross profit / gross loss), (7) Expectancy (avg P&L per trade), (8) Number of trades per day/week, (9) Return by session (Asian/London/NY), (10) Return by volatility regime, (11) Calmar ratio, (12) Monte Carlo bootstrap confidence intervals for Sharpe and win rate (1000 resamples of trade sequence). | | |
| TASK-030 | Create `ModelWorkbench/Learn/v2/walk_forward_backtest.py` with `WalkForwardBacktest` — implements proper walk-forward backtesting: (1) Split data into N sequential blocks. (2) For block i: train on blocks 0..i-1, test on block i, record predictions. (3) Never train on future data. (4) Aggregate all test-block predictions for final evaluation. This prevents the subtle leakage that occurs when a single model is evaluated on the same data used for hyperparameter tuning. | | |
| TASK-031 | Create Jupyter notebook `ModelWorkbench/4.1 Backtest Transformer.ipynb` — loads a trained model pack from Phase 2/3, runs walk-forward backtest, displays P&L curve, drawdown plot, trade scatter, and all metrics tables. Follows the pattern of `3.1 Backtest LGBM Regression.ipynb` but adapted for the transformer outputs. | | |
| TASK-032 | Create robustness tests in `ModelWorkbench/tests/v2/test_backtest.py`: test that a random signal generator produces Sharpe ≈ 0 (sanity check), test that backtest P&L matches manual calculation on a small synthetic dataset, test causality (no trade can open before its signal bar). | | |

### Implementation Phase 6: Production Deployment — MQL5 Bridge

- **GOAL-006**: Enable live trading by deploying the trained transformer model to MQL5 via ONNX, with feature computation and inference matching the Python training pipeline exactly.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-033 | Create `ModelWorkbench/Learn/v2/deploy.py` with `DeploymentPackager` — takes a trained PyTorch model and produces a deployment archive containing: (1) ONNX model file with the full inference graph (embedding + transformer + heads), (2) JSON metadata with model config, normalization parameters (mean/std per input channel), and feature specification, (3) Python reference implementation for validation, (4) MQL5 include file template. | | |
| TASK-034 | Create `ModelWorkbench/Learn/v2/feature_spec.py` with `FeatureSpec` — a formal specification of every input feature required by the model, including: feature name, computation formula, lookback window, normalization method. This replaces the ad-hoc feature computation in the current pipeline. The spec is serialized to JSON and consumed by both the Python inference code and the MQL5 indicator. | | |
| TASK-035 | Create `MQL5/Indicators/TransformerModel.mq5` — MQL5 indicator that: (1) reads the ONNX model file, (2) computes all required features (using `FeatureSpec` JSON), (3) runs ONNX inference at each new bar, (4) outputs signal values to a buffer readable by an Expert Advisor. Uses `OnnxRuntime` MQL5 bindings. Falls back gracefully if ONNX is not available (logs warning, outputs neutral signal). | | |
| TASK-036 | Create `MQL5/Experts/TransformerTrader.mq5` — Expert Advisor that: (1) reads signals from `TransformerModel` indicator buffer, (2) applies position sizing from `RiskManager` (reimplemented in MQL5), (3) manages orders with TP/SL, (4) handles partial close and trailing stops, (5) logs all trading activity. | | |
| TASK-037 | Create `ModelWorkbench/Learn/v2/parity_check.py` — validates that MQL5 inference output matches Python inference output exactly on the same input bars. Uses a small CSV of test bars: runs Python inference, exports expected outputs, then compares against MQL5 Strategy Tester output. Parity threshold: max absolute difference < 1e-5 for all outputs. | | |

## 3. Alternatives

- **ALT-001: Temporal Fusion Transformer (TFT)**: Considered TFT from Google Research for its built-in variable selection and interpretable attention. Rejected because TFT is designed for heterogeneous feature types (static, known-future, observed) which don't map cleanly to pure OHLCV data. Our architecture is simpler with fewer moving parts while capturing the same temporal dynamics.

- **ALT-002: N-BEATS/N-HiTS**: Considered these pure MLP-based architectures from ServiceNow/Element AI. Rejected because they are optimized for univariate point forecasting, not multivariate conditioning on OHLCV context. They lack the cross-channel attention that allows our model to learn relationships between price, volume, and volatility.

- **ALT-003: Graph Neural Networks over correlation matrices**: Considered building a graph where nodes are instruments/bars and edges represent correlations. Rejected for added complexity without clear benefit — cross-instrument relationships in FX are largely driven by USD correlation which a transformer can learn implicitly from multi-instrument training.

- **ALT-004: Reinforcement Learning from scratch (no supervised pre-training)**: Considered training a pure RL agent (PPO/SAC) on historical data. Rejected because RL from scratch on noisy financial data is extremely sample-inefficient and prone to overfitting on spurious patterns. The three-phase curriculum (pre-train → supervised → RL) provides the benefits of RL while leveraging the data efficiency of supervised pre-training.

- **ALT-005: Diffusion models for trajectory generation**: Considered using a denoising diffusion probabilistic model to generate plausible future price trajectories, then deriving trade decisions from the generated distribution. Rejected for inference latency reasons — diffusion requires many iterative denoising steps, making it impractical for sub-5ms per-bar inference on CPU.

- **ALT-006: TimesFM / Lag-Llama / Chronos — fine-tune a pre-trained foundation model**: Considered fine-tuning a publicly available time series foundation model. Rejected because (1) these models are trained on non-financial data and their representations don't transfer well to FX, (2) they are large (100M+ parameters) and slow at inference, (3) licensing restrictions on commercial trading use are unclear. Our smaller purpose-built model trained from scratch is more appropriate.

## 4. Dependencies

- **DEP-001**: PyTorch ≥ 2.1 (for `torch.nn.functional.scaled_dot_product_attention` flash attention support)
- **DEP-002**: `h5py` — for HDF5 label storage in `LabelStore`
- **DEP-003**: `onnx` + `onnxruntime` — for model export and MQL5 inference
- **DEP-004**: `optuna` — for automated hyperparameter optimization of learning rates, loss weights, and architecture dimensions
- **DEP-005**: `wandb` (Weights & Biases) or `tensorboard` — for training metric tracking
- **DEP-006**: `numba` — for accelerated label computation in `compute_forward_excursion_surface`
- **DEP-007**: `hmmlearn` — for Hidden Markov Model volatility regime labeling
- **DEP-008**: TA-Lib (system library) — retained only for ATR computation (may be replaced by a pure Numba implementation later)
- **DEP-009**: `pandas`, `numpy`, `scipy`, `scikit-learn` — already in requirements.txt
- **DEP-010**: NVIDIA GPU with CUDA 11.8+ and ≥ 12 GB VRAM (RTX 3060 or better) for training; CPU-only inference is supported for deployment
- **DEP-011**: MQL5 build ≥ 3800 with ONNX support enabled for live trading
- **DEP-012**: Historical OHLCV data from cTrader API (fetched via existing `fetch_datasets_bulk.py`)

## 5. Files

- **FILE-001**: `ModelWorkbench/Learn/v2/__init__.py` — package init, exports public API
- **FILE-002**: `ModelWorkbench/Learn/v2/labels.py` — distributional label computation (replaces `Learn/labels.py` triple-barrier)
- **FILE-003**: `ModelWorkbench/Learn/v2/model/__init__.py` — model package init
- **FILE-004**: `ModelWorkbench/Learn/v2/model/config.py` — `ModelConfig` dataclass
- **FILE-005**: `ModelWorkbench/Learn/v2/model/embedding.py` — `PatchEmbedding`, `TimeframeEmbedding`
- **FILE-006**: `ModelWorkbench/Learn/v2/model/transformer.py` — `CausalTransformerEncoder`
- **FILE-007**: `ModelWorkbench/Learn/v2/model/heads.py` — `DistributionHead`, `DirectionHead`, `VolatilityHead`, `RegimeHead`
- **FILE-008**: `ModelWorkbench/Learn/v2/model/full_model.py` — `TradeForecastTransformer`
- **FILE-009**: `ModelWorkbench/Learn/v2/model/mtf_fusion.py` — `MTFFusionModule`
- **FILE-010**: `ModelWorkbench/Learn/v2/model/export.py` — ONNX export
- **FILE-011**: `ModelWorkbench/Learn/v2/training/__init__.py` — training package init
- **FILE-012**: `ModelWorkbench/Learn/v2/training/pretrain.py` — MaskedPatchPretraining trainer
- **FILE-013**: `ModelWorkbench/Learn/v2/training/pretrain_data.py` — MultiInstrumentDataset
- **FILE-014**: `ModelWorkbench/Learn/v2/training/finetune.py` — DistributionalFinetuning trainer
- **FILE-015**: `ModelWorkbench/Learn/v2/training/rl_finetune.py` — RLPolicyFinetuning trainer
- **FILE-016**: `ModelWorkbench/Learn/v2/training/metrics.py` — TrainingMetricsTracker
- **FILE-017**: `ModelWorkbench/Learn/v2/training/folds.py` — PurgedWalkForwardSplit
- **FILE-018**: `ModelWorkbench/Learn/v2/training/losses.py` — composite loss functions (NLL, pinball, focal, etc.)
- **FILE-019**: `ModelWorkbench/Learn/v2/signals.py` — DistributionalSignalGenerator
- **FILE-020**: `ModelWorkbench/Learn/v2/position_sizing.py` — KellyPositionSizer
- **FILE-021**: `ModelWorkbench/Learn/v2/risk_manager.py` — RiskManager
- **FILE-022**: `ModelWorkbench/Learn/v2/signal_evaluator.py` — SignalEvaluator
- **FILE-023**: `ModelWorkbench/Learn/v2/backtest.py` — VectorizedBacktester
- **FILE-024**: `ModelWorkbench/Learn/v2/backtest_metrics.py` — BacktestMetrics
- **FILE-025**: `ModelWorkbench/Learn/v2/walk_forward_backtest.py` — WalkForwardBacktest
- **FILE-026**: `ModelWorkbench/Learn/v2/deploy.py` — DeploymentPackager
- **FILE-027**: `ModelWorkbench/Learn/v2/feature_spec.py` — FeatureSpec
- **FILE-028**: `ModelWorkbench/Learn/v2/parity_check.py` — Python/MQL5 parity checker
- **FILE-029**: `ModelWorkbench/Learn/v2/data.py` — data loading and preprocessing for v2 models (sliding window construction, normalization)
- **FILE-030**: `ModelWorkbench/train_transformer.py` — CLI training entry point
- **FILE-031**: `ModelWorkbench/4.1 Backtest Transformer.ipynb` — backtest notebook
- **FILE-032**: `MQL5/Indicators/TransformerModel.mq5` — MQL5 ONNX inference indicator
- **FILE-033**: `MQL5/Experts/TransformerTrader.mq5` — MQL5 Expert Advisor
- **FILE-034**: `ModelWorkbench/data/labels/` — HDF5 label cache directory (gitignored)
- **FILE-035**: `ModelWorkbench/ModelPacks/transformers/` — trained model pack output directory
- **FILE-036**: `ModelWorkbench/tests/v2/test_labels.py` — label computation tests
- **FILE-037**: `ModelWorkbench/tests/v2/test_model.py` — model architecture tests
- **FILE-038**: `ModelWorkbench/tests/v2/test_signals.py` — signal generation tests
- **FILE-039**: `ModelWorkbench/tests/v2/test_backtest.py` — backtest correctness tests

## 6. Testing

- **TEST-001**: `test_causality_labels` — verify that `compute_forward_excursion_surface` and `compute_directional_return_distribution` produce NaN for the last `max_horizon` rows (no future data leakage). Run on synthetic data with known forward paths.
- **TEST-002**: `test_causality_model` — create a synthetic input sequence `[0, 0, ..., 0, 1, 0, ..., 0]` (impulse at position k). Verify that model output at positions < k is identical regardless of the value at position k. This proves no information flows backward.
- **TEST-003**: `test_patch_embedding_shape` — verify that `PatchEmbedding` with `patch_len=16`, `stride=8`, and `seq_len=512` produces exactly `(512 - 16) / 8 + 1 = 63` patches.
- **TEST-004**: `test_distribution_head_calibration` — on synthetic data where returns are exactly normally distributed with known μ and σ, verify that the DistributionHead NLL loss recovers the true parameters within 5% tolerance.
- **TEST-005**: `test_kelly_position_sizing_bounds` — verify that KellyPositionSizer outputs are always in [0, max_position_pct] and that zero edge produces zero position.
- **TEST-006**: `test_backtest_pnl_reconciliation` — create a synthetic 100-bar dataset with predetermined OHLCV and hand-computed P&L. Verify that `VectorizedBacktester` produces identical P&L.
- **TEST-007**: `test_onnx_export_roundtrip` — train a tiny model (d_model=32, n_layers=2) for 1 epoch, export to ONNX, load with onnxruntime, and verify that outputs match PyTorch outputs within 1e-6 on 100 random inputs.
- **TEST-008**: `test_walk_forward_no_leakage` — verify that `WalkForwardBacktest` never uses test data during training: for fold i, model is trained on data ending before fold i's test data starts.
- **TEST-009**: `test_pinball_loss_gradient` — verify that the pinball loss gradient exists and is non-zero for all quantile levels τ ∈ {0.1, 0.25, 0.5, 0.75, 0.9}.
- **TEST-010**: `test_mtf_causality` — verify that `MTFFusionModule` uses only completed higher-timeframe bars (shifted by 1 HTF bar), matching the causal behavior of `add_multitimeframe_features`.
- **TEST-011**: `test_gradient_flow` — run a single forward/backward pass through the full `TradeForecastTransformer` with batch_size=4, verify that all parameters receive gradients (grad is not None) and that loss decreases over 10 training steps on a fixed batch.
- **TEST-012**: `test_signal_monotonicity` — verify that `DistributionalSignalGenerator` produces monotonically stronger signals as the predicted edge (μ/σ) increases, all else equal.

## 7. Risks & Assumptions

- **RISK-001**: **Overfitting to noise**. Financial time series have extremely low signal-to-noise ratio (SNR). A transformer with 10M parameters trained on 2M bars may memorize noise patterns that don't generalize. **Mitigation**: (1) Multi-instrument pre-training acts as a regularizer by forcing shared representations. (2) Heavy dropout (0.1-0.2) and weight decay (1e-4). (3) Purged walk-forward validation catches overfitting before deployment. (4) RL fine-tuning uses a separate reward distribution than supervised training, breaking memorization.

- **RISK-002**: **Inference latency exceeds MQL5 budget**. ONNX inference of an 8-layer transformer may exceed 5 ms on the MT5 thread, causing chart lag. **Mitigation**: (1) Export with ONNX graph optimizations (constant folding, operator fusion). (2) Use INT8 quantization for the ONNX model (target: 2× speedup, < 0.5% accuracy loss). (3) If still too slow, distill the transformer into a smaller model (e.g., 3-layer student via knowledge distillation). (4) Architecture includes the option to reduce `n_layers` to 4 and `d_model` to 128 for deployment-only models.

- **RISK-003**: **Distribution shift in live trading**. Market conditions change (regime shifts, volatility clustering, structural breaks) and the model's predictions become miscalibrated. **Mitigation**: (1) Weekly retraining with the most recent data. (2) The distribution head's predicted σ serves as an uncertainty estimate — when σ spikes, the signal generator automatically reduces position size or goes flat. (3) Online calibration: running estimate of prediction error, adjust signal threshold dynamically. (4) Ensemble of models trained on different lookback windows.

- **RISK-004**: **Transformer training instability**. Deep transformers are notoriously sensitive to learning rate and initialization in small-data regimes. **Mitigation**: (1) Use the pre-normalization (pre-LN) architecture, which is more stable. (2) Learning rate warmup for first 1000 steps. (3) Gradient clipping at 1.0. (4) Start with a very small model (d_model=64, n_layers=2) and scale up only after stable training is confirmed. (5) Monitor gradient norms and attention pattern sparsity during training.

- **RISK-005**: **Reinforcement learning phase degrades performance**. RL fine-tuning on noisy financial rewards can destroy the useful representations learned in pre-training and supervised fine-tuning. **Mitigation**: (1) RL only tunes the top 2 layers (bottom 6 frozen), preserving representations. (2) Very low learning rate (1e-5 vs 1e-3 for fine-tuning). (3) Early stopping based on validation Sharpe — stop RL if validation Sharpe drops for 3 consecutive epochs. (4) Maintain a supervised loss auxiliary term during RL training to prevent catastrophic forgetting.

- **RISK-006**: **GPU memory insufficient for batch training**. 512-bar sequences × 256 batch size with a 10M-parameter transformer may exceed 24 GB VRAM during gradient accumulation. **Mitigation**: (1) Use gradient accumulation with micro-batches of 32. (2) Mixed-precision training (FP16/BF16) via `torch.cuda.amp`. (3) Flash Attention reduces memory by O(seq_len) for the attention computation. (4) If still tight, reduce `max_seq_len` to 256 and `batch_size` to 128.

- **ASSUMPTION-001**: Historical OHLCV data from cTrader is sufficient quality (no significant gaps, no timestamp errors beyond what existing pipeline handles). The existing `fetch_datasets_bulk.py` provides adequate data for pre-training.

- **ASSUMPTION-002**: The 4 volatility regimes (low/normal/high/extreme) provide a useful inductive bias for the model. If HMM regime labeling is inaccurate, the regime head can still learn useful representations through the multi-task objective.

- **ASSUMPTION-003**: Self-supervised pre-training on OHLCV reconstruction transfers useful representations to the trading task. This is supported by literature (e.g., TST, PatchTST) showing that masked reconstruction pre-training improves downstream forecasting, but has not been specifically validated on FX data at this scale.

- **ASSUMPTION-004**: A single model can generalize across instruments (XAUUSD, EURUSD, BTCUSD) when trained jointly. These instruments have different volatility profiles, but market microstructure patterns (momentum, mean reversion, support/resistance) may share common structure.

- **ASSUMPTION-005**: MQL5's `OnnxRuntime` binding supports all ONNX ops used by our model (LayerNormalization, ScaledDotProductAttention, GELU/SwiGLU). If not, we may need to restrict the model to ONNX opset 17-supported operations.

## 8. Related Specifications / Further Reading

- [PatchTST: A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730) — ICLR 2023. Inspiration for patch-based time series processing.
- [A Time Series is Worth Five Words](https://arxiv.org/abs/2310.12627) — iTransformer, inverts the attention dimension (variates as tokens instead of time steps).
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) — MAE paper. Inspiration for the masked patch pre-training strategy.
- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) — TFT paper. Considered as an alternative (see ALT-001).
- [Deep Reinforcement Learning for Trading](https://arxiv.org/abs/1911.10107) — Survey of RL applications in finance. Informs the RL fine-tuning phase.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) — Guo et al. Relevant for ensuring the distribution head produces well-calibrated probabilities.
- [Does Self-Supervised Learning Really Improve Reinforcement Learning?](https://arxiv.org/abs/2206.05292) — Evidence that SSL pre-training helps downstream RL, supporting our three-phase curriculum.
- [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) — Marcos López de Prado. Source of purged walk-forward cross-validation and triple-barrier labeling methods.
- [MQL5 ONNX Runtime Documentation](https://www.mql5.com/en/docs/integration/onnx_runtime) — Reference for ONNX model deployment in MetaTrader 5.
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Dao et al. Used to reduce GPU memory during training.

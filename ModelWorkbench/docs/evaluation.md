# Evaluation — How Model Performance Is Measured

> **Package:** `train_transformer.py` training loop | `Learn/v2/training/metrics.py` | `Learn/v2/backtest_metrics.py`

## Overview

The model is evaluated at three levels of increasing realism:

| Level | What it measures | When |
|-------|-----------------|------|
| **Training metrics** (every epoch) | Validation loss, Spearman, direction accuracy | During training |
| **Backtest metrics** (post-training) | P&L, Sharpe, drawdown, win rate | After training completes |
| **Walk-forward metrics** (cross-validation) | Out-of-sample robustness | For production deployment |

---

## Level 1: Training Metrics (Every Epoch)

These metrics are computed on a held-out 10% validation set and logged every epoch to `metrics.jsonl`.

### Metric: Validation Loss (Gaussian NLL)

```python
loss = 0.5 × log(2πσ²) + 0.5 × ((y − μ) / σ)²
```

**What it tells you:** How well the model's predicted distribution (μ, σ) fits the actual outcomes. Lower is better.

**How to read it:** If loss is flat (e.g., 0.9189 every epoch), the model is outputting a constant prediction — it has not learned. If loss decreases steadily, the model is improving.

**Diagnostic table:**

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| Train loss ↓, Val loss flat | Overfitting | Add dropout, reduce model capacity, add more data |
| Train loss ↓, Val loss ↓ | Learning | Continue — the model is generalizing |
| Both losses flat | Underfitting / dead model | Check for NaN, increase LR, simplify task |
| Both losses oscillating | LR too high or batch too small | Reduce LR, increase batch size |

### Metric: Validation Spearman (Rank Correlation)

```python
for each horizon h:
    spearman_h = spearmanr(mu_pred[:, h], y_true[:, h])
mean_spearman = mean(spearman_h over all 6 horizons)
```

**What it tells you:** How well the model **ranks** bars from most-to-least favorable. This is the primary metric because profitable trading depends on correctly ordering opportunities, not predicting exact magnitudes.

**Interpretation:**

| Spearman | Meaning |
|----------|---------|
| ~0.00 | Random — model provides no useful ranking |
| 0.05-0.10 | Weak but detectable signal (our Run 2) |
| 0.15-0.25 | Moderate signal — usable for trading with thresholding (Run 3) |
| 0.30+ | Strong signal — comparable to v1 LightGBM pipeline |
| 0.50+ | Very strong — unlikely in M1 FX; check for data leakage |

**Per-horizon breakdown:** The `val_spearman_per_horizon` field in `metrics.jsonl` shows which horizons the model learns best:
```json
"val_spearman_per_horizon": [0.05, 0.12, 0.23, 0.18, 0.09, 0.02]
//                          h=5    h=10   h=20   h=40   h=60   h=120
```
This example shows the model is best at predicting 20-bar outcomes (most learnable horizon) and worst at 120-bar (too much noise over long horizons).

### Metric: Directional Accuracy

```python
dir_acc = mean(sign(mu_pred) == sign(y_true))  # for bars where both are non-zero
```

**What it tells you:** How often the model correctly predicts the sign (direction) of the forward return.

**Interpretation:**
- 50% = coin flip (no signal)
- 55% = weak edge (our Run 3 result)
- 57% = moderate edge (Run 3 final epoch)
- 60%+ = strong edge (profitable with proper position sizing)

### Metric: Gradient Norm

```python
max_grad_norm = max(||∇L||) across all batches in the epoch
```

**What it tells you:** Training stability. Clipped at 1.0 by design.

**Diagnostic table:**

| Pattern | Diagnosis |
|---------|-----------|
| Grad norm ≈ 1.0 every epoch | Gradients are being clipped — model is actively learning |
| Grad norm < 0.1 | Gradients are vanishing — learning has stalled |
| Grad norm > 100 (pre-clip) | Exploding gradients — reduce LR or add gradient clipping |

---

## Level 2: Backtest Metrics (Post-Training)

After training, the model is evaluated on completely unseen test data using `VectorizedBacktester` and `BacktestMetrics`.

### Backtest Setup

```python
backtester = VectorizedBacktester(
    spread_pips=0.3,          # Bid-ask spread
    commission_per_lot=7.0,   # Round-turn commission
    lot_size=100000,          # Standard lot
    max_hold_bars=120,        # Maximum trade duration
)

trades, equity_curve = backtester.run(
    df=test_ohlcv,
    signals=model_signals,    # from DistributionalSignalGenerator
    tp_atr_mult=3.0,          # Take-profit at 3× ATR
    sl_atr_mult=1.5,          # Stop-loss at 1.5× ATR
)
```

### Key Backtest Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Sharpe Ratio** | (mean(daily_returns) / std(daily_returns)) × √252 | > 1.0 (good), > 2.0 (excellent) |
| **Sortino Ratio** | mean / std(negative_returns_only) | > 1.5 (downside-only) |
| **Max Drawdown** | max(peak − trough) / peak | < 20% (acceptable), < 10% (good) |
| **Win Rate** | winning_trades / total_trades | > 55% (profitable with 1:2 R:R) |
| **Profit Factor** | gross_profit / gross_loss | > 1.5 (good), > 2.0 (excellent) |
| **Expectancy** | avg(P&L per trade) in % | > 0.1% (positive edge) |
| **Calmar Ratio** | CAGR / |max_drawdown| | > 1.0 |

### Per-Trade Tracking

Each trade is recorded with rich metadata:

```python
Trade(
    entry_time    = datetime,     # Bar when signal triggered
    exit_time     = datetime,     # Bar when trade closed
    direction     = 1 or -1,      # Long or short
    entry_price   = float,        # Entry fill price (with spread)
    exit_price    = float,        # Exit fill price
    pnl_pips      = float,        # Profit/loss in pips
    pnl_pct       = float,        # Profit/loss as % of account
    mfe_pips      = float,        # Maximum favorable excursion
    mae_pips      = float,        # Maximum adverse excursion
    duration_bars = int,          # How long the trade was open
    exit_reason   = "tp"/"sl"/"timeout"/"signal_reversed",
)
```

This enables detailed post-mortem analysis: "Do our losses come from early stop-outs or late reversals? Do our wins cap at TP or run past it?"

### Monte Carlo Confidence Intervals

```python
ci_low, ci_high = BacktestMetrics.monte_carlo_sharpe(daily_returns, n_simulations=1000)
# → "The true Sharpe is between 0.8 and 1.6 with 95% confidence"
```

Bootstraps trade sequences 1000 times to estimate how robust the backtest results are to trade ordering. If the confidence interval crosses zero, the strategy's edge is not statistically significant.

---

## Level 3: Walk-Forward Validation

The `WalkForwardBacktest` class implements purged walk-forward CV (from López de Prado, *Advances in Financial Machine Learning*):

```
Dataset: [Block0][Block1][Block2][Block3][Block4]

Fold 0: Train on [Block0]           → Test on [Block1] (with purge gap)
Fold 1: Train on [Block0+Block1]    → Test on [Block2] (with purge gap)
Fold 2: Train on [Block0..Block2]   → Test on [Block3] (with purge gap)
Fold 3: Train on [Block0..Block3]   → Test on [Block4] (with purge gap)
```

**Purge gap:** Bars within `max_horizon` of the test boundary are excluded from training. This prevents data leakage from overlapping forward-looking labels.

**Why this matters:** A single train/test split may capture a favorable market regime. Walk-forward testing across multiple sequential folds proves the model works across different market conditions.

---

## Signal Evaluation (Offline)

The `SignalEvaluator` class provides pre-trade quality analysis:

### Decile Analysis
Sort all bars by signal strength, group into 10 equal buckets, compute win rate per bucket. The top decile should have significantly higher win rate than the bottom decile — this proves the signal has predictive power.

### Calibration Curve
Bin predicted probabilities into 10 buckets, plot predicted P(win) vs. actual win rate. A perfectly calibrated model lies on the diagonal. Overconfident models sit above the diagonal.

### Threshold Sweep
For 20 threshold levels from 0 to max(signal): compute coverage (% of bars traded) vs. win rate. Used to find the optimal balance between trade frequency and quality.

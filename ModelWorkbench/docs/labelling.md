# Labelling Pipeline — How Training Targets Are Built

> **Package:** `Learn/v2/labels.py` | **Key functions:** `compute_forward_excursion_surface`, `compute_atr_normalized_targets`

## Overview

The labelling pipeline transforms raw OHLCV bars into **continuous, bounded, ATR-normalized trade-quality scores** that the model learns to predict. Unlike the v1 pipeline (which discretizes outcomes into TP/SL/timeout buckets), v2 produces **real-valued scores in [-1, +1]** that preserve information about *how strongly* the market moved in each direction.

The core insight: **a wide range of possible target values carries more signal for the model than a ternary label**, because the model can learn a smooth gradient from "strongly favorable" → "neutral" → "strongly adverse" rather than jumping between buckets.

---

## Step 1: Compute Forward Excursion Surface

**Function:** `compute_forward_excursion_surface(df, horizons, atr_window=14)`

### What it does

For every bar `t` and every horizon `h ∈ [5, 10, 20, 40, 60, 120]`, computes how far price moved favorably and adversely **from the perspective of both a buyer and a seller**, expressed in ATR (Average True Range) units.

### Algorithm (Numba-accelerated inner loop)

```
For each bar t:
  For each horizon h:
    Look forward from bar t to bar t+h:
    
    From a BUY perspective (entered long at Close[t]):
      best_price  = max(High[t+1 ... t+h])
      worst_price = min(Low[t+1 ... t+h])
      
      buy_MFE[t, h] = (best_price / Close[t] - 1) / (ATR[t] / Close[t])
      buy_MAE[t, h] = (1 - worst_price / Close[t]) / (ATR[t] / Close[t])
    
    From a SELL perspective (entered short at Close[t]):
      best_price  = min(Low[t+1 ... t+h])
      worst_price = max(High[t+1 ... t+h])
      
      sell_MFE[t, h] = (1 - best_price / Close[t]) / (ATR[t] / Close[t])
      sell_MAE[t, h] = (worst_price / Close[t] - 1) / (ATR[t] / Close[t])
```

### Key properties

| Property | Detail |
|----------|--------|
| **Causal** | Only uses bars `t+1` through `t+h` — no lookahead leakage |
| **ATR-normalized** | Dividing by `ATR[t]/Close[t]` makes the score comparable across different volatility regimes and instruments |
| **Numba-accelerated** | `@njit(cache=True)` inner loop runs in O(n × h) time; handles 500K bars × 6 horizons in ~30 seconds |
| **Tail handling** | Last `max(horizons)` rows produce NaN (no forward data); these are filtered during window construction |

### Output shape

```python
excursion.shape  # → (n_bars, 6, 2, 2)
#                      │      │  │  └── [:, 0] = buy, [:, 1] = sell
#                      │      │  └──── [:, 0] = MFE, [:, 1] = MAE
#                      │      └─────── 6 horizons: [5, 10, 20, 40, 60, 120]
#                      └────────────── n_bars in the dataset
```

### Example

| Bar | Horizon | buy_MFE | buy_MAE | Interpretation |
|-----|---------|---------|---------|----------------|
| 1000 | 5 bars | 2.1 | 0.3 | Price moved 2.1× ATR up before moving 0.3× ATR down — strong bullish impulse |
| 1000 | 20 bars | 1.8 | 2.5 | Price rallied 1.8× ATR but then reversed 2.5× ATR — a bull trap |
| 5000 | 60 bars | 0.0 | 3.2 | Price only moved down — straight loss for a buyer |

---

## Step 2: Convert Excursions to Trade Quality Score

**Function:** `compute_atr_normalized_targets(df, horizons, atr_window=14)` (in `train_transformer.py`)

### What it does

Collapses the (MFE, MAE) pair into a single **signed quality score** that captures the *balance* between favorable and adverse movement.

### Formula

```
score[t, h] = (buy_MFE[t, h] - buy_MAE[t, h]) / max(buy_MFE[t, h] + buy_MAE[t, h], 1e-8)
```

### Interpretation

| Score | Meaning |
|-------|---------|
| **+1.0** | Pure favorable movement (MAE ≈ 0, only made money) |
| **+0.5** | MFE is 3× MAE (moved 3 units up for every 1 unit down) |
| ** 0.0** | MFE = MAE (equal movement both ways, or no movement) |
| **−0.5** | MAE is 3× MFE (moved 3 units down for every 1 unit up) |
| **−1.0** | Pure adverse movement (MFE ≈ 0, only lost money) |

### Why this works better than raw returns

| Target Type | Range | Stationarity | Signal |
|-------------|-------|-------------|--------|
| **Raw log returns** `ln(C[t+h]/C[t])` | (−∞, +∞) | Non-stationary (volatility clusters) | ~99% noise on M1 |
| **ATR score** (above) | [−1, +1] | Scale-invariant (ATR absorbs volatility changes) | MFE/MAE ratio has persistent structure |

The ATR score is **bounded** (the model only needs to predict values in [-1, +1]), **scale-invariant** (works the same on XAUUSD at $4000 and EURUSD at $1.10), and captures **trade-relevant information** (the balance between favorable and adverse movement, not just the net outcome).

---

## Step 3: Other Label Functions (Reference)

### `compute_directional_return_distribution(df, horizons)`
- Pure forward log returns: `ln(Close[t+h] / Close[t])`
- Used when `--target-type log_return` (the original approach; inferior — see Run 1 results)
- Range: (−∞, +∞), highly non-stationary

### `compute_optimal_exit_labels(df, tp_atr_mult, sl_atr_mult, max_horizon)`
- Triple-barrier simulation with continuous outcomes
- Returns: trade outcome (+1/−1/0), duration to exit, MFE/MAE at exit
- Bridges v1 (ternary) and v2 (continuous) labeling paradigms
- Used for evaluation, not training

### `compute_volatility_regime_labels(df, lookback=20, n_regimes=4)`
- Assigns each bar to low/normal/high/extreme volatility regime
- Uses rolling realized volatility + spread proxy → quantile binning
- Causal: regime at bar t uses only data up to t
- Used by `RegimeHead` in the model and `RiskManager` in signal generation

### `LabelStore` (HDF5 cache)
- Persists computed label tensors to `ModelWorkbench/data/labels/`
- Keyed by `SHA256(dataset_fingerprint + params)`
- Transparent caching: `store.get_or_compute(key, compute_fn)` — recomputes only if params change

---

## Label → Training Window Mapping

```
Raw OHLCV DataFrame:  [t₀, t₁, t₂, ..., tₙ₋₁]  (n bars)
                                 ↓
Forward excursion labels:  [L₀, L₁, L₂, ..., Lₙ₋₁]  (n bars × 6 horizons)
                                 ↓
Sliding window at index i:
  Input:  OHLCV[i : i + seq_len]        ← seq_len=256 bars of context
  Target: labels[i + seq_len - 1]        ← the label at the LAST bar in the window
                                 ↓
Training example:
  X: (1, 256, 5)  ← normalized O/H/L/C/V for bars 0-255
  y: (1, 6)       ← ATR scores at bar 255, for horizons [5, 10, 20, 40, 60, 120]
```

The model sees 256 bars of history and must predict what WILL happen over the next 5/10/20/40/60/120 bars — **from the perspective of a buyer entering at the last bar's close**.

---

## Sanity Checks Run During Data Loading

1. **NaN filtering**: Windows where the target has NaN (last `max_horizon` rows) are excluded
2. **Feature NaN filtering**: Windows with NaN/Inf in normalized OHLCV are excluded
3. **Train/val split**: Chronological split preserves temporal ordering (the model never sees future bars during training — self-attention provides causality within each window)

# Labels, Signals & Execution Engine

> **Modules:** `Learn/v2/labels.py` · `Learn/v2/signals.py` · `Learn/v2/position_sizing.py`  
> **Supporting:** `Learn/v2/risk_manager.py` · `Learn/v2/backtest.py` · `Learn/v2/backtest_metrics.py` · `Learn/v2/signal_evaluator.py`

---

## Mental Map: From Prices to Profits

```
                         ┌─────────────────────────────┐
                         │  Normalized OHLCV DataFrame  │
                         │  (from data pipeline)        │
                         └─────────────┬───────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
         ▼                             ▼                             ▼
┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│ LABELS.PY        │    │ LABELS.PY            │    │ LABELS.PY            │
│                  │    │                      │    │                      │
│ Excursion        │    │ Optimal Exit         │    │ Volatility Regime    │
│ Surface          │    │ (Triple Barrier)     │    │ Classification       │
│                  │    │                      │    │                      │
│ MFE/MAE at       │    │ TP/SL/timeout        │    │ 4 classes:           │
│ multiple horizons│    │ outcomes per bar      │    │ 0=low vol            │
│ in ATR units     │    │ with MFE/MAE traces  │    │ 3=extreme vol        │
└────────┬─────────┘    └──────────┬───────────┘    └──────────┬───────────┘
         │                         │                           │
         │    ┌────────────────────┼─────────────────────┐     │
         │    │                    │                     │     │
         │    ▼                    ▼                     ▼     │
         │ ┌─────────────────────────────────────────────┐    │
         │ │           LabelStore (HDF5 Cache)           │    │
         │ │  Cache key = SHA-256(dataset_name + params) │    │
         │ │  Avoids recomputing expensive label arrays  │    │
         │ └─────────────────────┬───────────────────────┘    │
         │                       │                            │
         └───────────┬───────────┴────────────┬───────────────┘
                     │                        │
                     ▼                        ▼
              ┌─────────────┐         ┌─────────────┐
              │  TRAINING   │         │  INFERENCE  │
              │  PHASE 2    │         │  (live/     │
              │             │         │   backtest) │
              │  Model      │         │             │
              │  learns to  │         │  Model      │
              │  predict    │         │  outputs    │
              │  labels     │         │  predictions│
              └─────────────┘         └──────┬──────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │  SIGNALS.PY              │
                              │  DistributionalSignalGenerator │
                              │                          │
                              │  ModelOutput → scalar    │
                              │  trade signal [-1, 1]    │
                              │                          │
                              │  Algorithm:              │
                              │  1. μ/σ → Sharpe score   │
                              │  2. Direction confidence │
                              │  3. Composite: sign ×    │
                              │     tanh(|s|·c/T)       │
                              │  4. Regime gate (zero    │
                              │     in extreme vol)      │
                              │  5. Threshold gate       │
                              └────────────┬─────────────┘
                                           │
                                           ▼
                              ┌──────────────────────────┐
                              │  EXECUTION LAYER         │
                              │                          │
                              │  KellyPositionSizer      │
                              │  → optimal bet fraction  │
                              │                          │
                              │  RiskManager             │
                              │  → entry allowed?        │
                              │  → TP/SL/trailing calc   │
                              │                          │
                              │  VectorizedBacktester    │
                              │  → simulated fills       │
                              │  → spread, commission    │
                              │  → equity curve          │
                              └────────────┬─────────────┘
                                           │
                                           ▼
                              ┌──────────────────────────┐
                              │  EVALUATION              │
                              │                          │
                              │  BacktestMetrics         │
                              │  → Sharpe, Sortino, DD   │
                              │  → Monte Carlo CI        │
                              │                          │
                              │  SignalEvaluator         │
                              │  → decile analysis       │
                              │  → calibration curve     │
                              │  → threshold sweep       │
                              └──────────────────────────┘
```

---

## Part A: Label Engineering (`labels.py`)

All labels are computed **strictly causally** — the label at bar `t` is derived only from data at times `t+1` onward. The feature window at time `t` never sees this future data. The labels exist to provide supervised training targets for what the model should learn to forecast.

Four distinct label types are produced:

---

### Label Type 1: Forward Excursion Surface

**Function:** `compute_forward_excursion_surface(df, horizons, atr_window=14)`

**What it computes:** For each bar `t`, scans forward over `[t+1, t+h]` for every horizon `h` in the list. Records the maximum favorable excursion (MFE) and maximum adverse excursion (MAE) for both a hypothetical *long* (buy) and *short* (sell) trade entered at `Close[t]`.

**Output shape:** `(n_bars, n_horizons, 2, 2)` float64 array.

| Index | Meaning |
|-------|---------|
| `[t, h, 0, 0]` | Buy MFE — best price improvement for a long, in ATR units |
| `[t, h, 0, 1]` | Buy MAE — worst drawdown for a long, in ATR units |
| `[t, h, 1, 0]` | Sell MFE — best improvement for a short, in ATR units |
| `[t, h, 1, 1]` | Sell MAE — worst drawdown for a short, in ATR units |

**Why ATR units:** A 50-pip move means different things on EURUSD vs BTCUSD, and in 2020 vs 2024. Dividing by ATR/Close at entry time makes the excursion scale-free and comparable across instruments and volatility regimes. A value of 1.5 means "the price moved 1.5× the current ATR in that direction."

**Horizon examples:**
```
horizons = [5, 10, 20, 40, 60, 120]
h=5   → looks 5 bars ahead  (e.g., 25 minutes on M5)
h=120 → looks 120 bars ahead (e.g., 10 hours on M5)
```

**Implementation detail:** Uses a Numba `@njit(cache=True)` kernel (`_compute_excursion_surface_nb`) for the O(n_bars × n_horizons × max_horizon) forward scan. The last `max(horizons)` rows contain NaN (insufficient forward data).

**Use in training:** The `train_transformer.py` script can use this surface to derive an ATR-normalized trade-quality score:
```
score = (buy_MFE - buy_MAE) / max(buy_MFE + buy_MAE, ε)   # range [-1, +1]
```
A score of +1 means price only moved favorably (pure win); -1 means pure loss.

---

### Label Type 2: Directional Return Distribution

**Function:** `compute_directional_return_distribution(df, horizons)`

**What it computes:** Forward log returns at multiple horizons from every bar. For each bar `t` and horizon `h`:
```
return[t, h] = ln(Close[t + h] / Close[t])
```

**Output shape:** `(n_bars, n_horizons)` float64 array.

This is the primary target when running with `--target-type log_return`. The model's `DistributionHead` predicts a Gaussian `(μ, σ)` for the log return at each horizon. The model's `DirectionHead` predicts `P(return > 0)` — whether the return will be positive.

**Why log returns:** Log returns are additive over time, approximately symmetric, and numerically well-behaved. The Gaussian NLL loss works naturally on them.

---

### Label Type 3: Optimal Exit Labels (Triple Barrier)

**Function:** `compute_optimal_exit_labels(df, tp_atr_mult=2.5, sl_atr_mult=2.5, max_horizon=60, atr_window=14)`

**What it computes:** For every bar, simulates a hypothetical trade (both long and short) and records the outcome. Three possible exits:

```
          BUY                                          SELL
          TP barrier = Close[t] + 2.5×ATR              TP barrier = Close[t] - 2.5×ATR
          SL barrier = Close[t] - 2.5×ATR              SL barrier = Close[t] + 2.5×ATR
          Timeout    = bar t + 60                       Timeout    = bar t + 60

Price →
  │                           TP hit (+1)                                
  │    ╱╲                                                              
  │   ╱  ╲     ╱╲                                                       
  │  ╱    ╲   ╱  ╲─── entry                                           
  │ ╱      ╲╱    ╲    ╲           timeout (0)                          
  │╱              ╲    ╲─── target                                     
  │                ╲                                                    
  │                 ╲─── SL hit (-1)                                   
  └──────────────────────────────────────────► time (bars)
```

**Output DataFrame columns:**

| Column | Values | Meaning |
|--------|--------|---------|
| `buy_outcome` | +1, -1, 0, NaN | TP hit, SL hit, timeout, no data |
| `sell_outcome` | +1, -1, 0, NaN | Same for short side |
| `buy_duration` | bars held | How long until exit (or max_horizon for timeout) |
| `sell_duration` | bars held | |
| `buy_mfe` | ≥0 (ATR units) | Best favorable excursion during the trade |
| `buy_mae` | ≥0 (ATR units) | Worst adverse excursion during the trade |
| `sell_mfe` | ≥0 | |
| `sell_mae` | ≥0 | |

**Practical use:** These labels are used for evaluating signal quality — you can check whether model signals tend to lead to TP or SL outcomes. They are also the foundation for the RL reward function in Phase 3.

**Numba kernel:** `_compute_optimal_exit_nb` — performs the barrier simulation for every bar in a compiled loop. Returns 8 arrays simultaneously.

---

### Label Type 4: Volatility Regime Classification

**Function:** `compute_volatility_regime_labels(df, lookback=20, n_regimes=4)`

**What it computes:** Classifies each bar into one of 4 volatility regimes using two causal proxies:

1. **Rolling realised volatility** — standard deviation of log returns over `lookback` bars
2. **Rolling spread proxy** — mean of `(High-Low)/Close` over `lookback` bars

Each proxy is z-scored using a 100-bar rolling window (causal), then summed into a composite score. The composite is binned into equal-frequency quartiles using global quantile thresholds.

**Regime classes:**
- **Regime 0** — Lowest volatility (quiet, range-bound)
- **Regime 1** — Moderate-low
- **Regime 2** — Moderate-high
- **Regime 3** — Extreme volatility (crisis, news events)

**Important note about deployment:** The function uses global quantile thresholds computed from the full dataset. For production walk-forward use, thresholds must be pre-computed on training data and passed as fixed bin edges to avoid lookahead bias. Early bars (<100) return -1 (insufficient history).

**How the model uses it:** The `RegimeHead` predicts the current regime class. During signal generation, signals in Regime 3 (extreme vol) can be gated to zero as a safety measure.

---

### HDF5 Label Cache (`LabelStore`)

**Purpose:** Label computation is expensive — especially the excursion surface (O(n×h) forward scan on millions of bars) and the triple-barrier simulation. `LabelStore` avoids recomputation by caching results in an HDF5 file.

**Cache key construction:**
```
key = SHA-256(dataset_name + params_dict)[:32]
```

The key includes the dataset identity (filename, shape) and all parameters that affect the computation (horizons, ATR window, TP/SL multipliers). Different parameters produce different cache keys.

**Usage pattern:**
```python
store = LabelStore(base_dir="ModelWorkbench/data/labels")
key = store._make_key("BTCUSD_M5", {"horizons": [5,10,20], "atr_window": 14})
labels = store.get_or_compute(key, compute_forward_excursion_surface, df, horizons=[5,10,20])
```

First call: computes and stores. Subsequent calls with the same key: loads from HDF5 instantly.

---

## Part B: Signal Generation (`signals.py`)

Once the model has produced a `ModelOutput` (from the `TradeForecastTransformer.forward()` pass), the `DistributionalSignalGenerator` transforms those distributional predictions into a scalar trade signal.

### Algorithm: 5-Step Pipeline

```
ModelOutput
  │
  ├─ distribution: (μ[t], log_σ[t]) per horizon
  └─ direction: logit per horizon → P(return > 0)

Step 1: SHARPE-LIKE SCORE
         s = μ[h] / exp(log_σ[h])       at primary horizon h
         Captures expected return per unit of predicted risk.

Step 2: DIRECTIONAL CONFIDENCE
         p_up = sigmoid(dir_logits[h])    probability return > 0
         c = 2 × |p_up - 0.5|            confidence in [0, 1]
         c=0.0 → model is 50/50 (uncertain)
         c=1.0 → model is certain about direction

Step 3: COMPOSITE SIGNAL
         signal = sign(s) × tanh(|s| × c / temperature)

         |s| × c  → edge scaled by confidence
         tanh()   → squashes to [-1, 1]
         sign(s)  → preserves which direction
         temperature → higher = softer signals (default 1.0)

Step 4: REGIME GATE (optional)
         if regime_pred == extreme_regime_class:
             signal = 0.0
         Prevents trading during high-volatility chaos.

Step 5: THRESHOLD GATE
         if |signal| < threshold:
             signal = 0.0
         Filters out weak/noisy predictions.
```

### Signal Interpretation

| Signal Value | Action | Confidence |
|-------------|--------|-----------|
| +0.8 to +1.0 | Strong BUY | Model has high conviction of upward move |
| +0.2 to +0.8 | Moderate BUY | Positive edge, moderate confidence |
| -0.1 to +0.1 | HOLD | No clear edge or below threshold |
| -0.2 to -0.8 | Moderate SELL | Negative edge, moderate confidence |
| -0.8 to -1.0 | Strong SELL | Model has high conviction of downward move |

### Configuration Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `temperature` | 1.0 | Higher values produce softer (closer to 0) signals |
| `signal_threshold` | 0.1 | Signals with `|value|` below this are zeroed |
| `extreme_regime_gate` | True | Zero signals in extreme vol regime |
| `regime_idx` | 3 | Which regime class is "extreme" |
| `primary_horizon_idx` | 2 | Which horizon to use (0=5b, 1=10b, 2=20b, 3=40b, 4=60b, 5=120b) |

### Multi-Horizon Variant

`generate_multi_horizon()` produces a `(batch_size, n_horizons)` signal matrix instead of collapsing to a single horizon. Useful for multi-timeframe strategies where different horizons drive different trade legs.

---

## Part C: Execution Layer

The signal is not a trade. The execution layer translates signals into position sizes, risk-managed entries, and tracked trades.

### 1. Position Sizing (`KellyPositionSizer`)

Uses the Kelly criterion to compute optimal bet size per signal:

```
f* = (p_win × avg_win - p_loss × avg_loss) / (avg_win × avg_loss)
```

- Uses **half-Kelly** by default (f*/2) for conservative sizing
- Clamped to `max_position_pct` of account equity (default 5%)
- Returns 0 if expected edge is negative
- `batch_compute()` provides vectorized sizing for backtesting

**Connecting to model output:** The `win_prob` comes from the direction head's sigmoid output. The `avg_win`/`avg_loss` can be estimated from the distribution head's predicted excursion magnitudes or from historical trade statistics.

### 2. Risk Management (`RiskManager`)

Enforces trading discipline:
- **Max concurrent positions:** 3 (configurable)
- **Max total exposure:** 15% of account equity
- **Take-profit:** `take_profit_atr_mult × ATR` from entry (default 3.0×)
- **Stop-loss:** `trailing_stop_atr_mult × ATR` (default 1.5×)
- **Hard stop:** 2% of account equity
- **Trailing stop:** Updates as price moves favorably, locks in profit

### 3. Vectorized Backtest (`VectorizedBacktester`)

Simulates realistic trading with:
- **Entry:** Next bar's Open after signal (no lookahead)
- **Exit logic:** Checks intra-bar High/Low for TP/SL hits (realistic fill simulation)
- **Spread:** Applied at entry (buy at Ask, sell at Bid)
- **Commission:** Per-lot round-trip deduction
- **Max hold duration:** Configurable timeout (default 120 bars)
- **Signal reversal:** Exits position if signal flips strongly opposite
- **End of data:** Closes open position at last Close

### 4. Walk-Forward Backtest (`WalkForwardBacktest`)

Gold-standard evaluation that prevents the cardinal sin of backtesting — training on future data:

```
Time →
├── Fold 1 ──┤              ├── Fold 2 ──┤              ├── Fold 3 ──┤
[████ train ████▐gap▐ test ▐██████████████▐gap▐ test ▐████████████████▐gap▐ test]
```

For each fold:
1. **Train** the model on all data up to the test boundary (with purge gap)
2. **Predict** signals on the test period (model has never seen this data)
3. **Backtest** those signals against realized prices
4. **Aggregate** all out-of-fold signals and trades for final metrics

---

## Part D: Evaluation Framework

### `BacktestMetrics`

Comprehensive performance statistics from a trades list and equity curve:

| Metric Category | Metrics |
|----------------|---------|
| **Returns** | Total return, CAGR |
| **Risk-adjusted** | Sharpe ratio, Sortino ratio |
| **Drawdown** | Max drawdown, max drawdown duration |
| **Trade stats** | Win rate, profit factor, expectancy, avg win/loss, trades/day |
| **Confidence** | Monte Carlo bootstrap Sharpe 95% CI |

### `SignalEvaluator`

Evaluates signal quality *before* any trading simulation — purely from the signal and the realized outcome:

1. **Decile Analysis:** Sort bars by signal strength. If strong signals truly capture better outcomes, the top decile should have higher win rate and Sharpe than the bottom.
2. **Calibration Curve:** Binned predicted probability vs actual outcome frequency. Perfect calibration means a bar with `P(win)=0.7` actually wins 70% of the time.
3. **Threshold Sweep:** Coverage vs win-rate tradeoff. Raising the signal threshold filters to higher-quality signals but reduces the number of trades.
4. **Profit Curve:** Cumulative P&L if you trade the top N% strongest signals.

---

## Summary: Labels → Training → Signals → Execution

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  LABEL COMPUTE    │     │  MODEL TRAINING   │     │  SIGNAL GEN      │
│                   │     │                   │     │                   │
│  Excursion surface│────►│  Phase 2:         │────►│  μ/σ → Sharpe    │
│  Directional ret  │     │  Gaussian NLL on  │     │  + direction     │
│  Triple barrier   │     │  log returns      │     │  confidence      │
│  Vol regime       │     │  + aux losses     │     │  → [-1, 1]       │
│                   │     │                   │     │  signal          │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                                              ┌────────────▼─────────┐
                                              │  EXECUTION           │
                                              │                      │
                                              │  Kelly size → Risk   │
                                              │  check → Entry →     │
                                              │  TP/SL exit → P&L    │
                                              │                      │
                                              │  Walk-forward eval   │
                                              │  Sharpe, DD, stats   │
                                              └──────────────────────┘
```

**Key invariant across the entire pipeline:** Causality. Labels look forward from `t`. Features stop at `t`. The model bridges the gap. The execution layer respects the same temporal boundary.

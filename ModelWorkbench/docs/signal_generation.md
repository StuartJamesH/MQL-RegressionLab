# Signal Generation — From Model Predictions to Trade Decisions

> **Class:** `DistributionalSignalGenerator` in `Learn/v2/signals.py`

## Overview

The model outputs **distributional forecasts** for 6 forward horizons — it does NOT output binary buy/sell signals. The `DistributionalSignalGenerator` transforms the model's rich probabilistic output into a single scalar trade signal in the range **[-1, +1]**, where:

| Signal | Meaning |
|--------|---------|
| **+1.0** | Maximum confidence BUY |
| **+0.5** | Moderate BUY |
| ** 0.0** | No action (HOLD) |
| **−0.5** | Moderate SELL |
| **−1.0** | Maximum confidence SELL |

---

## Step-by-Step Algorithm

### Input: The Model's Distributional Forecast

For each bar, the model produces a `ModelOutput` dataclass:

```python
ModelOutput(
    distribution = (mu, log_sigma),  # each: (batch, 6) — 6 forward horizons
    direction    = logits,           # (batch, 6) — P(return > 0) logits
    regime       = logits,           # (batch, 4) — volatility regime classes
    volatility   = predictions,      # (batch, 6) — future vol estimates
)
```

### Step 1: Isolate the Primary Horizon

The generator focuses on a single "primary" horizon (default: **horizon index 2 = 20 bars**):

```python
h = 2  # 20-bar horizon
mu_h = mu[:, h]           # expected log-return over next 20 bars
sigma_h = exp(log_sigma[:, h])  # predicted standard deviation
```

### Step 2: Compute Sharpe-Like Score

```python
s = mu_h / sigma_h  # "Information ratio" — expected return per unit risk
```

**Intuition:** A bar with μ = 0.02 and σ = 0.01 gets s = 2.0 (strong signal). A bar with μ = 0.02 and σ = 0.10 gets s = 0.2 (weak signal — high uncertainty). The model's prediction uncertainty is directly priced into the signal strength.

### Step 3: Compute Directional Confidence

```python
p_up = sigmoid(direction_logits[:, h])  # P(return > 0) ∈ [0, 1]
c = 2 × |p_up − 0.5|                    # confidence ∈ [0, 1]
```

**Interpretation:**
- `p_up = 0.95` → c = 0.90 (strong directional conviction)
- `p_up = 0.55` → c = 0.10 (weak conviction, barely above coin-flip)
- `p_up = 0.50` → c = 0.00 (complete uncertainty — signal will be near-zero)

This folds the model's **directional certainty** (from the direction head) into the signal. Even if the expected return is large, if the model is uncertain about direction, the signal is dampened.

### Step 4: Composite Signal

```python
signal = sign(s) × tanh(|s| × c / temperature)
```

This formula has three multiplicative components:

| Component | Role |
|-----------|------|
| `sign(s)` | Direction: positive → BUY, negative → SELL |
| `|s|` | Magnitude of expected return per unit risk |
| `c` | Directional confidence from the model |
| `temperature` | Scales aggressiveness (higher = more conservative) |

The `tanh` function smoothly saturates strong signals to ±1 while keeping moderate signals proportional.

### Step 5: Regime Gate

```python
if regime_class == EXTREME (index 3):
    signal = 0.0  # No trades in extreme volatility
```

The model's `RegimeHead` classifies each bar into 4 volatility regimes (0=low, 1=normal, 2=high, 3=extreme). During extreme volatility, all signals are suppressed — the model is likely miscalibrated and risk is asymmetric.

### Step 6: Threshold Gate

```python
if |signal| < signal_threshold (default 0.1):
    signal = 0.0  # Filter weak signals
```

Weak signals near zero are clipped to avoid overtrading on marginal predictions. The threshold can be tuned to balance trade frequency vs. signal quality.

---

## Multi-Horizon Mode

`generate_multi_horizon()` applies the same algorithm but returns signals for **all 6 horizons simultaneously**, producing a (batch, 6) matrix. This allows the downstream system to:

- Use **short-horizon signals** (5-10 bars) for scalp entries
- Use **medium-horizon signals** (20-40 bars) for swing entries  
- Use **long-horizon signals** (60-120 bars) for trend-following
- **Vote across horizons**: if all horizons agree on direction, increase confidence

---

## Signal Generation Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `temperature` | 1.0 | Higher = softer signals (e.g., 2.0 halves signal strength) |
| `signal_threshold` | 0.1 | Minimum \|signal\| to trade; higher = fewer, higher-quality trades |
| `extreme_regime_gate` | True | Suppress signals when model detects extreme volatility |
| `regime_idx` | 3 | Which regime class is "extreme" |
| `primary_horizon_idx` | 2 | Which horizon to generate signals for (2 = 20 bars) |

### Tuning Strategy

| Goal | Adjust |
|------|--------|
| More trades (lower bar) | Lower `signal_threshold` (0.05), lower `temperature` (0.5) |
| Higher quality per trade | Raise `signal_threshold` (0.2), enable regime gate |
| Conservative sizing | Raise `temperature` (2.0) — signals saturate less aggressively |
| Aggressive sizing | Lower `temperature` (0.5) — signals saturate faster |

---

## Integration with Position Sizing

After the `DistributionalSignalGenerator` produces a scalar signal, the `KellyPositionSizer` converts it into an actual position size:

```python
signal = generator.generate(model_output)    # → [-1, 1]
direction = 1 if signal > 0 else -1

p_win = max(sigmoid(dir_logits[h]), 0.5)    # model's win probability
avg_win = abs(mu_h) if mu_h > 0 else abs(mu_h)  # conditional expected return
avg_loss = abs(sigma_h)                      # use std as loss proxy

position_pct = kelly.compute_size(p_win, avg_win, avg_loss, equity)
position_size = position_pct * equity        # in account currency
```

And `RiskManager` enforces limits (max 3 concurrent positions, max 15% total exposure, trailing stops at 1.5× ATR).

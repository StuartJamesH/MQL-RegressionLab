# Pipeline Mind Map

```
                              RAW OHLCV CSVs
                          (XAUUSD, BTCUSD, EURUSD)
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              load_ohlcv()   load_ohlcv()   load_ohlcv()
              n_rows=200K    n_rows=200K    n_rows=200K
                    │              │              │
                    ▼              ▼              ▼
              ┌─────────────────────────────────────────┐
              │        normalize_ohlcv(df)              │
              │                                         │
              │  O/H/L/C → log(price / prev_close)     │
              │  Volume → vol / rolling_median(252)     │
              │  Output: (n_bars, 5) float32            │
              │  Scale-free — $4000 gold ≡ $1.10 euro   │
              └─────────────────────────────────────────┘
                    │
                    ▼
              ┌─────────────────────────────────────────┐
              │      SessionFeatureEncoder              │
              │                                         │
              │  hour → sin(2π·h/24), cos(2π·h/24)     │
              │  weekday → sin(2π·d/7), cos(2π·d/7)    │
              │  Output: (n_bars, 4) float32            │
              └─────────────────────────────────────────┘
                    │
                    ▼
     ╔══════════════════════════════════════════════════╗
     ║             TARGET  ENGINEERING                  ║
     ║                                                  ║
     ║  compute_forward_excursion_surface(df, horizons) ║
     ║    ├── For each bar t, each horizon h:           ║
     ║    │   ├── Buy  MFE = max(H[t+1:t+h])/C[t] - 1  ║
     ║    │   ├── Buy  MAE = 1 - min(L[t+1:t+h])/C[t]  ║
     ║    │   ├── Sell MFE = 1 - min(L[t+1:t+h])/C[t]  ║
     ║    │   └── Sell MAE = max(H[t+1:t+h])/C[t] - 1  ║
     ║    │                                             ║
     ║    └── Normalize by ATR[t] / Close[t]            ║
     ║        (Numba @njit — O(n×h) — 30s for 1M rows) ║
     ║                                                  ║
     ║  compute_atr_normalized_targets()                ║
     ║    score = (MFE - MAE) / (MFE + MAE + 1e-8)     ║
     ║    Range: [-1, +1]                               ║
     ║    +1 = pure win, 0 = neutral, −1 = pure loss    ║
     ╚══════════════════════════════════════════════════╝
                    │
                    ▼
     ┌──────────────────────────────────────────────┐
     │         SLIDING  WINDOWS                      │
     │                                               │
     │  window[i] = raw[i : i+seq_len]                │
     │  label[i]  = target[i + seq_len - 1]           │
     │                                               │
     │  X: (n_windows, seq_len, 9)                   │
     │     └── 5 OHLCV + 4 session                   │
     │  y: (n_windows, 6)                            │
     │     └── scores at horizons [5,10,20,40,60,120]│
     │                                               │
     │  Filter: drop windows with NaN labels          │
     │          (last max_horizon bars)               │
     └──────────────────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
    Train (90%)           Val (10%)
    449K windows          50K windows
         │                     │
         ▼                     │
╔═════════════════════════════╗│
║     CAUSAL PATCH TRANSFORMER║│
║                              ║│
║  ┌─────────────────────────┐ ║│
║  │  PatchEmbedding         │ ║│
║  │  Conv1d(9→d_model, k=16,│ ║│
║  │         stride=8)       │ ║│
║  │  + position embed       │ ║│
║  │  + [CLS] token          │ ║│
║  │  (B, n_patches+1, 256)  │ ║│
║  └───────────┬─────────────┘ ║│
║              ▼               ║│
║  ┌─────────────────────────┐ ║│
║  │  CausalTransformer × 4  │ ║│
║  │  Pre-LN → MHA(causal)   │ ║│
║  │  → Add → LN → SwiGLU    │ ║│
║  │  → Add × 4 layers       │ ║│
║  └───────────┬─────────────┘ ║│
║              ▼               ║│
║       CLS token ─────────────╝│
║              │                │
║     ┌────────┼────────┐       │
║     ▼        ▼        ▼       │
║  ┌──────┐┌──────┐┌────────┐  │
║  │Distr ││Dir   ││Regime  │  │
║  │Head  ││Head  ││Head    │  │
║  │μ,logσ││logits││4-class │  │
║  │6horiz││6horiz││logits  │  │
║  └──────┘└──────┘└────────┘  │
║              │                │
╚══════════════╪════════════════╝
               │
               ▼
╔══════════════════════════════════════════╗
║          TRAINING  LOOP                  ║
║                                          ║
║  Loss = NLL(μ, logσ, y)                  ║
║       + 0.5 · BCE(dir_logits, sign(y))   ║
║       + 0.1 · MSE(vol, log|y|)           ║
║                                          ║
║  Optimizer: AdamW(lr=2e-4, wd=1e-4)     ║
║  Scheduler: CosineAnnealing(T_max=30)    ║
║  Gradient clipping: max_norm=1.0         ║
║                                          ║
║  ┌────────────────────────────────────┐  ║
║  │  EPOCH  LOGGING  (every epoch)     │  ║
║  │  Train Loss   Val Loss             │  ║
║  │  Val Spearman (per horizon)        │  ║
║  │  Direction Accuracy                │  ║
║  │  Gradient Norm   Learning Rate     │  ║
║  │  → saved to metrics.jsonl          │  ║
║  └────────────────────────────────────┘  ║
╚══════════════════════════════════════════╝
               │
               ▼
╔══════════════════════════════════════════════════╗
║              SIGNAL  GENERATION                   ║
║                                                   ║
║  DistributionalSignalGenerator                    ║
║                                                   ║
║  Step 1:  s = μ / σ         (Sharpe-like score)  ║
║  Step 2:  c = 2|P(up)−0.5|  (directional conf)   ║
║  Step 3:  signal = sign(s)·tanh(|s|·c/T)          ║
║  Step 4:  gate(regime==extreme) → signal = 0      ║
║  Step 5:  gate(|signal| < threshold) → signal = 0 ║
║                                                   ║
║  Output: scalar in [-1, +1]                       ║
║    +1 = strong buy   −1 = strong sell             ║
║     0 = no trade (HOLD)                           ║
╚══════════════════════════════════════════════════╝
               │
               ▼
╔══════════════════════════════════════════════════╗
║         POSITION  SIZING  &  RISK                ║
║                                                   ║
║  KellyPositionSizer                               ║
║    f = (p_win·avg_win − p_loss·avg_loss)          ║
║        / (avg_win · avg_loss)                     ║
║    half-Kelly: f = f/2                            ║
║    cap: max 5% of account                         ║
║                                                   ║
║  RiskManager                                      ║
║    max 3 concurrent positions                     ║
║    max 15% total exposure                         ║
║    trailing stop: 1.5× ATR                        ║
║    take-profit:   3.0× ATR                        ║
║    hard stop:     2% account equity               ║
╚══════════════════════════════════════════════════╝
               │
               ▼
╔══════════════════════════════════════════════════╗
║               BACKTESTING                         ║
║                                                   ║
║  VectorizedBacktester                             ║
║    Single-pass simulation with:                   ║
║    • 0.3 pip spread (XAUUSD)                      ║
║    • Commission per round-turn                    ║
║    • 120 bar max hold                             ║
║                                                   ║
║  Per-trade tracking:                              ║
║    entry/exit time, direction, P&L, MFE, MAE     ║
║    duration, exit reason (TP/SL/timeout)          ║
║                                                   ║
║  BacktestMetrics                                  ║
║    Sharpe · Sortino · Max Drawdown                ║
║    Win Rate · Profit Factor · Expectancy          ║
║    Monte Carlo CI (1000 bootstraps)               ║
║                                                   ║
║  WalkForwardBacktest                              ║
║    Purged expanding-window cross-validation        ║
║    Fold i: train [0..i] → test [i+1] with purge   ║
╚══════════════════════════════════════════════════╝
               │
               ▼
╔══════════════════════════════════════════════════╗
║              DEPLOYMENT                           ║
║                                                   ║
║  DeploymentPackager                               ║
║    ├── model.onnx         ← ONNX for MQL5         ║
║    ├── config.json        ← architecture spec     ║
║    ├── normalizer.json    ← input normalization   ║
║    ├── feature_spec.json  ← feature computation   ║
║    ├── model_info.json    ← training metadata     ║
║    └── model.pt           ← PyTorch checkpoint    ║
║                                                   ║
║  MQL5/Indicators/TransformerModel.mq5             ║
║    OnInit → load ONNX → OnCalculate → inference   ║
║                                                   ║
║  MQL5/Experts/TransformerTrader.mq5               ║
║    OnTick → read signals → position sizing        ║
║    → manage orders (TP/SL/trailing)               ║
╚══════════════════════════════════════════════════╝
```

---

## Key Data Shapes

```
Stage              Shape                      Notes
─────              ─────                      ─────
Raw CSV            (n_bars, 6)                Time, O, H, L, C, V
normalize_ohlcv    (n_bars, 5)                log-ratio prices + volume
SessionEncoder     (n_bars, 4)                hour_sin, hour_cos, dow_sin, dow_cos
Excursion surface  (n_bars, 6, 2, 2)          horizons × [buy,sell] × [MFE,MAE]
ATR score target   (n_bars, 6)                one score per bar per horizon

Sliding window X   (n_windows, 256, 9)        9 = 5 OHLCV + 4 session
Sliding window y   (n_windows, 6)             6 horizons

Patch embedding    (B, 33, 256)              33 = ceil(256/8) patches + CLS
Transformer out    (B, 33, 256)               same shape
CLS token          (B, 256)                   pooled global representation

Distribution head  μ: (B, 6)  logσ: (B, 6)   6 = [5,10,20,40,60,120] bar horizons
Direction head     (B, 6)                    P(return>0) logits per horizon  
Regime head        (B, 4)                    4 volatility classes
Volatility head    (B, 6)                    predicted future log-volatility

Signal             (B,)                      scalar in [-1, +1]
Position size      (B,)                      account currency units
Trade              per-trade dict            entry/exit, P&L, MFE, MAE, reason
```

---

## Design Decisions — Why Each Choice

| Decision | Why |
|----------|-----|
| **Conv1d patch embedding** not raw bars | 16-bar patches capture local candle patterns (engulfing, doji, hammer) automatically |
| **50% overlap (stride=8)** | Redundancy prevents boundary artifacts — each bar appears in 2 patches |
| **Causal attention** not bidirectional | No lookahead — bar t can only see bars ≤ t |
| **[CLS] attends to all** not just causal | CLS pools global context for multi-horizon prediction without violating causality at the patch level |
| **SwiGLU FFN** not GELU/ReLU | Better gradient flow, modern transformer standard |
| **Pre-LayerNorm** not post-LN | More stable training, especially with learning rate warmup |
| **Gaussian NLL** not MSE | Model outputs both prediction AND uncertainty — sigma feeds signal generation |
| **ATR score target** not raw returns | Bounded [-1,1], scale-invariant, higher signal-to-noise ratio |
| **Multi-instrument training** | Forces learning of transferable patterns, not instrument-specific artifacts |
| **Half-Kelly sizing** | Full Kelly is too aggressive with estimated probabilities; half-Kelly guards against estimation error |

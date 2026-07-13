# Causal Patch Transformer — Architecture Mind Map

> **Model:** TradeForecastTransformer | **Parameters:** ~1.1M (production) / ~8.5M (default)
> **Input:** 10 channels (5 OHLCV + 5 session) | **Output:** Scalar trade signal ∈ [-1, +1]

---

## Phase A: Data Preparation (per instrument, independently)

Each CSV goes through the same pipeline independently before pooling.

```
Raw CSV (one instrument)
┌──────────────────────────────────────┐
│ Time         O      H      L      C  │  e.g. XAUUSD M1, 500K rows
│ 2024-01-02.. 2068.5 2069.1 2068.3.. │
│ 2024-01-02.. 2068.8 2069.0 2068.5.. │
│ ...                                  │
└──────────────┬───────────────────────┘
               │
   ┌───────────┴───────────┐
   │                       │
   ▼                       ▼
normalize_ohlcv()      SessionFeatureEncoder
┌──────────────────┐   ┌──────────────────────────┐
│ O = ln(O[t]/C[t-1])│   │ hour_sin, hour_cos       │
│ H = ln(H[t]/C[t-1])│   │ dow_sin,  dow_cos        │
│ L = ln(L[t]/C[t-1])│   │ has_gap  (0 or 1)        │
│ C = ln(C[t]/C[t-1])│   │                          │
│ V = Vol/med_252(V) │   │ 5 channels               │
│                    │   │                          │
│ 5 channels         │   │                          │
└────────┬───────────┘   └────────────┬─────────────┘
         │                            │
         └──────────┬─────────────────┘
                    │
                    ▼
         Sliding windows: each window = 256 consecutive bars
         ┌──────────────────────────────────────────┐
         │ Window 0: bars [0..255]    label=bar 255 │  (n_windows, 256, 10)
         │ Window 1: bars [1..256]    label=bar 256 │
         │ Window 2: bars [2..257]    label=bar 257 │
         │ ...                                      │
         │ Window N: bars [N..N+255]  label=bar N+255│
         └──────────────────────────────────────────┘

Labels (ATR score target):
  For bar t, look forward at horizons [5,10,20,40,60,120]:
    score[t,h] = (MFE - MAE) / (MFE + MAE)    ∈ [-1, +1]
```

---

## Phase B: Data Pooling — The Critical Difference

### Single Instrument (e.g. XAUUSD only)

```
  XAUUSD M1 ──► (N_xau, 256, 10) windows
                    │
                    ▼
             Random shuffle
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   90% Train              10% Val
  (0.9 × N_xau)         (0.1 × N_xau)

  Model only sees XAUUSD patterns
```

### Three Instruments (XAUUSD + BTCUSD + EURUSD)

```
  XAUUSD ──► (N_xau, 256, 10) ─┐
  BTCUSD ──► (N_btc, 256, 10) ─┼──► np.concatenate()
  EURUSD ──► (N_eur, 256, 10) ─┘       │
                                       ▼
                          (N_total, 256, 10)
                          N_total = N_xau + N_btc + N_eur
                                       │
                                Random shuffle
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
                   90% Train                    10% Val
                 (mixed instruments)         (mixed)

  Model sees interleaved windows from all 3 instruments
  Batch might contain: [BTC_win, XAU_win, EUR_win, XAU..]

  ⚠️ No instrument ID — model can't tell them apart
  ⚠️ Relies on normalization to make them look the same
```

---

## Phase C: Model Forward Pass (one window → one prediction)

### Input: one window = (256 bars, 10 channels)

```
   bar 0      bar 1      bar 2           bar 255
   ┌─────┐    ┌─────┐    ┌─────┐         ┌─────┐
   │O H L│    │O H L│    │O H L│   ...   │O H L│  ← 5 OHLCV channels
   │C V  │    │C V  │    │C V  │         │C V  │
   │h_sin│    │h_sin│    │h_sin│         │h_sin│  ← hour sin
   │h_cos│    │h_cos│    │h_cos│         │h_cos│  ← hour cos
   │d_sin│    │d_sin│    │d_sin│         │d_sin│  ← dow sin
   │d_cos│    │d_cos│    │d_cos│         │d_cos│  ← dow cos
   │ gap │    │ gap │    │ gap │         │ gap │  ← has_gap (0 or 1)
   └──┬──┘    └──┬──┘    └──┬──┘         └──┬──┘
      │          │          │               │
      └──────────┴──────────┴───────────────┘
                       │
                       ▼
```

### Step 1: PatchEmbedding (Conv1d)

```
Conv1d(in=10, out=128, kernel=16, stride=8)

256 bars → sliding 16-bar patches, 50% overlap:

bars[0..15]    ──Conv1d──► patch_0   (128-d vector)
bars[8..23]    ──Conv1d──► patch_1   (128-d vector)
bars[16..31]   ──Conv1d──► patch_2   (128-d vector)
...
bars[240..255] ──Conv1d──► patch_30  (128-d vector)

(256 - 16) / 8 + 1 = 31 patches

+ learned position encoding per patch
+ prepend [CLS] token at position 0

Output: (32 positions, 128 dims)
  pos 0:  [CLS]      ← attends to EVERYTHING
  pos 1:  patch_0    ← bar window [0..15]
  pos 2:  patch_1    ← bar window [8..23]
  ...
  pos 31: patch_30   ← bar window [240..255]
```

### Step 2: CausalTransformerEncoder (4 layers)

```
Multi-Head Self-Attention with CAUSAL mask:

       [CLS]  patch_0  patch_1  patch_2  ...  patch_30
[CLS]    ✓       ✓        ✓        ✓             ✓      ← full attention
p_0      ✗       ✓        ✗        ✗             ✗      ← self only
p_1      ✗       ✓        ✓        ✗             ✗      ← look left
p_2      ✗       ✓        ✓        ✓             ✗
...
p_30     ✗       ✓        ✓        ✓             ✓

[CLS] can see the ENTIRE sequence (future patches)
Patches are causally masked — can only see past patches

SwiGLU FFN after each attention layer
Output: same shape (32 positions, 128 dims)
```

### Step 3: Prediction Heads (from [CLS] token only)

```
                    ┌──────────┴──────────┐
                    │  Take [CLS] token    │  ← ONLY the CLS token is used
                    │  (position 0)        │    for predictions
                    └──────────┬──────────┘
                               │
                               ▼  (128-dim vector summarizing ALL 256 bars)
                               │
     ┌─────────────────────────┼─────────────────────────┐
     │                         │                         │
     ▼                         ▼                         ▼
┌─────────────────┐  ┌──────────────┐  ┌────────────────┐
│ DistributionHead │  │ DirectionHead │  │ VolatilityHead  │
│                  │  │               │  │                 │
│ 128→256→12       │  │ 128→64→6     │  │ 128→64→6       │
│                  │  │               │  │                 │
│ μ[0..5]          │  │ logit[0..5]  │  │ log_vol[0..5]  │
│ log σ[0..5]     │  │               │  │                 │
│                  │  │ per horizon:  │  │                 │
│ per horizon:     │  │ P(ret > 0)   │  │ predicted       │
│ expected return  │  │               │  │ log-volatility  │
│ + uncertainty    │  │               │  │                 │
└────────┬────────┘  └──────┬───────┘  └───────┬─────────┘
         │                  │                   │
         │    ┌─────────────┴───────────────────┴──────────┐
         │    │              RegimeHead                     │
         │    │              128→32→4                       │
         │    │              4-class vol regime             │
         │    │              (0=low → 3=extreme)            │
         │    └────────────────────────────────────────────┘
         │
         ▼
    Horizons: h0=5 bars, h1=10, h2=20, h3=40, h4=60, h5=120
```

---

## Phase D: Signal Generation (ModelOutput → trade signal ∈ [-1, +1])

`DistributionalSignalGenerator.generate(model_output)`:

```
Input: ModelOutput for ONE window (one bar, one instrument)

Step 1 ── Sharpe-like score (at primary horizon h2 = 20 bars)
          s = μ[h2] / σ[h2]
            = expected_return / uncertainty
            e.g. μ=+0.002, σ=0.010 → s=+0.20

Step 2 ── Directional confidence
          p_up = sigmoid(direction_logits[h2])
            e.g. direction_logit=+1.5 → p_up=0.82
          c = 2 × |p_up - 0.5|
            e.g. c = 2 × 0.32 = 0.64    (64% confident)

Step 3 ── Composite signal
          signal = sign(s) × tanh(|s| × c / temperature)
                 = sign(+0.20) × tanh(0.20 × 0.64 / 1.0)
                 = +1.0 × tanh(0.128)
                 = +0.127

          Interpretation:
            +1.0 = strong BUY
             0.0 = HOLD (no conviction)
            -1.0 = strong SELL

Step 4 ── Regime gate
          if predicted_regime == 3 (extreme vol):
              signal = 0.0    ← suppress all trades in chaos

Step 5 ── Threshold gate
          if |signal| < 0.1:
              signal = 0.0    ← suppress weak/uncertain signals

Output: scalar ∈ [-1, +1] per bar per instrument
```

---

## Phase E: How It Works Across Instruments — The Key Insight

### Training Time (one model, pooled data)

The model has **ZERO** per-instrument parameters. No instrument embedding. No per-instrument head. No instrument ID anywhere.

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  XAUUSD window ──┐                                      │
│  BTCUSD window ──┼──►  Same model, same weights ──► signal
│  EURUSD window ──┘                                      │
│                                                         │
│  The model learns: "when channels look like THIS,       │
│   predict THAT" — regardless of which instrument        │
│   produced the channels.                                │
│                                                         │
│  Normalization makes this possible:                     │
│  • Log-ratio removes absolute price level               │
│  • ATR-normalized labels remove scale differences       │
│  • Session features are purely temporal/universal       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Inference Time (one model, deployed independently per instrument)

The same model file is loaded by three separate indicator instances — one per chart. Each instance runs independently on its own instrument's bars.

```
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  XAUUSD chart     │   │  BTCUSD chart     │   │  EURUSD chart     │
│  ┌──────────────┐ │   │  ┌──────────────┐ │   │  ┌──────────────┐ │
│  │ Live M1 bars  │ │   │  │ Live M5 bars  │ │   │  │ Live M1 bars  │ │
│  │ 256 bars →    │ │   │  │ 256 bars →    │ │   │  │ 256 bars →    │ │
│  │ normalize     │ │   │  │ normalize     │ │   │  │ normalize     │ │
│  │ + session     │ │   │  │ + session     │ │   │  │ + session     │ │
│  └──────┬───────┘ │   │  └──────┬───────┘ │   │  └──────┬───────┘ │
│         │         │   │         │         │   │         │         │
│         ▼         │   │         ▼         │   │         ▼         │
│  ┌──────────────┐ │   │  ┌──────────────┐ │   │  ┌──────────────┐ │
│  │ SAME model   │ │   │  │ SAME model   │ │   │  │ SAME model   │ │
│  │ (identical   │ │   │  │ (identical   │ │   │  │ (identical   │ │
│  │  weights)    │ │   │  │  weights)    │ │   │  │  weights)    │ │
│  └──────┬───────┘ │   │  └──────┬───────┘ │   │  └──────┬───────┘ │
│         │         │   │         │         │   │         │         │
│         ▼         │   │         ▼         │   │         ▼         │
│    signal ∈       │   │    signal ∈       │   │    signal ∈       │
│    [-1, +1]       │   │    [-1, +1]       │   │    [-1, +1]       │
│  ┌──────────────┐ │   │  ┌──────────────┐ │   │  ┌──────────────┐ │
│  │ Kelly size → │ │   │  │ Kelly size → │ │   │  │ Kelly size → │ │
│  │ Risk mgr →   │ │   │  │ Risk mgr →   │ │   │  │ Risk mgr →   │ │
│  │ TRADE        │ │   │  │ TRADE        │ │   │  │ TRADE        │ │
│  └──────────────┘ │   │  └──────────────┘ │   │  └──────────────┘ │
└──────────────────┘   └──────────────────┘   └──────────────────┘

╔══════════════════════════════════════════════════════════════════╗
║  THE MODEL IS INSTRUMENT-BLIND                                   ║
║                                                                  ║
║  It does not "know" it's processing XAUUSD vs BTCUSD.            ║
║  It only sees 10 normalized numbers per bar.                     ║
║                                                                  ║
║  The same weights process gold at $2600 and Bitcoin at $90,000   ║
║  because log-ratio normalization collapses both to:              ║
║    "returns relative to recent volatility and time context"      ║
║                                                                  ║
║  This is both the architecture's strength (broad generalization) ║
║  and its vulnerability (no instrument-specific fine-tuning).     ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Phase F: Full Production Pipeline (end-to-end)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING (offline, Python)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  XAUUSD_M1.csv ──┐                                                  │
│  BTCUSD_M5.csv ──┼──► normalize → windows → pool → shuffle          │
│  EURUSD_M1.csv ──┘                     │                            │
│                                        ▼                            │
│                              ┌──────────────────┐                   │
│                              │  Train 50 epochs  │                   │
│                              │  Cosine LR sched  │                   │
│                              │  Early stop on    │                   │
│                              │  val_spearman     │                   │
│                              └────────┬─────────┘                   │
│                                       │                             │
│                                       ▼                             │
│                              ┌──────────────────┐                   │
│                              │  Export to ONNX   │                   │
│                              │  + config.json    │                   │
│                              │  + normalizer.json│                   │
│                              └────────┬─────────┘                   │
│                                       │                             │
└───────────────────────────────────────┼─────────────────────────────┘
                                        │
                                        │  ModelPack/
                                        │  ├── model.onnx
                                        │  ├── config.json
                                        │  ├── normalizer.json
                                        │  └── feature_spec.json
                                        │
┌───────────────────────────────────────┼─────────────────────────────┐
│                     INFERENCE (live, MQL5)                           │
├───────────────────────────────────────┼─────────────────────────────┤
│                                       │                             │
│  ┌────────────────────┐    ┌──────────┴──────────┐                  │
│  │ TransformerModel   │    │ TransformerModel    │   ...3 instances │
│  │ .mq5 (XAUUSD M1)   │    │ .mq5 (BTCUSD M5)    │                  │
│  │                    │    │                     │                  │
│  │ OnCalculate():     │    │ OnCalculate():      │                  │
│  │  1. Collect 256    │    │  1. Collect 256     │                  │
│  │     latest M1 bars │    │     latest M5 bars  │                  │
│  │  2. Normalize      │    │  2. Normalize       │                  │
│  │     (log-ratio)    │    │     (log-ratio)     │                  │
│  │  3. Encode session │    │  3. Encode session  │                  │
│  │     features       │    │     features        │                  │
│  │  4. ONNX inference │    │  4. ONNX inference  │                  │
│  │  5. Signal →       │    │  5. Signal →        │                  │
│  │     trade decision │    │     trade decision  │                  │
│  └────────┬───────────┘    └────────┬───────────┘                  │
│           │                         │                               │
│           ▼                         ▼                               │
│      BUY/SELL/HOLD             BUY/SELL/HOLD                        │
│      for XAUUSD                for BTCUSD                           │
│                                                                     │
│  NOTE: ONNX Runtime loads identical model.onnx for each instance.   │
│  The indicator copies share weights but maintain separate buffers.  │
│  Each runs on its own chart's bar data → independent signals.       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Summary Table

| Aspect | Detail |
|--------|--------|
| **Input shape** | (batch, 256 bars, 10 channels) |
| **Channels** | 5 OHLCV (log-ratio) + hour_sin/cos + dow_sin/cos + has_gap |
| **Patch embedding** | Conv1d(in=10, out=128, kernel=16, stride=8) → 31 patches + [CLS] |
| **Transformer** | 4 layers, 8 heads, d_model=128, d_ff=512, SwiGLU, causal mask |
| **Heads** | Distribution (μ,σ), Direction (logits), Volatility (log), Regime (4-class) |
| **Output** | ModelOutput → DistributionalSignalGenerator → signal ∈ [-1,+1] |
| **Signal formula** | `sign(μ/σ) × tanh(|μ/σ| × confidence / temperature)` |
| **Gates** | Regime gate (zero in extreme vol) + threshold gate (zero weak signals) |
| **Training** | 50 epochs, batch=128, LR=2e-4 cosine, target=atr_score, val=0.1 random split |
| **Cross-instrument** | Yes — single model, all instruments pooled, no per-instrument parameters |
| **ONNX export** | model.onnx consumed by MQL5 TransformerModel.mq5 indicator |
| **Deployment** | One ONNX model file, loaded by multiple MQL5 indicator instances per chart |

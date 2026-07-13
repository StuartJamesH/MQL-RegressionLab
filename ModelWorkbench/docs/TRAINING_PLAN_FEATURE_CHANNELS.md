# Training Plan: Feature Channel Experiments

> **Created:** 2026-07-12 | **Branch:** `train-v2` | **Based on:** Run 3 baseline (Spearman 0.237)
>
> **Principle:** Change exactly ONE variable per run. Record everything. Kill dead runs early.
> **Primary metric:** Validation Spearman (higher = better). Secondary: Direction Accuracy (>55% = edge).

---

## Context

The v2 Causal Patch Transformer currently sees **9 input channels**:
- 5 OHLCV channels: log-ratio pricing (O, H, L, C / Close[t-1]) + rolling-median-scaled volume
- 4 session channels: cyclical sin/cos of hour + day-of-week

The v1 LGBM pipeline uses ~200 hand-crafted features (RSI, MACD, ADX, Bollinger Bands, etc.).
This plan tests whether adding structured features to the transformer input improves performance
beyond what the Conv1d patch embedding can learn from raw OHLCV alone.

**Why minimal first:** The transformer's `PatchEmbedding` (Conv1d over 16-bar windows) is designed to
learn representations from raw price. Adding too many pre-computed features risks overwhelming the
convolution, introducing redundancy, and breaking the stationarity that log-ratio normalization provides.
Every feature added must justify itself against this cost.

---

## Phase 0: Pre-Flight Checks

Before any run, verify the environment:

```bash
cd ModelWorkbench

# 1. Tests pass
../.venv/bin/python -m pytest tests/v2/ -q

# 2. GPU available
../.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')"

# 3. Data exists
ls -lh ../data/*.csv
```

---

## Phase 1: Baseline — Reproduce and Extend Run 3

Goal: Establish a solid baseline with the Run 3 config on all 3 instruments, then push data volume
to find the asymptote before testing any feature additions.

### Run 5 — Run 3 Reproduction on 3 Instruments (Baseline)

**What this tests:** Reproducibility of Run 3's Spearman 0.237 with 3 instruments at 250K rows each.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--ds-names` | XAUUSD_M1, BTCUSD_M5, EURUSD_M1 | Three instruments, proven in Run 4 |
| `--n-rows` | 250000 | 250K/instrument → ~750K windows pooled |
| `--seq-len` | 256 | Proven in Run 3 |
| `--d-model` | 128 | Sweet spot |
| `--n-layers` | 4 | 6 overfits, 4 generalizes |
| `--n-heads` | 8 | Standard for d_model=128 |
| `--d-ff` | 512 | 4× d_model |
| `--dropout` | 0.2 | Best from Run 3 |
| `--finetune-epochs` | 50 | Was still improving at 30 |
| `--batch-size` | 128 | Proven |
| `--lr` | 2e-4 | Best tested |
| `--direction-weight` | 0.5 | Auxiliary loss helps |
| `--volatility-weight` | 0.1 | Mild auxiliary |
| `--target-type` | atr_score | Critical — no alternative |
| `--val-split` | 0.1 | Standard 90/10 |
| `--model-name` | run5_baseline_3inst | |

**Command:**
```bash
cd ModelWorkbench
../.venv/bin/python -u train_transformer.py \
    --ds-names ../data/XAUUSD_M1_520weeks.csv ../data/BTCUSD_M5_260weeks.csv ../data/EURUSD_M1_520weeks.csv \
    --n-rows 250000 --seq-len 256 --d-model 128 --n-layers 4 --n-heads 8 --d-ff 512 \
    --dropout 0.2 --finetune-epochs 50 --batch-size 128 --device cuda --val-split 0.1 \
    --lr 2e-4 --direction-weight 0.5 --volatility-weight 0.1 --target-type atr_score \
    --output-dir ModelPacks/transformers --model-name run5_baseline_3inst
```

**Estimated duration:** ~50 min (50 epochs × ~1 min/epoch for ~750K windows at d=128, L=4)

**Success criteria:** Spearman >= 0.20 at epoch 30+, direction accuracy >= 55%. Kill if val loss flat for 5+ epochs before epoch 20.

**Expected Spearman:** 0.20-0.25 (Run 3 had 0.237 on 2 instruments, Run 4 had 0.155 on 3 with less data)

---

### Run 6 — Max Data, Max Epochs

**What this tests:** Whether the model saturates or continues improving when given more data. This is the
safest path to improvement — more data always beats more parameters per Run 3 vs Run 4.

| Parameter | Value | Δ from Run 5 |
|-----------|-------|-------------|
| `--n-rows` | 500000 | ↑ 250K → 500K/instrument |
| `--finetune-epochs` | 60 | ↑ 50 → 60 |
| `--model-name` | run6_maxdata_60ep | |

**Command:**
```bash
cd ModelWorkbench
../.venv/bin/python -u train_transformer.py \
    --ds-names ../data/XAUUSD_M1_520weeks.csv ../data/BTCUSD_M5_260weeks.csv ../data/EURUSD_M1_520weeks.csv \
    --n-rows 500000 --seq-len 256 --d-model 128 --n-layers 4 --n-heads 8 --d-ff 512 \
    --dropout 0.2 --finetune-epochs 60 --batch-size 128 --device cuda --val-split 0.1 \
    --lr 2e-4 --direction-weight 0.5 --volatility-weight 0.1 --target-type atr_score \
    --output-dir ModelPacks/transformers --model-name run6_maxdata_60ep
```

**Estimated duration:** ~3.5 hours (60 epochs × ~3 min/epoch for ~1.5M windows)

**Expected Spearman:** 0.22-0.28 if data scaling continues to help. If Spearman < Run 5, the model is
at capacity and needs architectural changes (e.g. d_model=256) with more data — but that's a different
experiment series.

**Decision point:** Run 6's best Spearman becomes the **feature experiment baseline**. All subsequent
runs compare against this number, not against Run 3.

---

## Phase 2: Session Regime Flags

Goal: Test whether explicit session context (Tokyo, London, New York) improves predictions beyond
what the model can infer from cyclical hour sin/cos alone.

**Requires code change:** The transformer currently uses `session_channels=4` (hour_sin, hour_cos, 
dow_sin, dow_cos). Adding 3 binary session flags (tokyo, london, ny) requires:

1. `ModelConfig.session_channels` increased from 4 → 7
2. `SessionFeatureEncoder` extended with `encode_sessions()` method that adds 3 boolean columns for
   whether the current timestamp falls within each session window
3. `prepare_ohlcv_windows()` in `train_transformer.py` updated to call the new encoder

These are bounded [0, 1], non-computable from price, and provide direct regime context. Low risk.

### Run 7 — Session Regime Flags (Tokyo, London, NY)

**What this tests:** Whether explicit session binary flags improve on the cyclical sin/cos time encoding.

| Parameter | Value | Δ from Run 6 |
|-----------|-------|-------------|
| `session_channels` | 7 (code change) | 4 → 7 |
| `--model-name` | run7_session_flags | |

**Command (needs code change first — see implementation notes below):**
```bash
cd ModelWorkbench
../.venv/bin/python -u train_transformer.py \
    --ds-names ../data/XAUUSD_M1_520weeks.csv ../data/BTCUSD_M5_260weeks.csv ../data/EURUSD_M1_520weeks.csv \
    --n-rows 500000 --seq-len 256 --d-model 128 --n-layers 4 --n-heads 8 --d-ff 512 \
    --dropout 0.2 --finetune-epochs 60 --batch-size 128 --device cuda --val-split 0.1 \
    --lr 2e-4 --direction-weight 0.5 --volatility-weight 0.1 --target-type atr_score \
    --output-dir ModelPacks/transformers --model-name run7_session_flags
```

**Implementation notes:** `SessionFeatureEncoder.encode()` currently returns (n_bars, 4). Extend to
return (n_bars, 7) by appending 3 binary columns. Tokyo session: 00:00–09:00 GMT. London: 08:00–17:00
GMT. New York: 13:00–22:00 GMT. Note that sessions overlap (London/NY from 13:00–17:00) — the model
can learn the interaction.

**Expected impact:** Low-Medium positive. Session context should help on M1 data (intraday patterns)
more than M5 or higher timeframes. If Spearman improves by >0.02, consider adding session flags as
a permanent input channel.

---

## Phase 3: Volatility Context

Goal: Test whether providing a direct ATR-based volatility signal helps the patch embedding.

**Requires code change:** `ModelConfig.in_channels` increased from 5 → 6. The 6th channel is `atr14_pct`
(ATR(14) / Close as a percentage). This is bounded, approximately stationary, and gives the Conv1d an
explicit volatility anchor alongside the raw OHLCV.

### Run 8 — ATR% Companion Channel

**What this tests:** Whether explicit volatility context (ATR%) improves the model beyond what it can
infer from raw OHLCV range bars.

| Parameter | Value | Δ from Run 7 |
|-----------|-------|-------------|
| `in_channels` | 6 (code change) | 5 → 6 (add atr14_pct) |
| `--model-name` | run8_atr_channel | |

**Command (needs code change):**
```bash
cd ModelWorkbench
../.venv/bin/python -u train_transformer.py \
    --ds-names ../data/XAUUSD_M1_520weeks.csv ../data/BTCUSD_M5_260weeks.csv ../data/EURUSD_M1_520weeks.csv \
    --n-rows 500000 --seq-len 256 --d-model 128 --n-layers 4 --n-heads 8 --d-ff 512 \
    --dropout 0.2 --finetune-epochs 60 --batch-size 128 --device cuda --val-split 0.1 \
    --lr 2e-4 --direction-weight 0.5 --volatility-weight 0.1 --target-type atr_score \
    --output-dir ModelPacks/transformers --model-name run8_atr_channel
```

**Implementation notes:** In `prepare_ohlcv_windows()`, after computing `X_raw = normalize_ohlcv(df)` 
which produces (n_bars, 5), append a 6th column: `atr14 = talib.ATR(high, low, close, 14); atr_pct = atr14 / close`. 
Normalize atr_pct to z-score within a rolling 252-bar window (causal). The `create_sliding_windows` 
Numba kernel is already channel-agnostic — it uses `data.shape[1]` dynamically. Set `ModelConfig.in_channels=6`.

**Expected impact:** Low-Medium positive. ATR provides a direct volatility baseline that the Conv1d can
use as a scaling reference. May particularly help with outlier bars during high-volatility regimes.

---

## Phase 4: Controlled Indicator Ablation

Goal: Test the hypothesis that the transformer benefits from a small set of well-chosen technical
indicators versus learning everything from raw OHLCV. This is the riskiest experiment — avoid
dumping 50+ indicators. Start with 3-4 carefully chosen ones.

**Requires code change:** `ModelConfig.in_channels` increased from 5/6 to ~10-12. Add RSI(14),
MACD histogram (normalized by ATR), ADX(14), and optionally a volatility ratio (RV_5 / RV_20).

### Run 9 — Core Indicators Only (RSI + MACD + ADX)

**What this tests:** Whether a small set of proven indicators provides useful inductive bias.

| Parameter | Value | Δ from Run 8 |
|-----------|-------|-------------|
| `in_channels` | ~10 (code change) | +4 indicator channels |
| `--model-name` | run9_core_indicators | |

**Channels to add (normalized to approximate stationarity):**
1. RSI(14) → already bounded [0, 100], mapped to [0, 1]
2. MACD histogram → `(MACD_line - MACD_signal) / ATR(14)` — dimensionless
3. ADX(14) → already bounded [0, 100], mapped to [0, 1]
4. Volatility regime ratio → `RV_5 / RV_20` — identifies vol expansion/contraction

**Command (needs code change):**
```bash
cd ModelWorkbench
../.venv/bin/python -u train_transformer.py \
    --ds-names ../data/XAUUSD_M1_520weeks.csv ../data/BTCUSD_M5_260weeks.csv ../data/EURUSD_M1_520weeks.csv \
    --n-rows 500000 --seq-len 256 --d-model 128 --n-layers 4 --n-heads 8 --d-ff 512 \
    --dropout 0.2 --finetune-epochs 60 --batch-size 128 --device cuda --val-split 0.1 \
    --lr 2e-4 --direction-weight 0.5 --volatility-weight 0.1 --target-type atr_score \
    --output-dir ModelPacks/transformers --model-name run9_core_indicators
```

**Expected impact:** Mixed. This is an ablation — the result tells us whether indicator shortcuts help
or hurt. If Spearman improves > 0.03, indicators provide useful inductive bias. If Spearman drops,
the transformer learns better representations from raw OHLCV alone and indicators add noise.

**Decision point:** If Run 9 degrades performance, skip Run 10 and do not add more indicators.

### Run 10 — If Run 9 passes: Add Price Location + Donchian Context (Optional)

Only run if Run 9 shows positive delta over Run 8. Adds 4 more channels:

5. Price location: `(Close - Donchian_Low_20) / (Donchian_High_20 - Donchian_Low_20)` → [0, 1]
6. Donchian trend: `+1/-1` depending on whether Donchian midline is above/below EMA(21)
7. Candle body ratio: `abs(Close - Open) / (High - Low + eps)` → [0, 1]
8. HL range z-score: `(High - Low) / rolling_mean(High-Low, 60)` → normalized range expansion

| Parameter | Value | Δ from Run 9 |
|-----------|-------|-------------|
| `in_channels` | ~14 (code change) | +4 more indicator channels |
| `--model-name` | run10_extended_indicators | |

**Expected impact:** Diminishing returns. If Run 9 is positive, Run 10 tests whether the marginal
benefit continues or saturates.

---

## Phase 5: Multi-Timeframe Fusion

Goal: Test whether cross-attention with higher-timeframe data provides structurally novel information.

**Requires significant code change:** Unlike previous phases, this is not a channel addition — it
requires feeding a second stream of data (e.g., H1 bars for M5 training) through a separate
patch embedding and fusing via cross-attention in the `MTFFusionModule` (`model/mtf_fusion.py`).
`use_mtf_fusion=True` must be set in ModelConfig, and the training loop must prepare HTF windows.

This is the highest-effort, potentially highest-reward experiment. The TRAINING_LOG marks it as
"theoretically valuable, not yet tested."

### Run 11 — Single HTF Cross-Attention (M5 → H1)

**What this tests:** Whether the model benefits from "zooming out" to a higher timeframe for macro
context while maintaining fine-grained M5 predictions.

**Approach:**
1. For each M5 window (256 bars, ~21 trading hours), build a corresponding H1 window by resampling
   the same time period into ~21 H1 bars
2. Pad/shape H1 windows to a fixed length (e.g., 32 bars)
3. Pass H1 through a separate `PatchEmbedding` with a different `patch_len` (e.g., 4 H1 bars)
4. Fuse via `MTFFusionModule`: M5 patch tokens query-attend to H1 patch tokens

**Expected impact:** Medium. The TRAINING_LOG estimates "theoretically valuable." MTF context should
help with trend identification — the M5 model might see a strong uptrend as a series of noisy
pullbacks, but the H1 context reveals it as a trending regime. Most valuable for horizons 40-120 bars.

**⚠️ This experiment requires substantial implementation work and is deferred until Phases 1-4
are complete and analyzed.** The MTF fusion scaffolding exists in the codebase
(`model/mtf_fusion.py`, `config.use_mtf_fusion`) but the training loop integration needs to be written.

---

## Phase 6: Synthesis — Best Config Production Run

After completing Phases 1-4 (and optionally Phase 5), take the best-performing configuration and:

### Run 12 — Production Polish

| Parameter | Value |
|-----------|-------|
| Config | Best from all experiments |
| `--finetune-epochs` | 80 (max — let it converge fully) |
| `--n-rows` | Max available for all instruments |
| `--ds-names` | All 5 instruments (XAU, BTC, EUR, US500, SpotCrude) at best timeframe |
| `--model-name` | run12_production |

**Command (example — fill in best config):**
```bash
cd ModelWorkbench
../.venv/bin/python -u train_transformer.py \
    --ds-names ../data/XAUUSD_M1_520weeks.csv ../data/BTCUSD_M5_260weeks.csv ../data/EURUSD_M1_520weeks.csv ../data/US500_M1_520weeks.csv ../data/SpotCrude_M1_520weeks.csv \
    --n-rows 500000 --seq-len 256 --d-model 128 --n-layers 4 --n-heads 8 --d-ff 512 \
    --dropout 0.2 --finetune-epochs 80 --batch-size 128 --device cuda --val-split 0.1 \
    --lr 2e-4 --direction-weight 0.5 --volatility-weight 0.1 --target-type atr_score \
    --output-dir ModelPacks/transformers --model-name run12_production
```

---

## Summary Table

| Run | Phase | What Changed | Expected Impact | Requires Code Change | Est. Duration |
|-----|-------|-------------|-----------------|---------------------|---------------|
| **5** | Baseline | Run 3 config on 3 instruments, 250K/instr, 50 ep | Spearman 0.20-0.25 | No | ~50 min |
| **6** | Baseline | 500K/instr, 60 epochs | Spearman 0.22-0.28 | No | ~3.5 hrs |
| **7** | Session | +3 session regime binary flags | Low-Medium positive | Yes (small) | ~3.5 hrs |
| **8** | Volatility | +1 ATR% channel | Low-Medium positive | Yes (small) | ~3.5 hrs |
| **9** | Indicators | +4 core indicators (RSI, MACD, ADX, VolRatio) | Uncertain (ablation) | Yes (medium) | ~4 hrs |
| **10** | Indicators | +4 extended indicators (if Run 9 passes) | Diminishing returns | Yes (medium) | ~4 hrs |
| **11** | MTF Fusion | H1 cross-attention for M5 model | Medium (theoretically) | Yes (large) | TBD |
| **12** | Production | Best config, all instruments, 80 epochs | Max achievable | No (uses best config) | ~5-6 hrs |

---

## Rules of Engagement

1. **One variable per run.** If Run 6 changes both `n_rows` and `epochs`, that's the exception —
   they're both "more training." For Runs 7+, change only the feature set, keeping all other
   hyperparameters identical to the best previous run.

2. **Kill early.** Val loss flat for 5 epochs? Stop. Train Spearman rising while Val Spearman flat
   or negative? Stop. NaN or Inf? Stop.

3. **Record everything in TRAINING_LOG.md.** After each run, add a new entry following the existing
   format: date, duration, config block, epoch metrics table, result diagnosis, and decision.

4. **Compare the right baseline.** Run 7 compares against Run 6 (or the best baseline). Run 8
   compares against the best of {Run 6, Run 7}. Never compare against Run 3 directly after
   Phase 1 is complete.

5. **Save per-epoch metrics.** Every run saves `metrics.jsonl` to `ModelPacks/transformers/<model_name>/`.
   Per-horizon Spearman breakdowns are critical for diagnosing whether features help specific horizons.

6. **If a feature degrades performance, remove it before the next run.** Don't carry a losing feature
   forward — each run should test against the current best configuration.

7. **Run 7 can skip Run 8's features if Run 7 fails.** The plan is sequential but conditional:
   if session flags don't help, test ATR without them (Run 8 compares against Run 6 baseline,
   not Run 7). Document which baseline was used.

---

## Data Inventory

| Instrument | M1 (520 weeks) | M5 (260 weeks) | ~Bars (M1/M5) |
|------------|---------------|----------------|---------------|
| XAUUSD | ✅ | ✅ | ~524K / ~105K |
| BTCUSD | ✅ | ✅ | ~524K / ~105K |
| EURUSD | ✅ | ✅ | ~524K / ~105K |
| US500 | ✅ | ✅ | ~524K / ~105K |
| SpotCrude | ✅ | ✅ | ~524K / ~105K |

**For Run 12 (all 5 instruments, M1):** ~2.6M bars total → ~2.3M overlapping windows at seq_len=256.
Estimate ~12 GB RAM usage. Monitor closely.

**For M5-only experiments (Runs 7-10):** Available but not recommended — M1 gives the model more
bars per instrument and is the primary training frequency. Mix M1+M5 as Run 5/6 do.

---

## Quick-Reference: Kill Conditions

| Symptom | Threshold | Action |
|---------|-----------|--------|
| Val loss flat | < 0.0001 change over 5 epochs | Kill — model stalled |
| Val Spearman < 0 at epoch 5+ | Negative correlation | Kill — learnable signal absent |
| Train loss NaN/Inf | Any occurrence | Kill — numerical instability |
| Grad norm > 1000 pre-clip | Consistently high | Kill or reduce LR |
| Train Spearman rising, Val Spearman flat | Divergence for 5+ epochs | Kill — overfitting |

---

## Post-Experiment Deliverables

When all runs are complete:

1. **Updated TRAINING_LOG.md** with all run entries
2. **Best model checkpoint** saved as `ModelPacks/transformers/production/model.pt`
3. **Per-horizon Spearman breakdown** for the best model
4. **ONNX export** ready for MQL5 deployment via `DeploymentPackager`
5. **Decision record:** Which features helped, which didn't, and the final recommended input channel set

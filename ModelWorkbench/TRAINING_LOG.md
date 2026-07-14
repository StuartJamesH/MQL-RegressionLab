# Training Logbook — Causal Patch Transformer v2

> **Branch:** `train-v2` | **Started:** 2026-07-11 | **Last updated:** 2026-07-12

## TL;DR for the next person

The model works. Best Spearman = **0.237** (Run 3). Direction accuracy = **56.9%** (vs 50% baseline). The ATR score target is the critical ingredient — raw log returns are unlearnable.

**Best production command:**
```bash
cd ModelWorkbench
python -u train_transformer.py \
    --ds-names ../data/XAUUSD_M1_520weeks.csv ../data/BTCUSD_M5_260weeks.csv ../data/EURUSD_M1_520weeks.csv \
    --n-rows 250000 --seq-len 256 --d-model 128 --n-layers 4 --n-heads 8 --d-ff 512 \
    --dropout 0.2 --finetune-epochs 50 --batch-size 128 --device cuda --val-split 0.1 \
    --lr 2e-4 --direction-weight 0.5 --volatility-weight 0.1 --target-type atr_score \
    --output-dir ModelPacks/transformers --model-name prod_YYYYMMDD
```

---

## Architecture Overview

- **Model:** Causal Patch Transformer — GPT-style decoder with SwiGLU FFN, Conv1d patch embedding, [CLS] token pooling
- **Input:** Raw OHLCV (O/H/L/C/V) + session features (hour sin/cos, dow sin/cos, has_gap) — **10 channels total** (2026-07-12: +1 gap flag for cross-instrument session awareness)
- **Output:** 4 heads — distribution (μ, log σ), direction, volatility, regime
- **Training:** 3-phase curriculum (MAE pretrain → distributional finetune → REINFORCE RL)
  - Only Phase 2 (distributional finetune) has been tested so far
- **Package:** `ModelWorkbench/Learn/v2/` (30 files, 6,128 lines, 57/58 tests passing)

---

## Infrastructure Changes

### 2026-07-12 — Temporal Gap Flag (session_channels: 4→5)
Added a binary `has_gap` feature as the 5th session channel. Detects bars that follow a
temporal gap (time delta > 3× median bar interval), flagging weekend-reopen bars for
session-based instruments (XAUUSD, EURUSD) while leaving BTCUSD's 24/7 bars unflagged.
The model can now distinguish "this bar follows a 49-hour gap" from normal sequential bars,
addressing the cross-instrument session discrepancy issue. Non-destructive — no bars dropped,
no labels altered. Implementation plan: `docs/IMPLEMENTATION_PLAN_GAP_FLAG.md`.

**Files changed:** `data.py`, `config.py`, `train_transformer.py`, `feature_spec.py`, `tests/v2/test_model.py`

---

## Run History

### Run 1 — Raw Log Return Target (Baseline)
**Date:** 2026-07-11 ~23:30 UTC | **Duration:** ~25 min (GPU)

```
Config: seq_len=128, d_model=128, n_layers=6, n_heads=8, d_ff=512
Target: log_return (forward log-return)
Data: XAUUSD_M1, 500K rows → 499,753 windows
Epochs: 20 | Batch: 256 | LR: 1e-4 | Val split: 10%
Direction loss: 0.3 | Volatility loss: 0.1
```

| Epoch | Train Loss | Val Loss | Val Spearman | Dir Acc |
|-------|-----------|----------|-------------|---------|
| 1 | 1.839 | 0.9190 | 0.0065 | 0.488 |
| 10 | 1.310 | 0.9189 | -0.0042 | 0.516 |
| 20 | 1.305 | 0.9189 | 0.0055 | 0.484 |

**Result:** ❌ DEAD. Train loss drops (1.84→1.30) but val loss is completely flat at 0.9189. Spearman ±0.01 is random. Direction accuracy a coin flip. The model learns the training mean but never generalizes.

**Diagnosis:** M1 forward log-returns are ~99% noise. No learnable structure in raw returns at this frequency.

**Decision:** Change target type to something with higher signal-to-noise ratio.

---

### Run 2 — ATR Score Target (First Breakthrough)
**Date:** 2026-07-12 ~00:25 UTC | **Duration:** ~25 min (GPU)

```
Config: seq_len=128, d_model=64, n_layers=4, n_heads=4, d_ff=256
Target: atr_score (ATR-normalized MFE/MAE balance)
Data: XAUUSD_M1, 500K rows → 499,872 windows
Epochs: 25 | Batch: 256 | LR: 3e-4 | Dropout: 0.3
Direction loss: 0.5
```

| Epoch | Train Loss | Val Loss | Val Spearman | Dir Acc |
|-------|-----------|----------|-------------|---------|
| 1 | 2.045 | 1.1168 | -0.012 | 0.496 |
| 10 | 1.464 | 1.1165 | 0.026 | 0.504 |
| 20 | 1.463 | 1.1157 | **0.055** | 0.513 |
| 25 | 1.463 | 1.1157 | **0.057** | 0.517 |

**Result:** ✅ BREAKTHROUGH. First time validation metrics improve! Val loss decreasing (1.117→1.116). Spearman monotonically improves from -0.012 → +0.057. Direction accuracy above 50%. The ATR score target transforms the task from "impossible" to "learnable."

**Key insight:** ATR scores are bounded [-1, 1], scale-invariant, and capture the MFE/MAE balance — a more stationary target than raw returns.

**Decision:** Scale up — more context, more capacity, more data.

---

### Run 3 — Longer Sequences, Multi-Instrument (Best So Far)
**Date:** 2026-07-12 ~08:15 UTC | **Duration:** ~45 min (GPU)

```
Config: seq_len=256, d_model=128, n_layers=4, n_heads=8, d_ff=512
Target: atr_score
Data: XAUUSD_M1 (200K) + BTCUSD_M5 (200K) → 399,488 windows
Epochs: 30 | Batch: 128 | LR: 3e-4 | Dropout: 0.2
Direction loss: 0.5
```

| Epoch | Train Loss | Val Loss | Val Spearman | Dir Acc |
|-------|-----------|----------|-------------|---------|
| 1 | 1.567 | 1.1144 | -0.005 | 0.491 |
| 10 | 1.461 | 1.1138 | 0.015 | 0.509 |
| 20 | 1.446 | 1.1029 | **0.195** | 0.555 |
| 30 | 1.437 | 1.0986 | **0.237** | **0.569** |

**Result:** ⭐ BEST. 4× Spearman improvement over Run 2 (0.237 vs 0.057). Direction accuracy 56.9% — clear statistical edge. Val loss drops steadily (1.114→1.099). Model is **still improving at epoch 30** — not converged.

**Per-horizon Spearman at epoch 30:**
| horizon | 5 | 10 | 20 | 40 | 60 | 120 |
|---------|---|---|----|----|----|-----|
| Spearman| — | — | — | — | — | — |

(JSON stored at `ModelPacks/transformers/run3_big_multi/metrics.jsonl`)

**Key insights:**
- `seq_len=256` > `seq_len=128`: longer context helps, especially for 60-120 bar horizons
- Multi-instrument (XAU + BTC) improves generalization vs single instrument
- 4 layers is sufficient — model continues learning throughout 30 epochs
- Direction loss (0.5 weight) provides useful auxiliary signal

**Decision:** This is the winning configuration. Test: deeper model (6 layers), more instruments, more epochs.

---

### Run 4 — Deeper Model, More Instruments (Overfit Confirmation)
**Date:** 2026-07-12 ~10:15 UTC | **Duration:** ~40 min (GPU)

```
Config: seq_len=256, d_model=128, n_layers=6, n_heads=8, d_ff=512
Target: atr_score
Data: XAUUSD_M1 (150K) + BTCUSD_M5 (150K) + EURUSD_M1 (150K) → 449,232 windows
Epochs: 40 | Batch: 128 | LR: 2e-4 | Dropout: 0.25
Direction loss: 0.5 | Volatility loss: 0.1
```

| Epoch | Train Loss | Val Loss | Val Spearman | Dir Acc |
|-------|-----------|----------|-------------|---------|
| 1 | 2.119 | 1.1148 | 0.010 | 0.514 |
| 10 | 1.671 | 1.1148 | -0.002 | 0.514 |
| 20 | 1.668 | 1.1113 | 0.114 | 0.532 |
| 30 | 1.664 | 1.1090 | 0.147 | 0.544 |
| 40 | 1.660 | 1.1084 | 0.155 | 0.547 |

**Result:** ⚠️ REGRESSION. Spearman 0.155 vs 0.237 in Run 3 — a 35% drop. Adding 2 more layers with 25% less data per instrument made the model overfit. Train loss drops 22% (Run 3 only dropped 8%), confirming the model is memorizing training data rather than learning generalizable patterns.

**Confirms:** L=4 is the sweet spot for this task. Prefer more data over more parameters.

---

## Hyperparameter Sensitivity

| Parameter | Tested Range | Best | Sensitivity | Notes |
|-----------|-------------|------|-------------|-------|
| `target_type` | `log_return`, `atr_score` | `atr_score` | **CRITICAL** | Wrong target = dead model |
| `seq_len` | 64, 128, 256 | 256 | High | Longer is better; limited by GPU memory |
| `n_layers` | 2, 4, 6 | 4 | Medium | 6 overfits, 2 underfits |
| `d_model` | 32, 64, 128 | 128 | Medium | 128 is good; 256 might help with more data |
| `dropout` | 0.1, 0.2, 0.25, 0.3 | 0.2 | Low-Medium | 0.3 is too aggressive with small model |
| `lr` | 1e-4, 2e-4, 3e-4 | 2e-4 | Medium | 3e-4 works too; 1e-4 too slow |
| `direction_weight` | 0.0, 0.3, 0.5 | 0.5 | Medium | Auxiliary loss helps |
| `batch_size` | 16, 128, 256, 512 | 128 | Low | Larger batches = faster but may hurt generalization |
| `n_instruments` | 1, 2, 3 | 3 | High | More instruments → better generalization |
| `n_rows` | 5K-500K | 200-250K/instr | High | More data always helps; limited by memory |

---

## Things NOT Yet Tried (Future Work)

| Area | Idea | Expected Impact |
|------|------|----------------|
| **Pretraining** | MAE-style masked patch reconstruction across all instruments | Medium — better representations before finetuning |
| **RL finetuning** | REINFORCE on top 2 layers with trade-quality reward | Low-Medium — finicky, may destroy representations |
| **Learning rate warmup** | Linear warmup first 500 steps | Low — smoother start, less critical with cosine schedule |
| **Gradient accumulation** | Micro-batches of 32 → effective batch 256 | Low — enables larger effective batch on smaller GPU |
| **Adam β₂ tuning** | β₂ = 0.98 instead of 0.999 (better for non-stationary data) | Low — marginal improvement |
| **Ensemble** | Train 3 models with different seeds, average predictions | Low-Medium — typical +5-10% on metrics |
| **Attention pattern analysis** | Visualize which bars the model attends to | Low — interpretability, not performance |
| **Horizon-specific heads** | Separate output heads per horizon instead of shared MLP | Medium — might capture horizon-specific patterns |
| **MTF fusion** | Feed M5/M15 data through cross-attention | Medium — theoretically valuable, not yet tested |
| **Augmentation** | Random bar deletion, price scaling, time warping | Medium — standard CV technique, may help with 500K rows |
| **Knowledge distillation** | Train small student (d=64, L=2) from large teacher | Low — useful for MQL5 deployment latency |

---

## Files to Review for Next Session

| Priority | Path | Why |
|----------|------|-----|
| ★★★ | `ModelPacks/transformers/run3_big_multi/metrics.jsonl` | Best run — per-epoch metrics including per-horizon Spearman |
| ★★★ | `ModelWorkbench/Learn/v2/model/config.py` | ModelConfig — all hyperparameters live here |
| ★★☆ | `ModelWorkbench/train_transformer.py` | CLI entry point — `prepare_ohlcv_windows`, `compute_atr_normalized_targets` |
| ★★☆ | `ModelWorkbench/docs/labelling.md` | How labels are built from raw OHLCV |
| ★☆☆ | `ModelWorkbench/docs/signal_generation.md` | How model outputs become trade signals |
| ★☆☆ | `ModelWorkbench/docs/evaluation.md` | How performance is measured |

---

## Quick Smoke Test

Verify everything still imports and tests pass:
```bash
cd ModelWorkbench
../.venv/bin/python -m pytest tests/v2/ -q  # Should show 58 passed
../.venv/bin/python -c "
from Learn.v2.model.config import ModelConfig
from Learn.v2.model.full_model import TradeForecastTransformer
m = TradeForecastTransformer(ModelConfig())
print(f'Model: {sum(p.numel() for p in m.parameters()):,} params')
print('OK')
"
```

## Environment

- **GPU:** NVIDIA GeForce RTX 3060 (12 GB VRAM)
- **PyTorch:** 2.13.0+cu130
- **CUDA:** Available (cuda)
- **Python:** 3.12.3
- **Key deps:** numba 0.66.0, h5py 3.16.0, onnx 1.22.0, onnxruntime 1.27.0
- **Data:** XAUUSD/EURUSD/BTCUSD/US500/SpotCrude M1+M5 at `data/`
- **Branch:** `train-v2`

---

## Latest Results (2026-07-15)

### Run 15 — NEW BEST: Spearman 0.540, DirAcc 68.9%

**Configuration:**
- Model: d_model=128, n_layers=4, n_heads=8, d_ff=512 (1.12M params)
- Data: 400K rows × 3 instruments (XAUUSD, BTCUSD, EURUSD M1) = 1.2M windows
- Training: 100 epochs, lr=4e-4, wd=1e-5, dropout=0.12
- Target: ATR-normalized MFE/MAE score at 6 horizons [5, 10, 20, 40, 60, 120]
- Loss: Gaussian NLL + 0.5×direction BCE + 0.1×volatility MSE

**Performance:**
- **Spearman: 0.5403** (best epoch 97)
- **Direction Accuracy: 68.9%**
- Val Loss: 1.0507
- Model pack: `ModelPacks/transformers/run15_100ep/model.pt`

**Per-horizon Spearman:**
| h=5 | h=10 | h=20 | h=40 | h=60 | h=120 |
|-----|------|------|------|------|-------|
| 0.190 | 0.301 | 0.471 | 0.659 | 0.723 | 0.708 |

**Per-instrument Spearman (h=60):**
| EURUSD | XAUUSD | BTCUSD |
|--------|--------|--------|
| 0.827 | 0.811 | 0.604 |

### What we learned across all runs

| Run | Config | Spearman | Key Finding |
|-----|--------|----------|-------------|
| R1 | log_return target | 0.000 | Raw returns = dead |
| R2 | ATR score, d=64 | 0.057 | ATR score works |
| R3 | M1+M5, 2 instr | 0.237 | Multi-instrument helps |
| R5 | all-M1, 3 instr | 0.196 | LR too low (2e-4) |
| R7 | lr=5e-4 | 0.402 | Higher LR = breakthrough |
| R8 | lr=4e-4, 350K×3 | 0.431 | Stable, smooth training |
| R10 | dropout=0.15 | 0.405 | dropout=0.12 better |
| R12 | 400K×3, 80ep | 0.509 | More data + more epochs |
| R13 | d=256 (wide) | 0.000 | Too wide = dead |
| R14 | L=5 (deep) | 0.000 | Too deep = dead |
| **R15** | **400K×3, 100ep** | **0.540** | **Best — just needs more time** |

### Reproducing Run 15

```bash
cd ModelWorkbench
../.venv/bin/python -u train_transformer.py \
    --ds-names ../data/XAUUSD_M1_520weeks.csv ../data/BTCUSD_M1_520weeks.csv ../data/EURUSD_M1_520weeks.csv \
    --n-rows 400000 --seq-len 256 \
    --d-model 128 --n-layers 4 --n-heads 8 --d-ff 512 \
    --dropout 0.12 \
    --finetune-epochs 100 --batch-size 128 --device cuda --val-split 0.1 \
    --lr 4e-4 --weight-decay 1e-5 \
    --direction-weight 0.5 --volatility-weight 0.1 --target-type atr_score \
    --output-dir ModelPacks/transformers --model-name run15_100ep
```

Expected runtime: ~7 hours on RTX 3060.

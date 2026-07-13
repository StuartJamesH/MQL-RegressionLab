---
name: model-training
description: Use when the user asks to train a model, tune hyperparameters, run ML experiments, or improve model performance. Covers training loop design, target/label engineering, model sizing vs data tradeoffs, overfitting diagnosis, metric monitoring, and systematic iteration. References the project's Learn/v2 Causal Patch Transformer and MQL-RegressionLab training infrastructure.
---

# Model Training Best Practices

Based on the MQL-RegressionLab Causal Patch Transformer training runs and
iterative hyperparameter tuning.  These guidelines apply to both the v2
transformer pipeline (`Learn/v2/`) and any future model training in this repo.

**Before starting any training work, read the project training logbook at
`ModelWorkbench/TRAINING_LOG.md` — it records prior run configurations,
results, and dead-ends.  Do not rediscover what is already known.**

---

## 1. Target Engineering Is the Most Important Decision

A wrong target produces a dead model regardless of architecture.  A good target
makes the problem learnable.

### Anti-patterns (proven failures)

| Target | Why it fails |
|--------|-------------|
| Raw forward log-returns on M1 data | 99% noise; model learns the training mean and never generalizes. Val loss flat at every epoch. |
| Ternary TP/SL/timeout labels | Discards information about *how strongly* price moved. Model can't learn a smooth gradient. |

### Proven target: ATR-normalized MFE/MAE score

```python
score = (buy_MFE - buy_MAE) / max(buy_MFE + buy_MAE, 1e-8)
```

- **Bounded**: Range [-1, +1] — the model only needs to predict in a narrow range
- **Scale-invariant**: ATR normalization removes instrument and time-period effects
- **Smooth gradient**: +0.9 → +0.5 → +0.1 → -0.1 provides more learning signal than win/loss/timeout
- **Trade-relevant**: Directly measures the balance of favorable vs adverse excursion

See `ModelWorkbench/docs/labelling.md` for the full step-by-step derivation.

### Decision tree for target selection

```
Can your target be expressed as a smooth, bounded, stationary score?
  ├── YES → Use it directly as a regression target
  └── NO  → Transform it:
             ├── Normalize by rolling ATR (removes scale effects)
             ├── Apply a bounded non-linearity (tanh, clip, normalize to [-1,1])
             └── Consider multi-task: predict distribution + direction jointly
```

---

## 2. Model Capacity vs Data — The Sweet Spot

### The rule of thumb

For financial time series with ~500K windows:
- **d_model=128, n_layers=4** (~1.1M params) is the sweet spot
- **n_layers=6** (~1.6M params) overfits — train loss drops faster, val metrics plateau earlier
- **n_layers=2** (~115K params) underfits — neither train nor val metrics move much
- **d_model=64** (~300K params) learns but more slowly — useful for quick iteration

### Signs of overfitting (kill the run)

| Symptom | What it looks like |
|---------|-------------------|
| Train loss dropping, val loss **completely flat** | Model memorizing training noise. Val loss unchanged across 10+ epochs on a scale where train loss dropped 30%. |
| Train Spearman rising, val Spearman **flat or negative** | Model learns training-specific patterns that don't generalize. |
| Val metrics peak early then degrade | Classic overfitting. The best checkpoint is not the last one. |

### Signs of underfitting (increase capacity)

| Symptom | What it looks like |
|---------|-------------------|
| Train and val loss both flat at a high value | Model too small to capture structure. Increase d_model or n_layers. |
| Train loss not decreasing past epoch 5 | Learning rate too low or model too small. |
| Gradient norms < 0.1 consistently | Vanishing gradients — check architecture, initialization, or increase capacity. |

### Data quantity over model quality

**More training data almost always beats a bigger model.** In our runs,
Run 3 (d=128, L=4, 200K rows/instrument, 2 instruments) achieved Spearman
0.237.  Run 4 (d=128, L=6, 150K rows/instrument, 3 instruments) only got
0.155 — the extra layers couldn't compensate for 25% less data.

---

## 3. The Iterative Tuning Workflow

### Phase A: Get a single run working (1-2 runs)

1. Start with **200K rows, seq_len=128, d_model=64, n_layers=2, batch_size=256**
2. Use the known-good target (`atr_score`) — do NOT experiment with targets yet
3. Run 5 epochs. If train loss decreases and val loss is NOT flat, you're good.
4. If val loss is flat at epoch 5, the target or data pipeline is broken — debug before tuning.

### Phase B: Scale one dimension at a time (3-5 runs)

Change exactly ONE thing per run.  Record every run in `TRAINING_LOG.md`.

| Dimension to scale | Command flag | Try |
|--------------------|-------------|-----|
| Sequence length | `--seq-len` | 128 → 256 → 512 |
| Model width | `--d-model` | 64 → 128 → 256 |
| Model depth | `--n-layers` | 2 → 4 → 6 |
| Data per instrument | `--n-rows` | 150K → 250K → 500K |
| Number of instruments | `--ds-names` | 1 → 2 → 3 |
| Learning rate | `--lr` | 3e-4, 2e-4, 1e-4 |
| Dropout | `--dropout` | 0.1, 0.2, 0.3 |
| Auxiliary losses | `--direction-weight` `--volatility-weight` | 0.0, 0.3, 0.5 |

### Phase C: Polish the best config (1-2 runs)

Once you've found the best config from Phase B:
1. Double the epochs (30 → 50 or 60) — our best run was still improving at epoch 30
2. Add all available instruments
3. Set `--n-rows` as high as memory allows

---

## 4. The Training Loop Must Log These Metrics Every Epoch

The training loop in this project (`train_transformer.py`) already logs all of
these.  If building a new training script, replicate this format:

```
================================================================================
 Epoch | Train Loss |   Val Loss |  Val Sprmn |  Dir Acc |        LR |  Grad Norm
================================================================================
    1/30  |   1.567312 |   1.114387 |    -0.0052 |  0.4913 | 2.99e-04 |   112.9432
    2/30  |   1.462212 |   1.114003 |    -0.0028 |  0.5087 | 2.97e-04 |     4.3054
    ...
   30/30  |   1.436820 |   1.098633 |     0.2365 |  0.5687 | 0.00e+00 |     1.1623
================================================================================
```

### Required per-epoch columns

| Column | Why |
|--------|-----|
| **Train Loss** | Is the model learning at all? Should decrease monotonically. |
| **Val Loss** | Is the model generalizing? Flat = dead. Decreasing = good. |
| **Val Spearman** | The primary trading metric. Should be positive and climbing. |
| **Dir Acc** | Direction accuracy. >55% means the model has directional edge. |
| **LR** | Learning rate. Should follow the schedule. Confirms scheduler is working. |
| **Grad Norm** | Training stability. ~1.0 after clipping = actively learning. <0.1 = stalled. |

### Always save per-epoch metrics to disk

The project saves `metrics.jsonl` to the output directory.  Each line is a JSON
object with the full epoch record.  This is non-negotiable — you cannot debug a
run you cannot inspect.

```json
{"epoch": 1, "train_loss": 1.567, "val_loss": 1.114, "val_spearman_mean": -0.005,
 "val_direction_accuracy": 0.491, "val_spearman_per_horizon": [0.01, 0.10, -0.13, ...],
 "grad_norm_max": 112.9, "learning_rate": 0.000299}
```

### Val split

Use `--val-split 0.1` (10% validation).  The split should be **random**, not
chronological, because within-window causality is handled by the causal
attention mask.  For final evaluation, use walk-forward (chronological) splits.

---

## 5. When to Kill a Run

Do not let a dead run consume GPU hours.  Kill immediately if:

| Condition | Threshold |
|-----------|-----------|
| Val loss unchanged for 5+ epochs | `|val_loss[epoch] - val_loss[epoch-5]| < 0.0001` |
| Val Spearman < 0.0 after epoch 3 | Model is worse than random — target or data is broken |
| Train loss = NaN or Inf | Numerical instability — check log_sigma clamping, gradient clipping |
| Grad norm > 1000 pre-clip | Exploding gradients — reduce LR or check model initialization |
| Train loss not decreasing after epoch 5 | LR too low, model too small, or vanishing gradients |

---

## 6. Reproducibility Checklist

Before starting a run, confirm:

- [ ] Branch is clean (`git status`)
- [ ] Run has a unique `--model-name` (e.g., `run5_seq512_d256`)
- [ ] Using `--target-type atr_score` (NOT `log_return`)
- [ ] `--val-split 0.1` configured
- [ ] `--output-dir ModelPacks/transformers` (so runs are organized)
- [ ] GPU is available and has >4 GB free VRAM
- [ ] `ModelWorkbench/TRAINING_LOG.md` has been reviewed for prior results

After a run completes:

- [ ] Per-epoch metrics saved to `metrics.jsonl`
- [ ] Model checkpoint saved to `model.pt`
- [ ] Run recorded in `TRAINING_LOG.md` with config, key metrics, and diagnosis
- [ ] Best epoch stats noted (Spearman, Dir Acc, horizon breakdown)

---

## 7. GPU Speed Reference

| Model config | Batch size | ms/batch (RTX 3060) | ~min/epoch (500K windows) |
|-------------|-----------|---------------------|--------------------------|
| d=32, L=2 (38K params) | 512 | 13 | 0.2 |
| d=64, L=2 (115K params) | 512 | 13 | 0.2 |
| d=128, L=4 (1.1M params) | 512 | 26 | 0.4 |
| d=128, L=6 (1.6M params) | 256 | ~35 | ~1.0 |
| d=128, L=8 (2.2M params) | 256 | 42 | 1.2 |

Use these to estimate total wall-clock time: `min_per_epoch × epochs = total`.
Add ~5 minutes for data loading + packaging.

---

## 8. Quick Start Command

```bash
cd ModelWorkbench
../.venv/bin/python -u train_transformer.py \
    --ds-names ../data/XAUUSD_M1_520weeks.csv ../data/BTCUSD_M5_260weeks.csv ../data/EURUSD_M1_520weeks.csv \
    --n-rows 200000 --seq-len 256 --d-model 128 --n-layers 4 --n-heads 8 --d-ff 512 \
    --dropout 0.2 --finetune-epochs 30 --batch-size 128 --device cuda --val-split 0.1 \
    --lr 2e-4 --direction-weight 0.5 --volatility-weight 0.1 --target-type atr_score \
    --output-dir ModelPacks/transformers --model-name runN_descriptive_name
```

Replace `runN_descriptive_name` with a unique identifier for each experiment.

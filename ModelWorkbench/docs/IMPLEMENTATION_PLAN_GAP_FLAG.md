# Implementation Plan: Temporal Gap Flag (Option B)

> **Created:** 2026-07-12 | **Branch:** `train-v2`
>
> Adds a binary `has_gap` feature to the session channels so the transformer
> can explicitly distinguish bars that follow a trading-session gap (e.g.,
> weekend close → reopen) from normal sequential bars.

---

## Problem

The v2 pipeline pools windows from instruments with different trading schedules:

| Instrument | Schedule | Weekend gap |
|------------|----------|-------------|
| XAUUSD M1  | Mon–Fri, closes ~21:00 Fri | ~49 hours (Fri 20:54 → Sun 22:00) |
| EURUSD M1  | Mon–Fri, closes ~21:00 Fri | ~48 hours (Fri 20:54 → Sun 21:05) |
| BTCUSD M5  | 24/7 | None |

The pipeline currently:
1. Treats the Sunday gap-open bar's log-return as a normal 1-minute return
2. Provides no signal to the model that "this bar follows a 49-hour gap"
3. Produces labels for Friday-close windows that span the weekend gap
4. Has `dow_sin`/`dow_cos` values that are instrument-ambiguous (Saturday values appear only in BTCUSD)

## Solution

Add a binary `has_gap` feature as the 5th session channel (increasing total
session channels from 4 to 5). The flag is `1` if the time since the previous bar
exceeds a threshold, `0` otherwise.

**Threshold:** 3× the median bar interval, auto-detected per dataset.
- M1 (60s median): threshold = 180s → only flags multi-minute gaps
- M5 (300s median): threshold = 900s → only flags multi-bar gaps

This is a **non-destructive, informative** approach — no bars are dropped, no
labels are altered. The model can learn to discount or ignore the flag as it sees fit.

---

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `Learn/v2/data.py` | `SessionFeatureEncoder.encode()` gains `include_gap` param and `_detect_gaps()` helper | ~+30 |
| `Learn/v2/model/config.py` | `session_channels: int = 4` → `5` | 1 |
| `train_transformer.py` | `prepare_ohlcv_windows()` passes `include_gap=True`; normalizer stats updated | ~5 |
| `Learn/v2/feature_spec.py` | Register `has_gap` in default features | +1 |
| `tests/v2/test_model.py` | Update `session_channels=4` → `5` in fixtures and hardcoded shapes | ~6 |

### Files that auto-adapt (no changes needed)

- `model/embedding.py` — `total_channels = in_channels + session_channels`, computed dynamically
- `model/full_model.py` — reads `config.session_channels`, passes to PatchEmbedding
- `deploy.py` — `dummy_session` shape uses `config.session_channels`
- `training/rl_finetune.py` — channel-agnostic, goes through model encode

### MQL5 parity

The `TransformerModel.mq5` indicator in `MQL5/Indicators/` must be updated to
compute `has_gap` in its feature array. This is a separate task tracked in the
MQL5 parity checklist — the Python training pipeline is the priority.

---

## Implementation Steps

### Step 1: `SessionFeatureEncoder` — gap detection

Add a `_detect_gaps()` static method:
```python
@staticmethod
def _detect_gaps(timestamps: pd.DatetimeIndex, threshold_multiplier: float = 3.0) -> np.ndarray:
    """Return boolean array where True = this bar follows a temporal gap."""
    if len(timestamps) < 2:
        return np.zeros(len(timestamps), dtype=np.float32)
    deltas = timestamps[1:].view('int64') - timestamps[:-1].view('int64')
    deltas_ns = deltas.astype(np.float64)
    median_delta = np.median(deltas_ns[deltas_ns > 0])
    threshold = median_delta * threshold_multiplier
    gaps = np.zeros(len(timestamps), dtype=np.float32)
    gaps[1:] = (deltas_ns > threshold).astype(np.float32)
    return gaps
```

Modify `encode()` signature:
```python
def encode(self, timestamps, include_gap: bool = False,
           gap_threshold_multiplier: float = 3.0) -> np.ndarray:
```

When `include_gap=True`, return (n_bars, 5) instead of (n_bars, 4).

### Step 2: `ModelConfig.session_channels` → 5

### Step 3: `prepare_ohlcv_windows()` → use `include_gap=True`

### Step 4: `FeatureSpec` → register `has_gap`

### Step 5: Tests → update channel count

### Step 6: Run full test suite

```bash
cd ModelWorkbench && ../.venv/bin/python -m pytest tests/v2/ -v
```

---

## Rollback

If the gap flag degrades performance (unlikely — it's purely informative):
1. Revert `session_channels` to 4 in `ModelConfig`
2. Revert `encode(include_gap=False)` default
3. Re-run tests

The flag adds 1 channel to a Conv1d that already handles 9 — the parameter
increase is `patch_len × d_model = 16 × 256 = 4,096` additional weights in the
convolution. Negligible relative to ~1.1M total parameters.

---

## Verification

After implementation, confirm:

- [ ] `encoder.encode(times, include_gap=True).shape[1] == 5`
- [ ] Gap flag = 1 for the first bar after a weekend (XAUUSD Sun 22:00)
- [ ] Gap flag = 0 for bars within a continuous trading session
- [ ] `tests/v2/` passes with zero failures
- [ ] `python -c "from Learn.v2.model.full_model import TradeForecastTransformer; m = TradeForecastTransformer(ModelConfig()); print('OK')"` succeeds
- [ ] Model forward pass with (B, L, 5) + (B, L, 5) produces correct output shapes

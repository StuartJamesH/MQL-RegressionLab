# Data Ingestion, Normalization & Windowing

> **Modules:** `Learn/v2/data.py` (core) · `training/pretrain_data.py` (multi-instrument)  
> **Key classes:** `SessionFeatureEncoder` · `MultiInstrumentDataset`

---

## Mental Map: From CSV Files to Model-Ready Tensors

```
                              ┌──────────────────────┐
                              │   CSV Files on Disk   │
                              │  BTCUSD_M5.csv        │
                              │  XAUUSD_H1.csv        │
                              │  EURUSD_M15.csv       │
                              │  ...                  │
                              └──────────┬───────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
           ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
           │ normalize_   │    │ Session      │    │ Label         │
           │ ohlcv()      │    │ Feature      │    │ Computation   │
           │              │    │ Encoder      │    │ (see LABELS   │
           │ OHLCV→log    │    │              │    │  doc)         │
           │ ratio        │    │ hour sin/cos │    │               │
           │ Vol→rolling  │    │ dow sin/cos  │    │ Forward       │
           │ median       │    │              │    │ returns/      │
           │              │    │              │    │ excursions    │
           └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
                  │                   │                    │
                  │  (n_bars, 5)      │  (n_bars, 4)       │  (n_bars, n_horizons)
                  │  float32          │  float32           │  float32
                  │                   │                    │
                  └───────────────────┼────────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │  create_sliding  │
                            │  _windows()      │
                            │                  │
                            │  Numba-compiled  │
                            │  O(n_windows)    │
                            │  overlapping     │
                            └────────┬─────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            X: (n_windows,     X_session:         Y: (n_windows,
               seq_len, 5)     (n_windows,           n_horizons)
               float64         seq_len, 4)
                               float64

         ┌─────────────────────────────────────────────────────┐
         │   Single Dataset Path (train_transformer.py)        │
         │   X_raw, X_sess, y → torch Dataset → DataLoader    │
         └─────────────────────────────────────────────────────┘

         ┌─────────────────────────────────────────────────────┐
         │   Multi-Instrument Path (pretrain_data.py)           │
         │   MultiInstrumentDataset → DataLoader                │
         │   (sampled proportional to instrument size)          │
         └─────────────────────────────────────────────────────┘
```

---

## Stage 1: Raw Data Loading

### Source Format

CSV files with OHLCV columns. The required columns are:

| Column | Type | Description |
|--------|------|-------------|
| `Time` | datetime string | Bar timestamp (used for session encoding) |
| `Open` | float | Opening price |
| `High` | float | Highest price in bar |
| `Low` | float | Lowest price in bar |
| `Close` | float | Closing price |
| `Volume` | float | Tick or real volume |

Files are located in `ModelWorkbench/data/` (gitignored). Generated via `fetch_datasets_bulk.py`.

### Loading Function

`load_ohlcv()` (from `Learn.train_utils`, shared with v1) reads a CSV, parses the `Time` column to datetime, sorts chronologically, and returns a DataFrame. The `--n-rows` CLI flag limits rows loaded for development iterations.

---

## Stage 2: OHLCV Normalization (`normalize_ohlcv`)

### Purpose

Raw OHLCV prices are non-stationary and instrument-dependent. We need a **scale-free, causal** representation that works across all instruments and time periods.

### Two-Part Transformation

**Part A — Log-Ratio Pricing (OHLC channels):**

For each bar at time `t`, all four OHLC channels are expressed as log-ratios to the **previous bar's close** (`Close[t-1]`):

```
normed_Open[t]  = ln(Open[t]  / Close[t-1])
normed_High[t]  = ln(High[t]  / Close[t-1])
normed_Low[t]   = ln(Low[t]   / Close[t-1])
normed_Close[t] = ln(Close[t] / Close[t-1])
```

**Why causal:** Uses only `Close[t-1]`, which is known at time `t`. This is the bar-to-bar log return for Close, and the relative positioning of O/H/L within that bar.

The first bar has no previous close, so its values are filled with `0.0` (neutral).

**Part B — Volume Scaling:**

Volume at bar `t` is divided by the rolling median volume over the last 252 bars:

```
normed_Volume[t] = Volume[t] / rolling_median(Volume[t-252 : t])
```

This produces a dimensionless relative-volume feature. When volume spikes above its recent norm, the value exceeds 1.0; when quiet, it falls below 1.0.

The 252-bar window uses `min_periods=1`, so early bars still get valid values (the first bar gets 1.0).

### Output Shape

`(n_bars, 5)` float32 array — columns: `[Open, High, Low, Close, Volume]`.

### Safety Guarantees

- All NaN/inf values are replaced with `0.0` via `np.nan_to_num()`
- Division-by-zero is guarded against with `eps = 1e-9`
- Empty DataFrames raise `ValueError`

---

## Stage 3: Session Feature Encoding (`SessionFeatureEncoder`)

### Purpose

Markets behave differently at different times of day and days of the week. A linear encoding of hour (0–23) or weekday (0–6) would incorrectly make 23:00 appear "far" from 00:00.

### Cyclical Encoding

Each timestamp is encoded as 4 continuous features using sine/cosine pairs:

```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
dow_sin  = sin(2π × dayofweek / 7)    (Monday=0, Sunday=6)
dow_cos  = cos(2π × dayofweek / 7)
```

This ensures:
- 23:00 and 00:00 produce nearly identical feature values (the circle closes)
- Monday and Sunday are close in feature space (weekend adjacency)
- Each pair uses 2 dimensions to represent a 1D circle — the model can learn to combine them

### Input Handling

- Accepts `pd.Series`, `np.ndarray`, or list of datetime-like values
- NaT values: filled with a Monday epoch sentinel for `.hour`/`.dayofweek` access, then the output features are masked back to NaN
- Empty input raises `ValueError`

### Output Shape

`(n_bars, 4)` float32 array — columns: `[hour_sin, hour_cos, dow_sin, dow_cos]`.

---

## Stage 4: Combining Channels

After normalization, the 5 OHLCV channels and 4 session channels are concatenated into a single feature matrix of shape `(n_bars, 9)`.

This combined 9-channel representation is what the `PatchEmbedding` receives:

```
    5 OHLCV channels          4 session channels
   ┌──────┬──────┬──────┬──────┬──────┐   ┌──────┬──────┬──────┬──────┐
   │  O   │  H   │  L   │  C   │  V   │   │ h_s  │ h_c  │ d_s  │ d_c  │
   └──────┴──────┴──────┴──────┴──────┘   └──────┴──────┴──────┴──────┘
                    │                                      │
                    └──────────────┬───────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  (n_bars, 9) feature matrix  │
                    └──────────────────────────────┘
```

In the model config, this split is preserved:
- `in_channels = 5` (OHLCV)
- `session_channels = 4` (cyclical time)
- `total_channels = 9` (computed property)

---

## Stage 5: Sliding Window Assembly (`create_sliding_windows`)

### Purpose

Transform the bar-aligned feature and label arrays into overlapping fixed-length windows that the model consumes as input sequences.

### Window Construction

```
Data:    [bar₀, bar₁, bar₂, bar₃, bar₄, bar₅, bar₆, bar₇, ...]
                   │                    │
         Window 0: ├──── seq_len ──────┤  → label at bar_{seq_len-1}
                   └── bars 0..(L-1) ──┘
                         │                    │
              Window 1:  ├──── seq_len ──────┤  → label at bar_{seq_len}
                         └── bars 1..L ──────┘
                              │                    │
                   Window 2:  ├──── seq_len ──────┤  → label at bar_{seq_len+1}
                              └── bars 2..(L+1) ─┘
```

### Causal Alignment Rule

**The label for window `w` is taken from bar `w + seq_len - 1`.**

The model sees bars `[w, w + seq_len - 1]` and must predict what happens after that window. The label at bar `w + seq_len - 1` was precomputed from forward data `[w + seq_len, w + seq_len + horizon]` — data the model has never seen.

This is the central causal contract: the features end at `t`, and the label starts at `t+1`.

### Implementation

- `_build_windows_nb()` — Numba `@njit` compiled kernel that extracts overlapping windows from the feature matrix
- `_extract_labels_nb()` — Numba kernel that extracts the matching label for each window
- Handles both 1-D labels (scalar per bar) and 2-D labels (vector per bar, e.g. per-horizon returns)

### Output Shapes

| Output | Shape | Description |
|--------|-------|-------------|
| `X` | `(n_windows, seq_len, 9)` | Feature windows (float64) |
| `y["name"]` | `(n_windows,)` or `(n_windows, d)` | Labels per window |

Where `n_windows = n_bars - seq_len + 1`.

### Validation

- Raises `ValueError` if `data` has fewer rows than `seq_len`
- Raises `ValueError` if any label array length doesn't match `n_bars`
- Only supports 1-D or 2-D label arrays (raises for 3-D+)

---

## Stage 6 (Alternative): Multi-Instrument Dataset for Pretraining

### Module: `training/pretrain_data.py`

Used during Phase 1 (self-supervised pretraining). Instead of processing a single CSV, this loads **all available instruments simultaneously**.

### `MultiInstrumentDataset` Construction

```
Constructor receives:  [ "data/BTCUSD_M5.csv",
                         "data/XAUUSD_H1.csv",
                         "data/EURUSD_M15.csv",
                         ...                     ]
                              │
                              ▼
                     For each CSV file:
                     ┌─────────────────────────────────────┐
                     │ 1. pd.read_csv(fp)                  │
                     │ 2. Sort by Time column              │
                     │ 3. _normalize_ohlcv_by_atr()        │
                     │    (EMA-centered, ATR-scaled OHLCV) │
                     │ 4. Count valid windows               │
                     │ 5. Infer timeframe_id from filename  │
                     │    M1→0, M5→1, M15→2, H1→4, D1→6   │
                     │ 6. Build flat window start indices   │
                     └─────────────────────────────────────┘
                              │
                              ▼
                     Flat index: (total_windows,) arrays
                       _flat_starts        — window start positions
                       _flat_instrument_ids — which instrument
                       _flat_timeframe_ids  — which timeframe
                       _sample_weights      — proportional to inst size
```

### Per-Instrument Normalization (`_normalize_ohlcv_by_atr`)

This differs from the single-dataset `normalize_ohlcv()` — it uses **EMA-centering + ATR scaling** rather than log-ratio pricing:

1. Compute rolling True Range and ATR (20-bar window)
2. Center each OHLC channel by its EMA: `(price - EMA) / ATR`
3. Optionally normalize Volume: `(volume - EMA_vol) / std_vol`
4. Clip to `[-10, 10]` to bound extreme values
5. Drop rows with any NaN (from insufficient ATR history)

This normalization is **per-instrument** — each instrument gets its own EMA/ATR statistics, ensuring cross-instrument comparability.

### Sampling Strategy

- Each instrument contributes windows proportional to its size (more bars = more sampling weight)
- `__getitem__()` returns `(features, features_clone, mask, instrument_id, timeframe_id)`
- For MAE pretraining, `features` and `features_clone` are identical (the mask is applied during training)
- `max_horizon` parameter ensures windows have enough forward bars for label computation

### Timeframe Inference

Timeframe IDs are parsed from CSV filenames using case-insensitive pattern matching:

| Pattern in filename | timeframe_id |
|--------------------|-------------|
| `M1` | 0 |
| `M5` | 1 |
| `M15` | 2 |
| `M30` | 3 |
| `H1` | 4 |
| `H4` | 5 |
| `D1` or `D` | 6 |
| `W1` or `W` | 7 |
| `MN1` | 8 |

These IDs are used for the optional `TimeframeEmbedding` in the model (adds a learned per-timeframe offset to patch embeddings when multi-timeframe fusion is enabled).

---

## Stage 7: Data Validation & Filtering

Before training (`train_transformer.py`):

1. **NaN filtering:** Windows containing any NaN in features or labels are dropped. NaN labels come from the tail of the data where not enough forward bars exist for the largest horizon.
2. **Train/validation split:** Random permutation split (default 80/20). For time-series integrity in production, use `PurgedWalkForwardSplit` from `training/folds.py`.
3. **Tensor conversion:** NumPy arrays are cast to `torch.float32` tensors for GPU training.

---

## `PurgedWalkForwardSplit` (Proper Temporal Splits)

For production-grade evaluation, `training/folds.py` provides purged walk-forward cross-validation following the methodology from *Advances in Financial Machine Learning*:

```
Fold 0:  [████████████████▐ train ▐  gap  ▐ test ▐·················]
Fold 1:  [███████████████████████████████▐  gap  ▐ test ▐············]
Fold 2:  [████████████████████████████████████████████▐  gap  ▐ test ▐]
```

**Key parameters:**
- `n_folds` — number of validation blocks
- `min_train_size` — minimum bars for initial training window
- `test_size` — bars per test block
- `gap_size` — purge gap (bars excluded from training on each side of test block)
- `rolling` — if True, use fixed-size lookback instead of expanding windows

The purge gap prevents label leakage: if a label at bar `t` looks `gap_size` bars into the future, then training bars within `gap_size` of the test block could indirectly see those future values.

---

## Summary: Data Flow in One Function Call

```python
# The complete data preparation for Phase 2 fine-tuning:
X_raw, X_sess, y = prepare_ohlcv_windows("BTCUSD_M5.csv", n_rows=500000, seq_len=512, target_type="log_return")
# X_raw:   (n_windows, 512, 5) — normalized log-ratio OHLCV
# X_sess:  (n_windows, 512, 4) — cyclical hour/dow encoding
# y:       (n_windows, 6)      — forward log returns at [5, 10, 20, 40, 60, 120] bars
```

All operations are strictly causal. The model sees bars 0..511 and predicts returns over bars 512..(512+h).

---
name: v2-trading-engine
description: Use when the user asks about running, launching, configuring, debugging, or extending the Learn/v2 transformer live trading engine. Triggers include v2 trading engine, live trading, MetaTrader5, model pack, run_v2.py, Engine/v2, ONNX inference, replay mode, TicketBook, and per-symbol launcher.
---

# v2 Trading Engine Skill

Instructions for using the live/replay trading runtime in `Engine/v2/`. This runtime consumes `ModelWorkbench/Learn/v2` transformer model packs, ingests bars from MetaTrader 5 (or CSV files), and emits managed trading signals.

## When to use this skill

Apply this skill whenever the user asks about:

- Launching or running the v2 live trading engine.
- Configuring `Engine/run_v2.py` or per-symbol launchers.
- Loading and using v2 model packs for inference.
- Choosing between ONNX and PyTorch inference backends.
- Running replay/backtest validation without live orders.
- Debugging signal generation, order lifecycle, or MT5 connectivity.
- Extending `Engine/v2/strategy.py`, `Engine/v2/inference.py`, or the data handlers.

## Architecture overview

The runtime follows the proven `Engine/` separation of concerns:

```
V2MT5DataHandler.get_next_bar()  (live or replay)
        ↓
V2SignalStrategy.on_bar(bar) → list[Order]
        ↓
V2MT5LiveExecutionHandler.submit_stop_order(order)
        ↓
process_pending_batch(bar_time)      → expire / detect fills
        ↓
process_position_updates_batch(bar_time) → detect SL/TP closures
```

- **No strategy code calls MT5 directly.** Broker interaction is centralized in the execution handler.
- **State is tracked in `TicketBook`** (in-memory cache + SQLite journal).
- **Inference uses ONNX Runtime by default**, with a PyTorch fallback from `model.pt`.

## Model pack requirements

A v2 model pack directory must contain these files (produced by `ModelWorkbench/Learn/v2/deploy.py`):

- `model.onnx` — ONNX model for inference (required for `--backend onnx`).
- `config.json` — `ModelConfig` architecture/hyperparameters.
- `normalizer.json` — input normalization metadata.
- `feature_spec.json` — feature computation specification.
- `model_info.json` — training metadata.
- `model.pt` — PyTorch checkpoint (required only for `--backend pytorch` or parity tests).

Pack path example:

```text
ModelWorkbench/ModelPacks/transformers/my_model/
```

## Quick start: live trading

From the **repo root** with the `.venv` activated:

```bash
.venv/bin/python Engine/run_v2.py \
  --symbol EURUSD \
  --pack-dir ModelWorkbench/ModelPacks/transformers/my_model \
  --magic 234001 \
  --db-path Engine/v2/ticketbook_EURUSD.db \
  --backend onnx \
  --mode live
```

### Important per-instance constraints

- `MAGIC` must be **unique** for every running symbol/strategy instance.
- `DB_PATH` must be **unique** for every running instance.
- Shared magic numbers or databases cause conflicting order operations and corrupted state.

## Per-symbol launcher (recommended for repeated use)

1. Copy the template to a git-ignored file:

   ```bash
   cp Engine/.run_v2_TEMPLATE.py Engine/run_v2_EURUSD.py
   ```

2. Edit `SYMBOL`, `MAGIC`, `PACK_DIR`, `DB_PATH`, and `BACKEND`.
3. Run:

   ```bash
   .venv/bin/python Engine/run_v2_EURUSD.py
   ```

Do **not** commit per-symbol launchers to version control.

## Replay validation (safe, no live orders)

### CSV replay (no MT5 required)

```bash
.venv/bin/python Engine/run_v2.py \
  --symbol EURUSD \
  --pack-dir ModelWorkbench/ModelPacks/transformers/my_model \
  --magic 234001 \
  --db-path Engine/v2/ticketbook_EURUSD_replay.db \
  --csv data/EURUSD_M1_260weeks.csv \
  --max-position-lots 0.1
```

### MT5 replay (requires running MT5 terminal)

```bash
.venv/bin/python Engine/run_v2.py \
  --symbol EURUSD \
  --pack-dir ModelWorkbench/ModelPacks/transformers/my_model \
  --magic 234001 \
  --db-path Engine/v2/ticketbook_EURUSD_replay.db \
  --mode replay \
  --start "2026-06-01" \
  --end "2026-06-02"
```

Replay mode simulates order lifecycle through `TicketBook` and writes a trade log CSV, but still submits orders to MT5 in MT5-replay mode (use CSV replay for fully offline validation).

## Inference backends

- **`--backend onnx`** (default): aligned with the MQL5 deployment path, smaller memory footprint, no PyTorch required at runtime.
- **`--backend pytorch`**: uses `model.pt`; useful for GPU inference or when ONNX providers are unavailable.

Verify both backends agree:

```python
from Engine.v2.inference import compare_backends

compare_backends(
    "ModelWorkbench/ModelPacks/transformers/my_model",
    sample_csv="data/EURUSD_M1_260weeks.csv",
    atol=1e-4,
)
```

## Key configuration parameters

| CLI flag | Default | Purpose |
|---|---|---|
| `--symbol` | required | MT5 instrument ticker |
| `--pack-dir` | required | Path to v2 model pack directory |
| `--magic` | required | Unique EA magic number |
| `--db-path` | required | Unique SQLite journal path |
| `--timeframe` | `M1` | Bar timeframe |
| `--risk` | `50.0` | Max account-currency risk per trade |
| `--patience` | `5` | Pending-order expiry in bars/minutes |
| `--signal-threshold` | `0.1` | Minimum `|signal|` to emit a trade |
| `--max-position-lots` | `0.5` | Hard cap on position size in lots |
| `--backend` | `onnx` | `onnx` or `pytorch` |
| `--device` | `cpu` | Inference device (`cpu` or `cuda`) |
| `--mode` | `live` | `live` or `replay` |
| `--csv` | `None` | CSV path for CSV replay |

## Logging and monitoring

- `trading.log` — root log with bar count, health metrics, and errors.
- `Engine/v2/Trade Logs/<symbol>_<model>_<timestamp>.csv` — per-bar trade log.
- Health metrics (buffer length, inference latency, last signal, last error) are emitted every 100 bars.

## Common troubleshooting

### `ModuleNotFoundError: No module named 'Learn'`

Ensure you run from the **repo root** and that `ModelWorkbench/` is on `sys.path`. `Engine/run_v2.py` and `Engine/v2/__init__.py` add it automatically, but imports fail if you run from inside `Engine/`.

### `ONNX model not found` or `INVALID_GRAPH`

Verify the pack contains `model.onnx`. Regenerate the pack with `ModelWorkbench/Learn/v2/deploy.py` if it is missing or corrupt.

### `mt5.initialize() failed`

- MetaTrader 5 terminal must be running and logged in.
- The target symbol must be visible in Market Watch.
- On Linux, MT5 runs under Wine; ensure the `MetaTrader5` Python package is installed in the same environment.

### Strategy never produces signals

- Check warm-up: the strategy needs `max_seq_len` bars (default 512) before inference starts.
- Check `--signal-threshold`; values too high suppress trades.
- Check the trade log CSV for `pending_order`/`open_position` gating.
- Verify the model pack was trained with the same timeframe/symbol characteristics as the live feed.

### Duplicate or missing bars

`V2MT5DataHandler` tracks `last_yield_time` and yields completed bars only. If you see gaps, check terminal timezone and ensure `refresh_rates` is returning updates.

### Position sizing looks wrong

`V2SignalStrategy` queries `mt5_executor.get_point_value(symbol)` to convert account-currency units to lots. If MT5 symbol info is unavailable, it logs a warning and falls back to `point_value=1.0`, which produces incorrect sizes. Ensure the symbol is active in Market Watch.

## Extending the runtime

- **Signal logic**: edit `Engine/v2/strategy.py::_signal_from_outputs()`.
- **Risk/position sizing**: edit `Engine/v2/strategy.py` or `Engine/v2/config.py::V2RiskConfig`.
- **Data source**: subclass `Engine.v2.data_handler.V2MT5DataHandler` or `V2CSVDataHandler`.
- **Execution behavior**: edit `Engine/v2/executor.py` (inherits from `Engine/Executor.py`).
- **Inference backend**: edit `Engine/v2/inference.py::V2InferenceEngine`.

## Safety checklist before live trading

1. Verify the model pack loads and backends agree (`compare_backends`).
2. Run CSV replay and inspect the trade log CSV.
3. Run MT5 replay on a recent date range.
4. Launch live with the **smallest allowed lot size** and monitor for one session.
5. Confirm `MAGIC` and `DB_PATH` are unique and not shared with another running instance.
6. Confirm `.env` does not contain secrets if the launcher directory is shared.

## Related files and docs

- `Engine/v2/README.md` — runtime documentation.
- `Engine/run_v2.py` — CLI launcher.
- `Engine/.run_v2_TEMPLATE.py` — per-symbol launcher template.
- `Engine/v2/strategy.py` — signal generation and order construction.
- `Engine/v2/inference.py` — ONNX/PyTorch inference engine.
- `Engine/v2/data_handler.py` — live and CSV data handlers.
- `Engine/v2/engine.py` — per-bar orchestrator.
- `ModelWorkbench/Learn/v2/README.md` — training the transformer.
- `ModelWorkbench/Learn/v2/deploy.py` — packaging model packs.
- `Engine/ARCHITECTURE.md` — legacy engine architecture (still relevant for lifecycle patterns).

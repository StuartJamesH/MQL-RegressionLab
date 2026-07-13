# Engine/v2 — Live Trading Runtime for Learn.v2 Transformer Packs

This package is the live/replay trading runtime for model packs produced by
[`ModelWorkbench/Learn/v2/deploy.py`](../ModelWorkbench/Learn/v2/deploy.py).
It replaces the legacy LSTM/TCN/PyTorch path with the Causal Patch Transformer
pipeline and uses **ONNX Runtime** as the primary inference backend.

## Directory layout

```
Engine/v2/
  __init__.py          # package marker; puts ModelWorkbench/ on sys.path
  model_pack.py        # ModelPackLoader — reads deployment artifacts
  inference.py         # V2InferenceEngine — ONNX primary + PyTorch fallback
  features.py          # causal live normalization / session encoding wrappers
  strategy.py          # V2SignalStrategy — signals, risk, order construction
  data_handler.py      # V2MT5DataHandler (live/replay) + V2CSVDataHandler
  executor.py          # V2MT5LiveExecutionHandler — wraps Engine/Executor.py
  engine.py            # V2LiveEngine — per-bar orchestrator
  config.py            # V2RuntimeConfig dataclass
  README.md            # this file
  Trade Logs/          # runtime CSV trade logs (gitignored)
```

## Requirements

* Python 3.11+ with the repo `.venv` activated.
* `onnxruntime` installed (`pip install onnxruntime`).
* `MetaTrader5` Python package installed for live or MT5 replay mode.
* A v2 model pack directory containing:
  * `model.onnx`
  * `config.json`
  * `normalizer.json`
  * `feature_spec.json`
  * `model_info.json`
  * `model.pt` (required only for the PyTorch fallback / parity tests)

## How to train and package a model

From `ModelWorkbench/`:

```bash
# Train (example; see ModelWorkbench/Learn/v2/README.md for full options)
python -m Learn.v2.training.train \
  --dataset data/BTCUSD_M5_260weeks.csv \
  --epochs 50

# Package the best checkpoint for deployment
python -m Learn.v2.deploy \
  --checkpoint runs/my_run/best.pt \
  --output-dir ModelWorkbench/ModelPacks/transformers/my_model
```

The pack directory must contain the six files listed above.

## How to launch live trading

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

* `MAGIC` must be **unique** for every running symbol/strategy instance.
* `DB_PATH` must be **unique** for every running instance.
  Collisions will cause pending-order and open-position state to be shared
  incorrectly between bots.

For repeated launches, copy `Engine/.run_v2_TEMPLATE.py` to a git-ignored
file such as `Engine/run_v2_EURUSD.py`, fill in the placeholders, and run that.
Do **not** commit per-symbol launchers.

## How to run replay validation

### MT5 replay

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

### CSV replay (no MT5 required)

```bash
.venv/bin/python Engine/run_v2.py \
  --symbol EURUSD \
  --pack-dir ModelWorkbench/ModelPacks/transformers/my_model \
  --magic 234001 \
  --db-path Engine/v2/ticketbook_EURUSD_replay.db \
  --csv data/EURUSD_M5_260weeks.csv \
  --max-position-lots 0.1
```

The CSV path is relative to the repo root.  The runtime will replay every bar,
generate signals, simulate order lifecycle through `TicketBook`, and write a
trade log CSV to `Engine/v2/Trade Logs/`.

## Inference backends

* **ONNX** (`--backend onnx`) — default, aligned with the MQL5 deployment path,
  smaller memory footprint.
* **PyTorch** (`--backend pytorch`) — fallback requiring `model.pt` in the pack;
  useful for GPU inference or when ONNX providers are unavailable.

Run the parity helper to verify both backends agree:

```python
from Engine.v2.inference import compare_backends

compare_backends(
    "ModelWorkbench/ModelPacks/transformers/my_model",
    sample_csv="data/EURUSD_M5_260weeks.csv",
    atol=1e-4,
)
```

## Architecture notes

The runtime follows the proven Engine pattern:

```
V2MT5DataHandler.get_next_bar()
      ↓
V2SignalStrategy.on_bar(bar) → list[Order]
      ↓
V2MT5LiveExecutionHandler.submit_stop_order(order)
      ↓
process_pending_batch(bar_time) → detect fills / expire stale orders
      ↓
process_position_updates_batch(bar_time) → detect SL/TP closures
```

Strategy code never calls the MT5 API directly; all broker interaction is
routed through the execution handler.

## Logging and monitoring

* `trading.log` — root log with bar count, health metrics, and errors.
* `Engine/v2/Trade Logs/<symbol>_<model>_<timestamp>.csv` — per-bar trade log
  containing signal, forecast moments, order fields, and exposure state.
* Health metrics (buffer length, last inference latency, last signal, last
  error) are emitted every 100 bars.

## Shutdown

`SIGINT` / `SIGTERM` are caught by the launcher, which calls
`V2LiveEngine.shutdown()`.  This cleanly closes:

* the MT5 terminal connection (`mt5.shutdown()`),
* strategy trade-log files,
* the per-bar loop at the next bar boundary.

# AGENTS.md — MQL-RegressionLab

> **WARNING:** `ModelWorkbench/README.md` is **stale** — it describes older LSTM/TCN training, an
> `Engine/` directory, and notebooks that no longer exist (2_0_a, 3_0, 4_0). Trust this file,
> the actual `.ipynb`/`.py` files on disk, and the `.gitignore` — not the README.

## Repo overview

Two-layer research pipeline: Python ML training in `ModelWorkbench/` + MQL5 indicators in `MQL5/Indicators/`
that replicate the same logic for live execution. This repo is the offline training workspace; the live
engine lives in a separate `Engine` project (not in this repo).

## Setup

One-time environment creation (repo root):

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt

Dependencies include LightGBM, pandas, scikit-learn, PyTorch, Numba, TA-Lib (system library required),
and ctrader-open-api (Twisted-based). TA-Lib must be installed at the OS level before pip will succeed.

## Working directory

All scripts and imports assume `ModelWorkbench/` is the CWD. The `Learn/` package is accessed via
`from Learn.* import ...` — no namespace package, no `sys.path` tricks. Run scripts from repo root
using the venv at the repo root:

```
# Linux
.venv/bin/python ModelWorkbench/train_lgbm.py --ds-name data/BTCUSD_M5_260weeks.csv ...

# or cd into ModelWorkbench first:
cd ModelWorkbench && ../.venv/bin/python fetch_datasets_bulk.py
```

## Environment & auth

`.env` at repo root holds cTrader API credentials (`CLIENT_ID`, `SECRET`, `ACCESS_TOKEN`, `ACCOUNT_ID`).
It is gitignored. `Learn/data.py` calls `load_dotenv()` internally, so it just works if `.env` exists.

## Data

- `data/*.csv` is gitignored. Regenerate via `fetch_datasets_bulk.py` (run from `ModelWorkbench/`).
- Twisted reactor (used by the cTrader API) can only start **once per Python process**. Restart the
  kernel if re-running data downloads in a Jupyter session.

## Architecture

```
ModelWorkbench/
  Learn/           — shared Python library (features, labels, preprocess, data fetching, utils)
    features.py    — add_all_features(), add_feature_library(), MTF indicators (~100+ features)
    labels.py      — causal_triple_barrier_hilow_trend_labeler(), causal_market_regime(),
                      calculate_trade_outcomes_capped() (Numba-compiled)
    preprocess.py  — preprocess_ohlcv() with RobustScaler, one-hot passthrough
    data.py        — fetch_ohlcv(), fetch_ohlcv_bulk() via cTrader Open API
    train_utils.py — walk_forward_folds, tune_lgbm_by_spearman, reg_metrics, safe_spearman
    feature_codegen.py — generates standalone feature modules, written to ModelPacks/generated_features/
    Util.py        — TwoHeadMLP, train_two_head_mlp, predict_two_head_mlp
  *.ipynb          — interactive exploration notebooks (1.0 → 1.1 → 1.1.b → 1.1.c → 2.1 → 3.1)
  train_lgbm.py    — CLI-driven LGBM regression training (two-head design)
  fetch_datasets_bulk.py — download OHLCV CSVs for all configured symbols
  ModelPacks/      — output: trained model packs (*.pkl) and generated feature modules

MQL5/
  Indicators/
    RegimeMA.mq5   — MQL5 replica of causal_market_regime()
    DonchianTrend.mq5 — MQL5 replica of donchian_trend()
```

## Notebook workflow (sequential)

1. `1.0 Get Historical Data` — download OHLCV CSVs
2. `1.1 Regression Lab` — tune labelling parameters (the most critical step)
3. `1.1.b Feature Lab - LGBM Regression` — feature selection via LightGBM
4. `1.1.c Feature Parity Check` — verify bulk vs sliding-window features are identical
5. `2.1 Train LGBM Regression Model` — final model training
6. `3.1 Backtest LGBM Regression` — offline P&L simulation on held-out data

## `train_lgbm.py` invocation

From repo root:
```
.venv/bin/python ModelWorkbench/train_lgbm.py --ds-name data/BTCUSD_M5_260weeks.csv --n-rows 500000
```

Output: `ModelWorkbench/ModelPacks/<dataset>_LightGBM_<date>_spearman_prod_TP<tp>_SL<sl>_model.pkl`

MQL5 indicators in `MQL5/Indicators/` must stay in parity with their Python counterparts in
`ModelWorkbench/Learn/`. When you change label logic or feature formulas in Python, check whether
the corresponding MQL5 `.mq5` file needs updating.

## Key style notes

- Notebooks are the primary exploration interface; `train_lgbm.py` is the batch-training entry point.
- All label/feature computation is strictly **causal** (no lookahead). Multi-timeframe features are
  shifted by one completed HTF bar before merging.
- The feature codegen pipeline (`feature_codegen.py`) writes hash-named standalone modules to
  `ModelPacks/generated_features/`. These are what get exported in model packs — not the generic
  `add_feature_library()`.
- This repo has **no CI**, **no tests**, **no linter config**, and **no `pyproject.toml`**.
  Dependencies are managed via `requirements.txt` into the root `.venv`.

---
goal: Commit Plan for feature/transformer Branch
version: 1.0
date_created: 2026-07-13
owner: MQL-RegressionLab
status: 'Planned'
tags: [git, commit-plan, feature/transformer]
---

# Commit Plan: `feature/transformer`

## Branch status snapshot

- **Branch:** `feature/transformer`
- **Modified files:** 2
- **Untracked files/directories:** ~70 files across 16 top-level paths
- **Total effective changes:** v2 transformer training pipeline + MQL5 deployment + live trading engine + opencode config + plans/docs

## Pre-commit cleanup checklist

Run these steps once before executing any commits to avoid committing generated artifacts:

1. Remove Python cache:
   ```bash
   find Engine -type d -name __pycache__ -exec rm -rf {} +
   find ModelWorkbench -type d -name __pycache__ -exec rm -rf {} +
   find .opencode -type d -name __pycache__ -exec rm -rf {} +
   ```
2. Remove runtime artifacts:
   ```bash
   rm -f Engine/v2/Trade\ Logs/*.csv
   rm -f Engine/v2/ticketbook_*.db
   rm -f trading*.log
   ```
3. Verify `.gitignore` ignores `__pycache__/`, `.venv/`, `.env`, `data/`, `ModelPacks/`, and `Engine/v2/.gitignore` ignores `Trade Logs/*.csv`, `ticketbook_*.db`, `*.log`.
4. Run tests:
   ```bash
   .venv/bin/python -m pytest Engine/tests/ ModelWorkbench/tests/v2/ -q
   ```

## Proposed commit sequence

Commits are ordered so that foundational changes (dependencies, training code) come before dependent layers (MQL5 deployment, live engine, config). Each commit is independently coherent.

---

### Commit 1 — Dependencies

**Message:**
```text
deps: add h5py, MetaTrader5, and onnxruntime to requirements.txt

- h5py: required by v2 data pipelines
- MetaTrader5: required for live MT5 market data and order execution
- onnxruntime: required for ONNX inference in Engine/v2 and MQL5 parity
```

**Files:**
- `requirements.txt`

**Rationale:** Keep dependency changes isolated and at the start so downstream commits can assume the environment is correct.

---

### Commit 2 — v2 Transformer Training Pipeline

**Message:**
```text
feat: add Learn/v2 causal patch transformer training pipeline

- Add ModelWorkbench/Learn/v2/ transformer model, heads, losses, data utils,
  labels, signals, risk management, position sizing, backtesting, and training.
- Add ModelWorkbench/train_transformer.py CLI entry point.
- Add ModelWorkbench/tests/v2/ unit tests for model, labels, signals, backtest.
- Add ModelWorkbench/docs/ design and implementation notes.
```

**Files:**
- `ModelWorkbench/Learn/v2/`
- `ModelWorkbench/train_transformer.py`
- `ModelWorkbench/tests/v2/`
- `ModelWorkbench/docs/`

**Rationale:** This is the largest standalone unit of work. Committing it separately keeps the live-engine commit focused on runtime concerns.

---

### Commit 3 — MQL5 Deployment

**Message:**
```text
feat: add MQL5 ONNX transformer indicator and expert advisor

- Add MQL5/Indicators/TransformerModel.mq5 for ONNX model inference in MT5.
- Add MQL5/Experts/TransformerTrader.mq5 for automated trading based on
  transformer signals.
```

**Files:**
- `MQL5/Indicators/TransformerModel.mq5`
- `MQL5/Experts/TransformerTrader.mq5`

**Rationale:** MQL5 files are a distinct deployment target and should not be mixed with Python training code.

---

### Commit 4 — Training Plans and Revision Notes

**Message:**
```text
docs: add training plans, revision notes, and training log

- Add TP/SL grid implementation plan.
- Add major-revision.md and minor-uplift.md revision tracking.
- Add TRAINING_LOG.md for experiment records.
```

**Files:**
- `ModelWorkbench/plans/TP_SL_GRID_IMPLEMENTATION_PLAN.md`
- `ModelWorkbench/TRAINING_LOG.md`
- `ModelWorkbench/major-revision.md`
- `ModelWorkbench/minor-uplift.md`

**Rationale:** Documentation/planning artifacts are grouped together so the feature branch's design rationale is traceable.

---

### Commit 5 — v2 Live Trading Engine

**Message:**
```text
feat: add Engine/v2 live trading runtime for transformer model packs

- Add Engine/v2/ package: model pack loader, ONNX/PyTorch inference engine,
  causal feature normalization, signal strategy, MT5 data handler, executor,
  per-bar orchestrator, and runtime config.
- Add Engine/run_v2.py CLI launcher and Engine/.run_v2_TEMPLATE.py per-symbol
  launcher template.
- Add Engine/tests/test_v2_*.py unit tests and Engine/conftest.py.
- Update Engine/__init__.py to support top-level imports.
```

**Files:**
- `Engine/`

**Rationale:** The live engine is the consumer of the v2 pipeline and MQL5 deployment. It depends on commits 1–3 but is large enough to warrant its own commit.

---

### Commit 6 — Live Trading Implementation Plan

**Message:**
```text
docs: add live trading v2 implementation plan

- Add plan/feature-live-trading-v2-1.md with phased tasks for model pack
  regeneration, MT5 data ingestion, signal generation, and order lifecycle.
```

**Files:**
- `plan/feature-live-trading-v2-1.md`

**Rationale:** The implementation plan documents the design decisions behind Commit 5 and should be versioned alongside the code it describes.

---

### Commit 7 — Opencode Agents and Skills

**Message:**
```text
config: add opencode agents and v2 trading engine skill

- Add .opencode/agents/data-scientist.md and pytorch-patterns.md agent defs.
- Add .opencode/skills/model-training/SKILL.md.
- Add .opencode/skills/v2-trading-engine/SKILL.md for using the new runtime.
```

**Files:**
- `.opencode/agents/data-scientist.md`
- `.opencode/agents/pytorch-patterns.md`
- `.opencode/skills/model-training/SKILL.md`
- `.opencode/skills/v2-trading-engine/SKILL.md`

**Rationale:** Opencode configuration is orthogonal to trading code and should be committed separately so it can be reverted without affecting runtime behavior.

---

### Commit 8 — Notebooks

**Message:**
```text
feat: update LGBM notebook and add SL/TP grid search notebook

- Update ModelWorkbench/2.1 Train LGBM Regression Model.ipynb with latest
  experimental cells.
- Add ModelWorkbench/2.2 SL TP Grid Search.ipynb for stop/take-profit search.
```

**Files:**
- `ModelWorkbench/2.1 Train LGBM Regression Model.ipynb`
- `ModelWorkbench/2.2 SL TP Grid Search.ipynb`

**Rationale:** Notebooks are large, often contain outputs, and change frequently. Keep them in a final commit so they do not pollute the diffs of core code commits.

**Note:** Review the notebooks for embedded outputs and cell execution counts before committing. Consider clearing outputs if the repo policy is to keep notebooks source-only.

---

## Execution commands

Run the commits in order. Adjust messages if the repo uses a different commit convention.

```bash
# 1. Dependencies
git add requirements.txt
git commit -m "deps: add h5py, MetaTrader5, and onnxruntime to requirements.txt"

# 2. v2 training pipeline
git add ModelWorkbench/Learn/v2 ModelWorkbench/train_transformer.py ModelWorkbench/tests/v2 ModelWorkbench/docs
git commit -m "feat: add Learn/v2 causal patch transformer training pipeline"

# 3. MQL5 deployment
git add MQL5/Indicators/TransformerModel.mq5 MQL5/Experts/TransformerTrader.mq5
git commit -m "feat: add MQL5 ONNX transformer indicator and expert advisor"

# 4. Plans and logs
git add ModelWorkbench/plans/TP_SL_GRID_IMPLEMENTATION_PLAN.md ModelWorkbench/TRAINING_LOG.md ModelWorkbench/major-revision.md ModelWorkbench/minor-uplift.md
git commit -m "docs: add training plans, revision notes, and training log"

# 5. Live engine
git add Engine/
git commit -m "feat: add Engine/v2 live trading runtime for transformer model packs"

# 6. Implementation plan
git add plan/feature-live-trading-v2-1.md
git commit -m "docs: add live trading v2 implementation plan"

# 7. Opencode config
git add .opencode/agents/data-scientist.md .opencode/agents/pytorch-patterns.md .opencode/skills/model-training/SKILL.md .opencode/skills/v2-trading-engine/SKILL.md
git commit -m "config: add opencode agents and v2 trading engine skill"

# 8. Notebooks
git add "ModelWorkbench/2.1 Train LGBM Regression Model.ipynb" "ModelWorkbench/2.2 SL TP Grid Search.ipynb"
git commit -m "feat: update LGBM notebook and add SL/TP grid search notebook"
```

## Post-commit verification

After all commits:

1. Confirm clean working tree:
   ```bash
   git status --short
   ```
2. Confirm commit graph:
   ```bash
   git log --oneline -10
   ```
3. Run the test suite:
   ```bash
   .venv/bin/python -m pytest Engine/tests/ ModelWorkbench/tests/v2/ -q
   ```

## Risks and notes

- **Notebook outputs:** `2.1 Train LGBM Regression Model.ipynb` is modified and may contain large outputs. Clear outputs before committing if the repo prefers source-only notebooks.
- **Engine/ is fully untracked:** Verify no secrets, logs, or large binary files exist under `Engine/` before adding the entire directory.
- **Commit granularity:** If the branch will be squash-merged, this 8-commit plan can be collapsed into fewer logical commits. If it will be rebased and kept, 8 commits provide clear history.

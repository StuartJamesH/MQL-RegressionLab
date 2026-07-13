"""
Engine/v2 — Live trading runtime for Learn.v2 transformer model packs.

This package provides a self-contained runtime that loads a v2 deployment
pack (ONNX + JSON artifacts), ingests live or replay bars from MetaTrader 5,
and emits managed trading signals through the proven Engine order lifecycle.

All modules are designed to run from the repository root with the existing
``.venv`` and assume ``ModelWorkbench/`` is on ``sys.path`` so that
``from Learn.v2...`` imports resolve.
"""
from __future__ import annotations

import os
import sys

# Ensure the training library is importable when running from repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_WORKBENCH = os.path.join(_REPO_ROOT, "ModelWorkbench")
if _MODEL_WORKBENCH not in sys.path:
    sys.path.insert(0, _MODEL_WORKBENCH)

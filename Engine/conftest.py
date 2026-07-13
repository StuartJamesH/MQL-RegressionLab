"""
Pytest configuration for Engine tests.

Ensures ModelWorkbench/ is on sys.path so ``from Learn.v2...`` imports resolve
when tests are executed from the repo root.
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_WORKBENCH = os.path.join(_REPO_ROOT, "ModelWorkbench")
if _MODEL_WORKBENCH not in sys.path:
    sys.path.insert(0, _MODEL_WORKBENCH)

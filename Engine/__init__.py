"""
Engine package — live trading runtime components.

This marker makes ``Engine`` a regular Python package so subpackages such as
``Engine.v2`` and ``Engine.tests`` can be imported consistently by launchers,
tests, and tooling.

The legacy modules inside ``Engine/`` (``DataHandler.py``, ``Executor.py``,
``TicketBook.py``, ``Engine.py``) use top-level absolute imports such as
``from DataHandler import Order``.  Adding the ``Engine/`` directory to
``sys.path`` preserves that legacy import style when the package is imported
from the repo root.
"""
from __future__ import annotations

import os
import sys

_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

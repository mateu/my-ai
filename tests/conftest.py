"""Test configuration and helpers.

Ensures the project root (where cli.py and the my_ai package live) is on sys.path
so tests can reliably import them regardless of how pytest is invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root is the parent of the tests/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Prepend to sys.path so it takes precedence over any installed packages.
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

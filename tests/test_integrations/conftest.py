"""Ensure src/ is on sys.path for the integrations package (pre-pip-reinstall)."""
from __future__ import annotations

import sys
from pathlib import Path

# Add the src directory so Python finds agent_eval.integrations even when
# the installed package in site-packages was built before the integrations/
# subdirectory was created.
_SRC = str(Path(__file__).parent.parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

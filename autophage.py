#!/usr/bin/env python3
"""
AutoPhage 2.0 CLI Entry Point.
Forwarding runner for MACRO autonomous maintenance agent.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SKILLS_MAINT = _REPO_ROOT / "skills" / "maintenance"
if str(_SKILLS_MAINT) not in sys.path:
    sys.path.insert(0, str(_SKILLS_MAINT))

from skills.maintenance.macro.cli import main

if __name__ == "__main__":
    sys.exit(main())

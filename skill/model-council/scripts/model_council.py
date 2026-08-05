#!/usr/bin/env python
"""Hermes Skill launcher for the D-drive project checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

project = Path(
    os.environ.get("MODEL_COUNCIL_HOME", r"D:\Projects\hermes-model-council")
)
if not (project / "model_council" / "cli.py").is_file():
    raise SystemExit(
        "Hermes Model Council source was not found. Set MODEL_COUNCIL_HOME "
        f"or clone the project to {project}."
    )
sys.path.insert(0, str(project))

from model_council.cli import main

raise SystemExit(main())

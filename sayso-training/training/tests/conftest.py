
from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import settings

TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent

for path in (str(TRAINING_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

settings.register_profile("sayso", derandomize=True, deadline=None, database=None)
settings.load_profile("sayso")

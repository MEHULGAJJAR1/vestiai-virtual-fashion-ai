#!/usr/bin/env python
"""Root-level launcher for VestiAI (thin wrapper around ``backend.main``).

    python run.py                      # http://localhost:8000
    python run.py --port 9000 --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

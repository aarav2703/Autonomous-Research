"""Run the FastAPI backend for Stage 7."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


if __name__ == "__main__":
    uvicorn.run(
        "autonomous_multi_hop_research_agent.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

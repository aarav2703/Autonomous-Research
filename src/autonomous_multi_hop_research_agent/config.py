"""Shared project configuration."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RETRIEVAL_ARTIFACTS_DIR = ARTIFACTS_DIR / "retrieval"
HOTPOTQA_RETRIEVAL_DIR = RETRIEVAL_ARTIFACTS_DIR / "hotpotqa_distractor"


def get_llm_settings() -> dict[str, str]:
    """Load provider settings for the Stage 4 grounded generator."""
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    return {
        "api_key": api_key,
        "base_url": os.getenv("LLM_API_BASE_URL", "https://api.deepseek.com"),
        "model": os.getenv("LLM_MODEL", "deepseek-chat"),
    }

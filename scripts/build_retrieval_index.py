"""Stage 2 index builder for dense paragraph retrieval."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if os.name == "nt":
    # Avoid Windows OpenMP runtime conflicts between torch and numerical deps.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from autonomous_multi_hop_research_agent.retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    build_retrieval_artifacts,
    load_paragraph_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index over HotpotQA paragraph chunks.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional path to the paragraph parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for the FAISS index and metadata parquet.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paragraphs = load_paragraph_chunks(input_path=args.input_path)
    artifacts = build_retrieval_artifacts(
        paragraphs=paragraphs,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    print("Stage 2 index build completed.")
    print(f"Model: {artifacts.model_name}")
    print(f"Embedding dimension: {artifacts.embedding_dim}")
    print(f"Chunk count: {artifacts.chunk_count}")
    print(f"Metadata path: {artifacts.metadata_path}")
    print(f"FAISS index path: {artifacts.index_path}")
    print(f"Retrieval config path: {artifacts.config_path}")


if __name__ == "__main__":
    main()
